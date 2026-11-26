# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The tuned kernel tables TokenSpeed adds to FlashInfer's split MegaMoE.

Covers the shipped data file, its installation into FlashInfer's table lookup
and, where a FlashInfer with the split kernels is importable, that lookup
itself. No case needs a GPU.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120_split as split

# DeepSeek-V4.1-Flash: hidden, gate + up intermediate, top-k, expert slots.
DSV41 = (5120, 4608, 6, 384)
EP16 = "ep16_2x2numa"
# RTX 6000D: SMs, smallest Green Context partition, partition alignment.
SM156 = (156, 8, 8)
# What FlashInfer keys a table without a device by.
SM110 = (110, 8, 8)

# Row buckets of FlashInfer's adapter that a decode workspace selects, and one
# far above the usual anchors.
ROW_BUCKETS = (7, 16, 32, 64, 128, 168, 256, 2048)

# The knobs FlashInfer's tuned_table.overrides_for reads.
INTEGER_KNOBS = {
    "k1_n",
    "k2_n",
    "k1_ctas_per_sm",
    "k2_launch_factor",
    "k1_stages",
    "weight_prefetch_ahead",
    "dispatch_chunk_tokens",
    "k1_t0_prefetch",
    "k2_warps",
    "k2_ready_queue_bundle",
}
SWITCH_KNOBS = {"k2_tail_reclaim", "fc1_utility_postprocess", "push_dispatch"}
KNOBS = INTEGER_KNOBS | SWITCH_KNOBS | {"backend", "weight_eviction_policy", "sms"}


def _shipped() -> dict:
    return json.loads(split._TUNED_TABLES_PATH.read_text())


def _shipped_entries():
    for table in _shipped()["tables"]:
        for entry in table["entries"]:
            yield table, entry


def _key(table: dict, legacy_device=SM110):
    return (
        tuple(table["signature"]),
        table["topology"],
        tuple(table.get("device", legacy_device)),
    )


def test_shipped_file_is_one_the_installer_accepts():
    data = _shipped()
    assert data["schema"] == 1
    assert split._read_tuned_tables(split._TUNED_TABLES_PATH) == data["tables"]
    for table in data["tables"]:
        assert all(type(n) is int and n > 0 for n in table["signature"])
        assert len(table["signature"]) == 4
        assert isinstance(table["topology"], str) and table["topology"]
        assert all(type(n) is int and n > 0 for n in table["device"])
        assert len(table["device"]) == 3
    assert (DSV41, EP16, SM156) in [_key(table) for table in data["tables"]]


def test_shipped_tables_are_final():
    # A table taken from a tuning run that has not finished says so in its
    # provenance. It serves to try the mechanism out, not a release and not a
    # measurement: replace it with the finished run's table.
    status = _shipped().get("provenance", {}).get("status", "")
    assert not status.startswith("provisional"), status


def test_shipped_anchors_ascend():
    for table in _shipped()["tables"]:
        anchors = [entry["tokens"] for entry in table["entries"]]
        assert anchors, table["topology"]
        assert all(type(tokens) is int and tokens > 0 for tokens in anchors)
        assert all(low < high for low, high in zip(anchors, anchors[1:])), anchors


def test_shipped_knobs_are_ones_flashinfer_reads():
    for table, entry in _shipped_entries():
        knobs = entry["knobs"]
        where = (table["topology"], entry["tokens"])
        assert isinstance(knobs, dict), where
        assert set(knobs) <= KNOBS == split._TABLE_KNOBS, where
        for name in knobs.keys() & INTEGER_KNOBS:
            assert type(knobs[name]) is int, (where, name)
        for name in knobs.keys() & SWITCH_KNOBS:
            assert knobs[name] in ("on", "off"), (where, name)
        assert knobs.get("weight_eviction_policy") in (None, "default", "evict_first")


def test_shipped_cross_numa_entries_name_the_split_transport():
    # FlashInfer applies an entry of these topologies only to the transport it
    # names, and the adapter runs p2p_split on them.
    for table, entry in _shipped_entries():
        topology = table["topology"]
        if topology == "ep8_2numa" or topology.startswith("ep16_"):
            assert entry["knobs"].get("backend") == "p2p_split", (
                topology,
                entry["tokens"],
            )


def test_shipped_sm_partitions_cover_their_device():
    # [K1, K2] or [K1, TX, RX, K2]. FlashInfer wants the device's SMs covered:
    # every partition but the last a multiple of the alignment, K2 the rest.
    # The P2P transports have no TX and RX partitions.
    for table, entry in _shipped_entries():
        sms = entry["knobs"].get("sms")
        if sms is None:
            continue
        where = (table["topology"], entry["tokens"], sms)
        num_sms, min_partition, alignment = table["device"]
        assert len(sms) in (2, 4), where
        assert all(type(count) is int and count >= 0 for count in sms), where
        assert all(count % alignment == 0 for count in sms[:-1]), where
        assert min(sms[0], sms[-1]) >= min_partition, where
        assert sum(sms) == sms[0] + sms[-1] == num_sms, where


OURS = {
    "model": "dsv41",
    "signature": list(DSV41),
    "topology": EP16,
    "device": list(SM156),
    "entries": [
        {"tokens": 16, "knobs": {"backend": "p2p_split", "k1_stages": 3}},
        {"tokens": 128, "knobs": {"backend": "p2p_split"}},
    ],
}
# A table FlashInfer has for the same model and topology on its 110-SM device.
THEIRS = {
    "model": "dsv41",
    "signature": list(DSV41),
    "topology": EP16,
    "entries": [{"tokens": 16, "knobs": {"backend": "p2p_split", "k1_n": 16}}],
}


# Layouts of the stand-in FlashInfer, as its tuned_table module keeps them:
# (EP size, GPUs per P2P domain) of the groups its adapter folds into one P2P
# group, and (EP size, peers in the P2P domain, peers behind the network).
SPLIT_LAYOUTS = {(8, 4): "ep8_2numa", (16, 4): EP16}
REAL_LAYOUTS = {(4, 3, 0): "ep4_1numa", (16, 3, 12): EP16}


@pytest.fixture
def our_file(tmp_path, monkeypatch):
    """Make a file ours and the next install the first of its process."""

    def write(content) -> Path:
        path = tmp_path / "tuned.json"
        if content is not None:
            path.write_text(
                content if isinstance(content, str) else json.dumps(content)
            )
        monkeypatch.setattr(split, "_TUNED_TABLES_PATH", path)
        monkeypatch.setattr(split, "_installed_tuned_tables", None)
        monkeypatch.delenv(split._DISABLE_TUNED_TABLES_ENV, raising=False)
        return path

    return write


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Stand in for FlashInfer's kernel selection; returns what it holds.

    ``tables`` is the list its lookup reads, ``resolved`` the (shape,
    overrides) pairs it was asked to resolve, ``cleared`` the selection caches
    that were dropped. It cannot resolve an entry with zero K1 stages.
    """

    def fake(tables: list, cached: bool = True, **attributes) -> SimpleNamespace:
        data = {"schema": 1, "tables": tables}
        state = SimpleNamespace(tables=tables, resolved=[], cleared=[])

        def select_config_and_dispatch(shape, overrides):
            if overrides.get("k1_stages") == 0:
                raise ValueError("K1 needs a pipeline stage")
            state.resolved.append((shape, overrides))

        def select_compile_spec(**problem):
            raise AssertionError("an install compiles nothing")

        if cached:
            select_config_and_dispatch.cache_clear = lambda: state.cleared.append(
                "select_config_and_dispatch"
            )
            select_compile_spec.cache_clear = lambda: state.cleared.append(
                "select_compile_spec"
            )
        tuned_table = SimpleNamespace(
            **{
                "_load": lambda: data,
                "LEGACY_DEVICE": SM110,
                "_SPLIT_LAYOUTS": SPLIT_LAYOUTS,
                "_REAL_LAYOUTS": REAL_LAYOUTS,
                "overrides_for": lambda knobs: {
                    name: value for name, value in knobs.items() if name != "backend"
                },
                **attributes,
            }
        )
        for name, value in attributes.items():
            if value is None:
                delattr(tuned_table, name)
        heuristic = SimpleNamespace(
            MegaMoEHeuristicInput=lambda **fields: fields,
            MegaMoEHeuristicOverrides=lambda **fields: fields,
            select_config_and_dispatch=select_config_and_dispatch,
        )
        api = SimpleNamespace(select_compile_spec=select_compile_spec)
        monkeypatch.setattr(
            split, "_flashinfer_selection", lambda: (tuned_table, heuristic, api)
        )
        return state

    return fake


def _records(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def test_install_fills_a_gap(our_file, fake_flashinfer, caplog):
    our_file({"schema": 1, "provenance": {"date": "any"}, "tables": [OURS]})
    flashinfer = fake_flashinfer([copy.deepcopy(THEIRS)])

    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ((DSV41, EP16, SM156),)

    assert split.installed_tuned_tables() == ((DSV41, EP16, SM156),)
    assert flashinfer.tables == [THEIRS, OURS]
    assert not _records(caplog, logging.WARNING)
    (message,) = _records(caplog, logging.INFO)
    assert EP16 in message and "anchors [16, 128]" in message
    assert split.tuned_tables_digest() in message


def _shape(tokens: int, **layout) -> dict:
    hidden, intermediate, top_k, experts = DSV41
    num_sms, min_partition, alignment = SM156
    return dict(
        tokens_per_rank=tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_topk=top_k,
        num_total_experts=experts,
        data_parallel_size=1,
        tensor_parallel_size=1,
        num_sms=num_sms,
        sm_min_partition=min_partition,
        sm_partition_alignment=alignment,
        **layout,
    )


def test_install_has_flashinfer_resolve_the_anchors_it_adds(our_file, fake_flashinfer):
    """Stated as FlashInfer's adapter states the EP group of the topology."""
    one_domain = dict(
        OURS,
        topology="ep4_1numa",
        entries=[{"tokens": 64, "knobs": {"k2_n": 32}}],
    )
    our_file({"schema": 1, "tables": [OURS, one_domain]})
    flashinfer = fake_flashinfer([])

    assert len(split.install_tuned_tables()) == 2

    # A group that spans NUMA nodes is folded into one P2P group.
    folded = dict(
        expert_parallel_size=16,
        ep_same_numa_peer_count=15,
        ep_cross_numa_peer_count=0,
        ep_numa_split=True,
        ep_split_domain_size=4,
    )
    direct = dict(
        expert_parallel_size=4,
        ep_same_numa_peer_count=3,
        ep_cross_numa_peer_count=0,
    )
    assert flashinfer.resolved == [
        (_shape(16, **folded), {"comm_backend": "p2p_direct", "k1_stages": 3}),
        (_shape(128, **folded), {"comm_backend": "p2p_direct"}),
        (_shape(64, **direct), {"comm_backend": "p2p_direct", "k2_n": 32}),
    ]


@pytest.mark.parametrize("cached", [True, False])
def test_install_drops_what_flashinfer_resolved_before(
    our_file, fake_flashinfer, caplog, cached
):
    our_file({"schema": 1, "tables": [OURS]})
    flashinfer = fake_flashinfer([], cached=cached)

    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ((DSV41, EP16, SM156),)

    assert not _records(caplog, logging.WARNING)
    assert sorted(flashinfer.cleared) == (
        ["select_compile_spec", "select_config_and_dispatch"] if cached else []
    )


def test_install_can_be_turned_off(our_file, fake_flashinfer, monkeypatch, caplog):
    our_file({"schema": 1, "tables": [OURS]})
    flashinfer = fake_flashinfer([])
    monkeypatch.setenv(split._DISABLE_TUNED_TABLES_ENV, "1")

    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()
        assert split.install_tuned_tables() == ()

    assert split.installed_tuned_tables() == ()
    assert flashinfer.tables == [] and not flashinfer.resolved
    assert not _records(caplog, logging.WARNING)
    (message,) = _records(caplog, logging.INFO)
    assert split._DISABLE_TUNED_TABLES_ENV in message

    # No other value turns it off.
    monkeypatch.setenv(split._DISABLE_TUNED_TABLES_ENV, "0")
    monkeypatch.setattr(split, "_installed_tuned_tables", None)
    assert split.install_tuned_tables() == ((DSV41, EP16, SM156),)


@pytest.mark.parametrize("device", [SM156, SM110])
def test_install_leaves_a_key_flashinfer_has(our_file, fake_flashinfer, caplog, device):
    ours = dict(OURS, device=list(device))
    theirs = copy.deepcopy(THEIRS)
    if device != SM110:
        theirs["device"] = list(device)
    our_file({"schema": 1, "tables": [ours]})
    flashinfer = fake_flashinfer([theirs])
    before = copy.deepcopy(flashinfer.tables)

    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()

    assert split.installed_tuned_tables() == ()
    assert flashinfer.tables == before
    assert not flashinfer.resolved and not flashinfer.cleared
    assert not _records(caplog, logging.WARNING)
    assert len(_records(caplog, logging.INFO)) == 1


def test_install_is_idempotent(our_file, fake_flashinfer, monkeypatch):
    our_file({"schema": 1, "tables": [OURS]})
    flashinfer = fake_flashinfer([])

    first = split.install_tuned_tables()
    assert split.install_tuned_tables() == first == ((DSV41, EP16, SM156),)
    assert flashinfer.tables == [OURS]
    assert len(flashinfer.cleared) == 2

    # A process that lost track of its install finds the key taken.
    monkeypatch.setattr(split, "_installed_tuned_tables", None)
    assert split.install_tuned_tables() == ()
    assert flashinfer.tables == [OURS]
    assert len(flashinfer.cleared) == 2


def _changed(**changes) -> dict:
    table = copy.deepcopy(OURS)
    for name, value in changes.items():
        if value is None:
            del table[name]
        else:
            table[name] = value
    return {"schema": 1, "tables": [table]}


MALFORMED = {
    "missing": None,
    "not_json": "{",
    "not_an_object": [OURS],
    "schema": {"schema": 2, "tables": [OURS]},
    "tables_not_a_list": {"schema": 1, "tables": OURS},
    "table_not_an_object": {"schema": 1, "tables": ["dsv41"]},
    "signature_length": _changed(signature=[5120, 4608, 6]),
    "signature_type": _changed(signature=[5120, 4608, 6, "384"]),
    "signature_bool": _changed(signature=[5120, 4608, 6, True]),
    "no_topology": _changed(topology=None),
    "no_device": _changed(device=None),
    "device_length": _changed(device=[156, 8]),
    "no_entries": _changed(entries=[]),
    "entries_not_a_list": _changed(entries={"tokens": 16, "knobs": {}}),
    "tokens_zero": _changed(entries=[{"tokens": 0, "knobs": {}}]),
    "tokens_type": _changed(entries=[{"tokens": "16", "knobs": {}}]),
    "no_knobs": _changed(entries=[{"tokens": 16}]),
    "knobs_type": _changed(entries=[{"tokens": 16, "knobs": ["k1_n"]}]),
    "repeated_anchor": _changed(
        entries=[{"tokens": 16, "knobs": {}}, {"tokens": 16, "knobs": {}}]
    ),
    "repeated_key": {"schema": 1, "tables": [OURS, OURS]},
    "knob_flashinfer_ignores": _changed(
        entries=[{"tokens": 16, "knobs": {"backend": "p2p_split", "k1_stage": 3}}]
    ),
}


@pytest.mark.parametrize("content", list(MALFORMED.values()), ids=list(MALFORMED))
def test_install_survives_a_file_it_cannot_use(
    our_file, fake_flashinfer, caplog, content
):
    path = our_file(content)
    flashinfer = fake_flashinfer([copy.deepcopy(THEIRS)])

    with pytest.raises((OSError, ValueError)):
        split._read_tuned_tables(path)
    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()
        assert split.install_tuned_tables() == ()

    assert split.installed_tuned_tables() == ()
    assert flashinfer.tables == [THEIRS]
    assert len(_records(caplog, logging.WARNING)) == 1
    assert not _records(caplog, logging.INFO)


def _entry(**knobs) -> dict:
    return _changed(entries=[{"tokens": 16, "knobs": knobs}])


# Files of the right form with a table that FlashInfer would not use, or
# would fail on when the first layer is warmed up.
UNUSABLE = {
    "topology_flashinfer_lacks": _changed(topology="ep16_ring"),
    "no_backend": _entry(k1_stages=3),
    "backend_of_another_transport": _entry(backend="hybrid", k1_stages=3),
    "backend_no_adapter_runs": _changed(
        topology="ep4_1numa", entries=[{"tokens": 16, "knobs": {"backend": "hybrid"}}]
    ),
    "unresolvable": _entry(backend="p2p_split", k1_stages=0),
    "k2_warps_the_adapter_replaces": _entry(backend="p2p_split", k2_warps=12),
    "misaligned_sm_partition": _entry(backend="p2p_split", sms=[118, 0, 0, 38]),
    "sm_partitions_short_of_the_device": _entry(backend="p2p_split", sms=[112, 36]),
    "one_table_of_two": {
        "schema": 1,
        "tables": [OURS, dict(OURS, topology="ep16_ring")],
    },
}


@pytest.mark.parametrize("content", list(UNUSABLE.values()), ids=list(UNUSABLE))
def test_install_survives_a_file_flashinfer_cannot_use(
    our_file, fake_flashinfer, caplog, content
):
    path = our_file(content)
    flashinfer = fake_flashinfer([copy.deepcopy(THEIRS)])

    assert split._read_tuned_tables(path) == content["tables"]
    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()
        assert split.install_tuned_tables() == ()

    assert split.installed_tuned_tables() == ()
    assert flashinfer.tables == [THEIRS]
    assert not flashinfer.cleared
    assert len(_records(caplog, logging.WARNING)) == 1
    assert not _records(caplog, logging.INFO)


def _fresh_tables() -> dict:
    return {"schema": 1, "tables": []}


# Tables of other forms, each one object as FlashInfer's cached loader returns.
OTHER_SCHEMA = {"schema": 2, "tables": []}
OTHER_SHAPE = {"schema": 1, "tables": [{}]}
NOT_A_LIST = {"schema": 1, "tables": {}}

WITHOUT_THE_HOOK = {
    "no_load": {"_load": None},
    "load_not_cached": {"_load": _fresh_tables},
    "schema": {"_load": lambda: OTHER_SCHEMA},
    "not_keyed_by_device": {"LEGACY_DEVICE": None},
    "tables_of_another_shape": {"_load": lambda: OTHER_SHAPE},
    "tables_not_a_list": {"_load": lambda: NOT_A_LIST},
    "no_layouts": {"_SPLIT_LAYOUTS": None},
    "no_knob_mapping": {"overrides_for": None},
}


@pytest.mark.parametrize(
    "attributes", list(WITHOUT_THE_HOOK.values()), ids=list(WITHOUT_THE_HOOK)
)
def test_install_survives_a_flashinfer_without_the_hook(
    our_file, fake_flashinfer, caplog, attributes
):
    our_file({"schema": 1, "tables": [OURS]})
    flashinfer = fake_flashinfer([], **attributes)

    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()
        assert split.install_tuned_tables() == ()

    assert split.installed_tuned_tables() == ()
    assert flashinfer.tables == []
    assert len(_records(caplog, logging.WARNING)) == 1
    assert not _records(caplog, logging.INFO)


def test_install_survives_a_missing_flashinfer(our_file, monkeypatch, caplog):
    our_file({"schema": 1, "tables": [OURS]})

    def missing():
        raise ModuleNotFoundError("No module named 'flashinfer'")

    monkeypatch.setattr(split, "_flashinfer_selection", missing)
    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()

    assert len(_records(caplog, logging.WARNING)) == 1
    assert split.tuned_tables_digest() == ""


def _group(monkeypatch, *others: str) -> SimpleNamespace:
    """An EP group whose other ranks report these digests of their tables."""
    ranks = SimpleNamespace(size=lambda: 1 + len(others), rank=lambda: 0)

    def all_gather_object(object_list, obj, group=None):
        assert group is ranks and len(object_list) == ranks.size()
        object_list[:] = [obj, *others]

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    return ranks


def test_ranks_have_to_read_the_same_tables(our_file, fake_flashinfer, monkeypatch):
    our_file({"schema": 1, "tables": [OURS]})
    fake_flashinfer([copy.deepcopy(THEIRS)])
    without = split.tuned_tables_digest()
    split.install_tuned_tables()
    mine = split.tuned_tables_digest()
    assert without and mine and mine != without

    split._check_tuned_tables_match(_group(monkeypatch, mine, mine))
    # A rank whose install added nothing, and one without the lookup.
    for other in (without, ""):
        with pytest.raises(RuntimeError, match=f"'{mine}', '{other}'"):
            split._check_tuned_tables_match(_group(monkeypatch, mine, other))


@pytest.fixture
def flashinfer_tables(monkeypatch):
    """FlashInfer's own lookup with our tables installed, and its own file.

    Takes back what a test adds to the lookup.
    """
    if not split.available():
        pytest.skip("the installed FlashInfer has no split MegaMoE on NCCL")
    monkeypatch.delenv(split._DISABLE_TUNED_TABLES_ENV, raising=False)
    split.install_tuned_tables()
    # The modules FlashInfer's kernel selection reads, under the names the
    # kernels import them by once the adapter is loaded.
    import flashinfer.moe_ep.shims.sm120.split_cutedsl  # noqa: F401
    from moe_sm120_mxfp4mxfp8_split import api, heuristic, tuned_table

    assert split._flashinfer_selection() == (tuned_table, heuristic, api)
    shipped = json.loads(
        Path(tuned_table.__file__).with_name("tuned_table.json").read_text()
    )
    tables = tuned_table._load()["tables"]
    held = list(tables)
    yield tuned_table, shipped["tables"]
    tables[:] = held
    api.select_compile_spec.cache_clear()
    heuristic.select_config_and_dispatch.cache_clear()


def _table(tables: list, key: tuple, legacy_device=SM110) -> dict | None:
    return next((t for t in tables if _key(t, legacy_device) == key), None)


def _kernel(tokens: int, experts: int = DSV41[3]):
    """The kernel configuration FlashInfer's adapter resolves at EP16."""
    from moe_sm120_mxfp4mxfp8_split.api import MegaMoEProblemSpec, select_compile_spec
    from moe_sm120_mxfp4mxfp8_split.heuristic import MegaMoEHeuristicOverrides

    hidden, intermediate, top_k, _ = DSV41
    num_sms, min_partition, alignment = SM156

    def resolve(**overrides):
        return select_compile_spec(
            problem=MegaMoEProblemSpec(
                tokens_per_rank=tokens,
                num_topk=top_k,
                num_total_experts=experts,
                hidden=hidden,
                intermediate=intermediate,
                expert_parallel_size=16,
                expert_parallel_rank=0,
                data_parallel_size=1,
                tensor_parallel_size=1,
                gate_up_clamp=10.0,
            ),
            # p2p_split as the adapter states it: every peer folded into one
            # P2P group, and four GPUs per real P2P domain.
            ep_same_numa_peer_count=15,
            ep_cross_numa_peer_count=0,
            num_sms=num_sms,
            sm_min_partition=min_partition,
            sm_partition_alignment=alignment,
            overrides=MegaMoEHeuristicOverrides(comm_backend="p2p_direct", **overrides),
            ep_numa_split=True,
            ep_split_domain_size=4,
        ).kernel

    kernel = resolve()
    if kernel.rank_local_combine or kernel.k2_warps != 8:
        kernel = resolve(rank_local_combine=False, k2_warps=8)
    return kernel


def test_knobs_are_the_ones_flashinfer_reads(flashinfer_tables):
    tuned_table, _ = flashinfer_tables
    read = set()

    class Knobs(dict):
        def get(self, name, default=None):
            read.add(name)
            return default

    assert tuned_table.overrides_for(Knobs()) == {}
    assert read == split._TABLE_KNOBS == KNOBS


def test_flashinfer_finds_our_anchors(flashinfer_tables):
    tuned_table, theirs = flashinfer_tables
    key = (DSV41, EP16, SM156)
    if _table(theirs, key, tuned_table.LEGACY_DEVICE) is not None:
        pytest.skip("FlashInfer ships this table itself")
    ours = _table(_shipped()["tables"], key)
    assert ours is not None

    for tokens in ROW_BUCKETS:
        assert tuned_table.nearest_entry(DSV41, EP16, tokens, SM156) in ours["entries"]
    # Expert replicas add slots, and the slot count is part of the signature:
    # a slot count the file has no table for (it has one for 448) finds nothing.
    with_replicas = (*DSV41[:3], 416)
    assert tuned_table.nearest_entry(with_replicas, EP16, 16, SM156) is None


def test_flashinfer_tables_stay_as_shipped(flashinfer_tables):
    tuned_table, theirs = flashinfer_tables

    assert tuned_table._load()["tables"][: len(theirs)] == theirs
    ep8 = _table(theirs, (DSV41, "ep8_2numa", SM156), tuned_table.LEGACY_DEVICE)
    if ep8 is not None:
        for tokens in ROW_BUCKETS:
            entry = tuned_table.nearest_entry(DSV41, "ep8_2numa", tokens, SM156)
            assert entry in ep8["entries"]


def test_every_row_bucket_resolves_through_our_table(flashinfer_tables):
    """What FlashInfer's adapter resolves when it allocates an EP16 workspace."""
    tuned_table, theirs = flashinfer_tables
    if _table(theirs, (DSV41, EP16, SM156), tuned_table.LEGACY_DEVICE) is not None:
        pytest.skip("FlashInfer ships this table itself")

    for tokens in split.row_buckets(split.row_capacity(4096)):
        kernel = _kernel(tokens)
        assert kernel.total_sms == SM156[0], tokens
        sms = tuned_table.nearest_entry(DSV41, EP16, tokens, SM156)["knobs"].get("sms")
        if sms is not None:
            assert (kernel.k1_sms, kernel.k2_sms) == (sms[0], sms[-1]), tokens


def test_the_check_before_an_install_accepts_the_tables_flashinfer_ships(
    flashinfer_tables,
):
    tuned_table, theirs = flashinfer_tables
    from moe_sm120_mxfp4mxfp8_split import heuristic

    assert theirs
    for table in theirs:
        table = dict(table, device=list(table.get("device", tuned_table.LEGACY_DEVICE)))
        # The adapter runs none of the entries tuned for the GIN-ring transport.
        table["entries"] = [
            entry
            for entry in table["entries"]
            if entry["knobs"].get("backend") != "hybrid"
        ]
        split._resolve_anchors(table, tuned_table, heuristic)


# A table for a placement with expert replicas at a slot count that neither
# FlashInfer nor our file has, and knobs FlashInfer resolves for it.
WITH_REPLICAS = (*DSV41[:3], 416)
RESOLVABLE = {
    "backend": "p2p_split",
    "k1_stages": 3,
    "k2_tail_reclaim": "on",
    "sms": [112, 0, 0, 44],
}


def _with_replicas(**knobs) -> dict:
    table = dict(
        OURS,
        signature=list(WITH_REPLICAS),
        entries=[{"tokens": 16, "knobs": {**RESOLVABLE, **knobs}}],
    )
    return {"schema": 1, "tables": [table]}


def test_a_selection_made_before_the_install_does_not_outlive_it(
    flashinfer_tables, our_file
):
    tuned_table, _ = flashinfer_tables
    our_file(_with_replicas())
    by_rule = _kernel(16, experts=WITH_REPLICAS[3])
    assert (by_rule.k1_sms, by_rule.k2_sms) != (112, 44)

    assert split.install_tuned_tables() == ((WITH_REPLICAS, EP16, SM156),)

    assert tuned_table.nearest_entry(WITH_REPLICAS, EP16, 16, SM156) is not None
    by_table = _kernel(16, experts=WITH_REPLICAS[3])
    assert (by_table.k1_sms, by_table.k2_sms) == (112, 44)


# Knobs of the right names that abort FlashInfer's kernel selection.
UNRESOLVABLE = {
    "text_for_an_integer": {"k1_stages": "three"},
    "more_sms_than_the_device": {"sms": [200, 0, 0, 44]},
    "three_sm_partitions": {"sms": [112, 0, 44]},
    "eviction_policy": {"weight_eviction_policy": "bogus"},
    "hangs_at_the_first_launch": {"k2_launch_factor": 2, "k2_tail_reclaim": "off"},
}


@pytest.mark.parametrize("knobs", list(UNRESOLVABLE.values()), ids=list(UNRESOLVABLE))
def test_install_leaves_out_a_file_flashinfer_cannot_resolve(
    flashinfer_tables, our_file, caplog, knobs
):
    tuned_table, _ = flashinfer_tables
    path = our_file(_with_replicas(**knobs))
    before = copy.deepcopy(tuned_table._load()["tables"])

    assert split._read_tuned_tables(path)
    with caplog.at_level(logging.INFO, logger=split.__name__):
        assert split.install_tuned_tables() == ()

    assert tuned_table._load()["tables"] == before
    (message,) = _records(caplog, logging.WARNING)
    assert "anchor 16" in message
    assert not _records(caplog, logging.INFO)
