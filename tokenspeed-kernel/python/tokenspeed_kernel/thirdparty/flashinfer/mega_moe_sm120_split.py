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

"""FlashInfer's SM120 MXFP4 x MXFP8 split MegaMoE (``sm120_mxfp4_mxfp8_bf16_cutedsl``).

Three CuTe DSL kernels per launch instead of one persistent kernel: the first
dispatches rows to the experts' ranks and runs FC1, the second runs FC2 and
returns the rows, the third reduces the top-k results. The first two run at the
same time on two Green Context partitions of the GPU's SMs, launched as one
native CUDA graph. The EP ranks exchange data over the kernel's own NCCL
session: load/store between GPUs of one host and NUMA node, and local shadow
buffers synchronized with GPU-initiated networking for every other pair.

The kernels are compiled per row bucket (``row_buckets``), and each layer
keeps one launch graph per bucket, so every bucket a CUDA graph will use has
to be warmed up before capture. The kernel runs the routed experts only.

FlashInfer picks each bucket's kernel configuration from its tuned tables and,
where it has none, by rule. ``mega_moe_sm120_split_tuned.json`` carries tables
for setups FlashInfer ships none for; ``install_tuned_tables`` adds them to
FlashInfer's lookup before the first layer is built, and ``build_layer``
refuses an EP group whose ranks do not read the same tables.

FlashInfer imports stay inside the functions, as in ``mega_moe_sm120``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# The row capacity is a workspace size here, not a compile-time parameter; it
# keeps the multiple-of-128 form of the persistent kernel's.
ROW_ALIGNMENT = 128

# The adapter of the kernel drop and the NCCL transport it runs on. FlashInfer
# releases before the NCCL transport ship the same backend name on NVSHMEM,
# which cannot span NUMA nodes or hosts.
_REQUIRED_FILES = (
    "moe_ep/shims/sm120/split_cutedsl/__init__.py",
    "moe_ep/kernels/flashinfer/sm120_split/moe_sm120_mxfp4mxfp8_split/"
    "transport/nccl/session.py",
)

# Tuned kernel tables added to FlashInfer's own, in FlashInfer's table schema.
_TUNED_TABLES_PATH = Path(__file__).with_name("mega_moe_sm120_split_tuned.json")
_TUNED_TABLES_SCHEMA = 1
# Set to 1 to add none of them and keep FlashInfer's own selection.
_DISABLE_TUNED_TABLES_ENV = "TOKENSPEED_DISABLE_MEGA_MOE_SPLIT_TUNED_TABLES"

# The knobs of a table entry that FlashInfer's ``tuned_table.overrides_for``
# reads. It ignores any other name.
_TABLE_KNOBS = frozenset(
    {
        "backend",
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
        "k2_tail_reclaim",
        "fc1_utility_postprocess",
        "push_dispatch",
        "weight_eviction_policy",
        "sms",
    }
)

# A table's key: (signature, topology, device). The signature is (hidden,
# gate + up intermediate, top-k, expert slots); the device is (SMs, smallest
# Green Context partition, partition alignment).
TableKey = tuple[tuple[int, ...], str, tuple[int, ...]]

# Keys of the tables added to FlashInfer's lookup. None before the first
# attempt, which is the only one.
_installed_tuned_tables: tuple[TableKey, ...] | None = None


def available() -> bool:
    """Whether the installed FlashInfer ships the split MegaMoE on NCCL.

    Looks for the backend's files without importing FlashInfer.
    """
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        all((Path(location) / name).is_file() for name in _REQUIRED_FILES)
        for location in spec.submodule_search_locations
    )


def supports_top_k(top_k: int) -> bool:
    """Whether the kernels can run layers with this many routes per token.

    Up to 64 rows per rank the kernels push each row's route weights to the
    peer GPUs in 8-byte words, so a row's ``4 * top_k`` bytes must be a
    multiple of 8. With an odd top-k the launch faults on a misaligned address
    (or leaves the last route weight unsent).
    """
    return top_k > 0 and top_k % 2 == 0


def _is_ints(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(item) is int for item in value)
    )


def _table_key(table: dict) -> TableKey:
    return tuple(table["signature"]), table["topology"], tuple(table["device"])


def _describe(key: TableKey) -> str:
    signature, topology, device = key
    return f"{list(signature)} {topology} {list(device)}"


def _read_tuned_tables(path: Path) -> list[dict]:
    """Read a tuned-table file and check what FlashInfer's lookup relies on.

    Args:
        path: File in FlashInfer's table schema whose tables all name their
            ``device`` and whose knobs are ones FlashInfer reads.

    Returns:
        Its tables.

    Raises:
        OSError: The file cannot be read.
        ValueError: The content is not such a file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != _TUNED_TABLES_SCHEMA:
        raise ValueError(f"expected an object with schema {_TUNED_TABLES_SCHEMA}")
    tables = data.get("tables")
    if not isinstance(tables, list):
        raise ValueError("tables is not a list")
    keys = set()
    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            raise ValueError(f"table {index} is not an object")
        if not _is_ints(table.get("signature"), 4):
            raise ValueError(f"table {index}: signature is not four integers")
        if not isinstance(table.get("topology"), str):
            raise ValueError(f"table {index}: topology is not a string")
        if not _is_ints(table.get("device"), 3):
            # The tuning tools' merge_table.py leaves it out of a 110-SM table.
            raise ValueError(
                f"table {index}: device is not three integers (a table without "
                "one is a 110-SM table and needs [110, 8, 8])"
            )
        key = _table_key(table)
        if key in keys:
            raise ValueError(f"table {index} repeats the key {_describe(key)}")
        keys.add(key)
        entries = table.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"table {index} has no entries")
        anchors = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or type(entry.get("tokens")) is not int
                or entry["tokens"] <= 0
                or not isinstance(entry.get("knobs"), dict)
            ):
                raise ValueError(
                    f"table {index}: an entry needs a positive integer tokens "
                    "and a knobs object"
                )
            if entry["tokens"] in anchors:
                raise ValueError(f"table {index} repeats the anchor {entry['tokens']}")
            anchors.add(entry["tokens"])
            unread = sorted(set(entry["knobs"]) - _TABLE_KNOBS)
            if unread:
                raise ValueError(
                    f"table {index}, anchor {entry['tokens']}: FlashInfer reads "
                    f"no knob named {', '.join(unread)}"
                )
    return tables


def _flashinfer_selection() -> tuple[object, object, object]:
    """FlashInfer's kernel selection of the split kernels.

    The kernels import their modules as a top-level package of their source
    drop, which FlashInfer's adapter puts on ``sys.path``. The same files
    imported under their ``flashinfer`` package path would be other modules
    with tables of their own.

    Returns:
        The tuned-table, heuristic and API modules.
    """
    import flashinfer.moe_ep.shims.sm120.split_cutedsl  # noqa: F401
    from moe_sm120_mxfp4mxfp8_split import api, heuristic, tuned_table

    return tuned_table, heuristic, api


def _resolve_anchors(table: dict, tuned_table: object, heuristic: object) -> None:
    """Have FlashInfer resolve every anchor of a table as its adapter states it.

    The adapter folds the peers of an EP group that spans P2P domains into one
    P2P group (``p2p_split``) and runs the ``p2p_direct`` kernels either way.

    Args:
        table: A table of ``_read_tuned_tables`` whose key FlashInfer lacks.
        tuned_table: FlashInfer's tuned-table module.
        heuristic: FlashInfer's heuristic module.

    Raises:
        ValueError: FlashInfer would not use an entry, or cannot resolve it.
    """
    hidden, intermediate, top_k, experts = table["signature"]
    num_sms, min_partition, alignment = table["device"]
    topology = table["topology"]
    split = {name: layout for layout, name in tuned_table._SPLIT_LAYOUTS.items()}
    real = {name: layout for layout, name in tuned_table._REAL_LAYOUTS.items()}
    if topology in split:
        ranks, domain = split[topology]
        layout = dict(
            expert_parallel_size=ranks,
            ep_same_numa_peer_count=ranks - 1,
            ep_cross_numa_peer_count=0,
            ep_numa_split=True,
            ep_split_domain_size=domain,
        )
    elif topology in real:
        ranks, same, cross = real[topology]
        layout = dict(
            expert_parallel_size=ranks,
            ep_same_numa_peer_count=same,
            ep_cross_numa_peer_count=cross,
        )
    else:
        raise ValueError(f"FlashInfer has no topology named {topology}")
    for entry in table["entries"]:
        where = f"{_describe(_table_key(table))} anchor {entry['tokens']}"
        knobs = entry["knobs"]
        backend = knobs.get("backend")
        # FlashInfer applies an entry of a folded group to the transport it
        # names only, and the adapter never runs the hybrid transport.
        if backend == "hybrid" or (topology in split and backend != "p2p_split"):
            raise ValueError(
                f"{where}: FlashInfer's adapter does not run its backend ({backend})"
            )
        # The adapter runs 8 K2 warps on a folded group whatever the entry says.
        if topology in split and knobs.get("k2_warps") not in (None, 8):
            raise ValueError(f"{where}: p2p_split runs 8 K2 warps")
        # The Green Context partitions are created as the entry states them;
        # FlashInfer's selection does not check their alignment.
        sms = knobs.get("sms")
        if sms is not None and not (
            isinstance(sms, list)
            and len(sms) in (2, 4)
            and all(type(count) is int and count >= 0 for count in sms)
            and all(count % alignment == 0 for count in sms[:-1])
            and min(sms[0], sms[-1]) >= min_partition
            and sum(sms) == num_sms
        ):
            raise ValueError(
                f"{where}: sms {sms} does not split the device's {num_sms} SMs "
                f"into partitions of at least {min_partition}, aligned to "
                f"{alignment}"
            )
        try:
            shape = heuristic.MegaMoEHeuristicInput(
                tokens_per_rank=entry["tokens"],
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
            overrides = heuristic.MegaMoEHeuristicOverrides(
                **{"comm_backend": "p2p_direct", **tuned_table.overrides_for(knobs)}
            )
            heuristic.select_config_and_dispatch(shape, overrides)
        except Exception as error:
            raise ValueError(
                f"{where}: FlashInfer cannot resolve it ({error})"
            ) from error


def _tables_digest(tables: list) -> str:
    text = json.dumps(tables, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def install_tuned_tables() -> tuple[TableKey, ...]:
    """Add the tables of ``mega_moe_sm120_split_tuned.json`` to FlashInfer's.

    FlashInfer looks a row bucket's kernel configuration up by (signature,
    topology, device) and selects by rule where no table has that key. Only
    tables whose key FlashInfer's own file lacks are added, so a table that
    FlashInfer ships wins over ours. FlashInfer caches what it resolves for a
    bucket: the selections it cached before are dropped here, but a layer
    keeps what it was built with, so the first call has to precede the first
    layer; every function of this module that reaches FlashInfer's kernel
    selection sees to that. Code that edits FlashInfer's tables itself calls
    this first. Later calls add nothing.

    ``TOKENSPEED_DISABLE_MEGA_MOE_SPLIT_TUNED_TABLES=1`` adds none.

    Never raises: with a FlashInfer that has no such lookup, or a file that
    cannot be used (FlashInfer has to resolve every anchor of a table that is
    added), it logs one warning and FlashInfer's selection stays as it is.

    Returns:
        Keys of the tables this process added.
    """
    global _installed_tuned_tables
    if _installed_tuned_tables is not None:
        return _installed_tuned_tables
    _installed_tuned_tables = ()
    if os.environ.get(_DISABLE_TUNED_TABLES_ENV) == "1":
        logger.info(
            f"Split MegaMoE: the tuned kernel tables of {_TUNED_TABLES_PATH.name} "
            f"are not used ({_DISABLE_TUNED_TABLES_ENV}=1)"
        )
        return ()
    try:
        ours = _read_tuned_tables(_TUNED_TABLES_PATH)
        tuned_table, heuristic, api = _flashinfer_selection()
        data = tuned_table._load()
        if tuned_table._load() is not data:
            # Lookups would read another object than the one extended here.
            raise RuntimeError("FlashInfer does not cache its tuned tables")
        if data.get("schema") != _TUNED_TABLES_SCHEMA:
            raise ValueError(f"FlashInfer's tables have schema {data.get('schema')}")
        # What FlashInfer keys a table without a device by. A lookup without
        # it is not keyed by device and would apply our tables to any GPU.
        legacy_device = tuple(tuned_table.LEGACY_DEVICE)
        theirs = {
            (
                tuple(table["signature"]),
                table["topology"],
                tuple(table.get("device", legacy_device)),
            )
            for table in data["tables"]
        }
        missing = [table for table in ours if _table_key(table) not in theirs]
        for table in missing:
            _resolve_anchors(table, tuned_table, heuristic)
        selections = (api.select_compile_spec, heuristic.select_config_and_dispatch)
        tables = data["tables"]
        if not isinstance(tables, list):
            raise ValueError("FlashInfer's tables are not a list")
    except Exception as error:
        logger.warning(
            f"Split MegaMoE: the tuned kernel tables of {_TUNED_TABLES_PATH.name} "
            "are not used, FlashInfer's own selection applies "
            f"({type(error).__name__}: {error})"
        )
        return ()
    added, shipped = [], []
    for table in ours:
        key = _table_key(table)
        if key in theirs:
            shipped.append(_describe(key))
            continue
        tables.append(table)
        anchors = sorted(entry["tokens"] for entry in table["entries"])
        added.append((key, f"{_describe(key)} anchors {anchors}"))
    if added:
        # What FlashInfer resolved without the tables must not outlive them.
        for selection in selections:
            cache_clear = getattr(selection, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()
    _installed_tuned_tables = tuple(key for key, _ in added)
    logger.info(
        "Split MegaMoE: tuned kernel tables added to FlashInfer's: "
        f"{'; '.join(text for _, text in added) or 'none'}; left to the table "
        f"FlashInfer ships: {'; '.join(shipped) or 'none'}; digest of the "
        f"tables FlashInfer reads now: {_tables_digest(tables)}"
    )
    return _installed_tuned_tables


def installed_tuned_tables() -> tuple[TableKey, ...]:
    """Keys of the tables ``install_tuned_tables`` added to FlashInfer's lookup.

    Empty before it ran, and when it added none.
    """
    return _installed_tuned_tables or ()


def tuned_tables_digest() -> str:
    """Digest of the tuned tables FlashInfer's kernel selection reads here.

    Covers FlashInfer's own tables and the ones added to them, in lookup
    order. Empty with a FlashInfer that has no such lookup.
    """
    try:
        return _tables_digest(_flashinfer_selection()[0]._load()["tables"])
    except Exception:
        return ""


def _check_tuned_tables_match(process_group: object) -> None:
    """Raise unless every rank of an EP group reads the same tuned tables.

    Collective. The tables are installed per process, and FlashInfer compares
    workspace sizes across the ranks, which do not depend on the kernel
    configuration a table selects.

    Args:
        process_group: torch.distributed group of the EP ranks.

    Raises:
        RuntimeError: The ranks would select different kernel configurations.
    """
    import torch.distributed as dist

    if process_group.size() == 1:
        return
    mine = tuned_tables_digest()
    digests = [None] * process_group.size()
    dist.all_gather_object(digests, mine, group=process_group)
    if any(digest != mine for digest in digests):
        raise RuntimeError(
            "Split MegaMoE: the EP ranks read different tuned kernel tables "
            f"(digest per rank: {digests}); every rank needs the same "
            f"FlashInfer, the same {_TUNED_TABLES_PATH.name} and the same "
            f"{_DISABLE_TUNED_TABLES_ENV}"
        )


def _split_shim() -> object:
    """FlashInfer's split MegaMoE adapter, once the tuned tables are in place."""
    from flashinfer.moe_ep.shims.sm120 import split_cutedsl

    install_tuned_tables()
    return split_cutedsl


def row_capacity(max_tokens_per_rank: int) -> int:
    """Rows the workspace holds: ``max_tokens_per_rank`` rounded up to 128.

    Args:
        max_tokens_per_rank: Largest per-rank token count of one forward.

    Returns:
        The workspace's row capacity.
    """
    if max_tokens_per_rank <= 0:
        raise ValueError(
            f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}"
        )
    return -(-max_tokens_per_rank // ROW_ALIGNMENT) * ROW_ALIGNMENT


def row_buckets(row_capacity: int) -> tuple[int, ...]:
    """Row counts the kernels are compiled for, ascending.

    A launch with ``max_tokens_per_rank`` rows runs the smallest bucket that
    holds them.

    Args:
        row_capacity: Result of ``row_capacity``.

    Returns:
        Every bucket a workspace of this capacity can select; the last one is
        the capacity itself.
    """
    return _split_shim().workspace_compile_buckets(row_capacity)


def transform_weights(
    w13_interleaved: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> object:
    """Lay this rank's routed expert weights out for the kernels.

    Args:
        w13_interleaved: Packed E2M1 ``[local_experts, 2 * intermediate,
            hidden // 2]`` whose rows are already in the kernel order: 16-row
            groups of 8 gate rows then 8 up rows. Used as is, not copied.
        w13_scale: UE8M0 bytes ``[local_experts, 2 * intermediate,
            hidden // 32]`` in checkpoint order, gate rows then up rows.
        w2: Packed E2M1 ``[local_experts, hidden, intermediate // 2]``. Used as
            is, not copied.
        w2_scale: UE8M0 bytes ``[local_experts, hidden, intermediate // 32]``.
        hidden_size: Model hidden size.
        intermediate_size: Expert intermediate size.

    Returns:
        FlashInfer's kernel-layout weights for ``build_layer``.
    """
    shim = _split_shim()
    (_, fc1_scale), (fc2_weight, fc2_scale) = shim.transform_prequantized_weights(
        w13_interleaved,
        w2,
        w13_scale,
        w2_scale,
        hidden=hidden_size,
        intermediate=intermediate_size,
    )
    # FlashInfer interleaves the gate/up rows it is given. Ours already are,
    # so keep the caller's tensor and drop the twice-permuted copy.
    return (w13_interleaved.view(shim.FP4_DTYPE), fc1_scale), (fc2_weight, fc2_scale)


def build_layer(
    *,
    process_group: object,
    device_index: int,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    swiglu_limit: int,
    transformed_weights: object,
    knobs: dict | None,
) -> object:
    """Create this rank's split MegaMoE layer over an EP process group.

    Collective: every rank of ``process_group`` calls it with the same
    arguments. Layers of one geometry share one workspace and NCCL session.
    Raises if the ranks do not read the same tuned kernel tables.

    Args:
        process_group: torch.distributed NCCL group of the EP ranks. The
            kernel creates its own NCCL communicator over the same ranks.
        device_index: CUDA device this rank runs on.
        num_experts: Routed experts across the EP group.
        max_tokens_per_rank: Result of ``row_capacity``.
        hidden_size: Model hidden size.
        intermediate_size: Expert intermediate size.
        top_k: Routed experts per token.
        swiglu_limit: SwiGLU clamp ``gate <= limit``, ``|up| <= limit``; 0
            leaves the activation unclamped.
        transformed_weights: Result of ``transform_weights``.
        knobs: Overrides of FlashInfer's kernel selection, by the field names
            of its ``MegaMoEHeuristicOverrides``; they apply to every row
            bucket. None keeps FlashInfer's selection, tuned tables included.

    Returns:
        FlashInfer's ``MoEEpMegaLayer``.
    """
    # FlashInfer resolves and caches every row bucket's kernel configuration
    # when it allocates the workspace.
    install_tuned_tables()
    _check_tuned_tables_match(process_group)
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    bootstrap = BootstrapConfig(
        world_size=process_group.size(),
        rank=process_group.rank(),
        # The serving runtime owns torch.distributed.
        auto_bootstrap=False,
        device=device_index,
        process_group=process_group,
    )
    fleet_params = FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        token_hidden_size=hidden_size,
    )
    config = Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate_size,
        top_k=top_k,
        gate_up_clamp=float(swiglu_limit) if swiglu_limit else None,
        knobs=knobs,
    )
    return MoEEpLayer(
        bootstrap,
        fleet_params,
        None,
        backend=MegaConfig(
            megakernel=config,
            # The caller stages E4M3 rows and their UE8M0 scales itself.
            quantize_input=False,
            preprocess_weights=False,
            transformed_weights=transformed_weights,
        ),
    )


def _tensors(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> object:
    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=x_fp8,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        # Same bytes; FlashInfer checks for its scale dtype.
        scales=x_scale.view(torch.float8_e8m0fnu),
    )


def warmup(
    layer: object,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    buckets: tuple[int, ...],
) -> None:
    """Launch the layer once per row bucket so later calls can be captured.

    Collective, outside graph capture. The first call of a geometry allocates
    its workspace and NCCL session; the first call of a bucket compiles that
    bucket's kernels; each layer's first call of a bucket builds the layer's
    launch graph for it. A bucket that is not warmed up here is compiled at
    its first launch, which a CUDA graph capture refuses.

    Args:
        layer: Result of ``build_layer``.
        x_fp8: E4M3 codes ``[rows, hidden]``.
        x_scale: UE8M0 bytes ``[rows, hidden // 32]``.
        topk_ids: int64 global expert ids ``[rows, top_k]``.
        topk_weights: float32 route weights ``[rows, top_k]``.
        buckets: Row buckets to warm up, from ``row_buckets``, each at least
            the number of rows passed.
    """
    tensors = _tensors(x_fp8, x_scale, topk_ids, topk_weights)
    layer.warmup(tensors)
    for bucket in buckets:
        layer.stage_inputs(tensors, compile_tokens_per_rank=bucket)
        layer.compute_staged(output=None)
    torch.cuda.synchronize()


def run(
    layer: object,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    max_tokens_per_rank: int | None,
    output: torch.Tensor,
    before_launch: Callable[[], None] | None = None,
) -> None:
    """Launch the kernels on this rank's rows. Collective over the EP group.

    Args:
        layer: Result of ``build_layer``, already warmed up.
        x_fp8: E4M3 codes ``[rows, hidden]``.
        x_scale: UE8M0 bytes ``[rows, hidden // 32]``.
        topk_ids: int64 global expert ids ``[rows, top_k]``; -1 masks a route.
        topk_weights: float32 route weights ``[rows, top_k]``.
        max_tokens_per_rank: Largest per-rank row count of this call, the same
            value on every rank; it selects the row bucket. None selects the
            workspace capacity.
        output: BF16 ``[rows, hidden]`` to receive this rank's rows.
        before_launch: Optional callback, run on the calling stream after the
            inputs are staged into the kernel's workspace and before the
            kernels are launched. None (the default) changes nothing.
    """
    layer.stage_inputs(
        _tensors(x_fp8, x_scale, topk_ids, topk_weights),
        compile_tokens_per_rank=max_tokens_per_rank,
    )
    if before_launch is not None:
        before_launch()
    # The launch graph is tied to its output buffer, so launch into the
    # workspace's and copy: a caller buffer of full capacity would otherwise
    # get a graph of its own on every call.
    output.copy_(layer.compute_staged(output=None))


def destroy(layer: object) -> None:
    """Release a layer's launch graphs and its share of the workspace. Collective.

    Call on every EP rank, in the same order, before the EP process group is
    destroyed and once no captured CUDA graph still launches the layer.

    Args:
        layer: Result of ``build_layer``.
    """
    layer.destroy()
