#!/usr/bin/env python3
"""Kernel-level timing of the split MegaMoE over an EP group, on the path TokenSpeed runs.

Run under ``python -m torch.distributed.run`` (one process per GPU; EP16 = two nodes of 8) inside the
TokenSpeed container with a tree's PYTHONPATH. For every requested rows-per-rank value T, ONE layer of the
DeepSeek-V4.1-Flash shape (hidden 5120, post-SwiGLU intermediate 2304, top-k 6, 384 experts, clamp 10) is
built through the TokenSpeed wrapper ``tokenspeed_kernel.thirdparty.flashinfer.mega_moe_sm120_split``
(transform_weights -> build_layer -> warmup -> run), which drives FlashInfer's SM120 CuTe DSL split kernels.
The workspace capacity is the largest requested row bucket rounded as the wrapper rounds it (to 128;
``--capacity`` sets a larger one, e.g. a serving engine's); only the buckets the requested T values select
are warmed (plus the capacity bucket, which the wrapper's warmup always launches first).

What is timed (``--regions``, the first one is the primary number of a BENCHRESULT line)
  launch  the kernel launch alone: ``layer.compute_staged(output=None)`` captured in a ``torch.cuda.graph``.
          The graph holds the per-launch reset, the cross-NUMA/cross-host transport kernels, the K1/K2
          Green Context child graph and the K3 top-k reduction, i.e. the same content as the tuning
          package's "whole graph". The wrapper's input staging (the copy of the E4M3 rows, their scales
          and the routes into the kernel workspace) runs eagerly BEFORE the L2 flush and is not timed; the
          output stays in the kernel workspace (no copy).
  run     the wrapper's whole ``run()`` captured in a ``torch.cuda.graph``: input staging into the kernel
          workspace + the launch above + the copy of the output rows to the caller's tensor. This is what
          one MoE layer adds to a TokenSpeed decode graph after TokenSpeed's own BF16 -> E4M3 staging
          kernel (that Triton kernel is not part of either region: the inputs here are already E4M3).
Per round, as ``mega_runner --perf_run --use_cuda_events --whole_graph --pr_timing --cold_l2`` does:
write 512 MB (L2 flush), ``torch.cuda.synchronize()``, a host barrier over all ranks, a GPU-side gate on the
timing stream (a one-word all-reduce the stream waits for), then start event, graph replay, end event.
20 warm-up rounds and 100 timed rounds (``--route real``: two passes over the selected dumped calls, see
below); a round's time is the maximum over the ranks, the reported number is the median over the rounds
(the package's "global_iters=... p50=..."), with the mean, a trimmed mean (the lowest and the highest 10%
of the rounds dropped), min/p10/p90/max and the median of every rank. Differences to the package: its 20
warm-up launches are eager launches before the
graph is captured (here they are replays with the same per-round protocol), its graph is instantiated by
its own capture code (here by torch), its input tensors hold exactly T rows in a workspace of T rows
(here the workspace has the capacity above and a bucket of at least T rows; rows T..bucket are masked).
The kernel sources can differ too: the package carries its own snapshot of the split kernels, TokenSpeed
runs the FlashInfer checkout's. On 2026-11-25 the package packed the cross-NUMA/cross-host dispatch at
EP16 from 32 rows per rank (its log line "[xnuma] ... pack=True"), an option the FlashInfer checkout's
transport did not have (BENCHSPEC "xnuma_pack": false): the two agree at 16 rows and differ above.

Route modes (``--route``)
  balanced  the package runner's ``--route_distribution balanced``: for every source rank, one padded block
            of 384 token slots; a random permutation P of the 384 experts and 6 distinct random offsets
            o_0..o_5 (the first 6 entries of a second random permutation); token t routes to
            P[(t + o_c) mod 384] for c = 0..5; the first T tokens of the block are used. A token's six
            experts are distinct, and each of the six columns visits distinct experts along the tokens.
            Both permutations come from ``numpy.random.default_rng(seed)`` in the runner's order (all
            ranks' permutations first, then all ranks' offsets), which is the first use of that generator
            in the runner, so ``--seed 9341`` gives the very route table of the tuner's trials.
  uniform   per row, 6 distinct experts drawn uniformly without replacement (seeded per rank).
  real      rows and routes (ids and weights) from a decode route dump of the engine at EP16
            (``--routes-dir``; ``routes-rank{rank}.pt`` = list of (layer, ids[rows, k], weights[rows, k])
            in call order; 6/12/24/48/96/192 rows per rank = 1..32 decode streams per GPU). Only whole
            decode steps are used: 40 consecutive calls with T rows and top-k 6 whose layers are 0..39 in
            order, the same calls on every rank. (The dump also holds the draft model's top-k-3 calls of
            layers 40..42 and 128/129-row calls of layers 20..39 only; those are not decode steps of the
            40 layers.) ``--route-samples`` calls (default 800 = 20 steps; rounded down to whole steps) are
            taken as whole steps spread evenly over the steps the dump holds for T (64 for the six row
            counts above). The timed rounds go through them in order, ``--rounds`` rounds in total
            (default: every call twice); the warm-up rounds use the last calls. A T without a whole step
            is skipped with a BENCHSKIP line before anything is built.
            One round is one call, so the rounds are heterogeneous (40 layers, several steps): a decode
            step pays the sum over its 40 calls, i.e. the mean, not the median, and a difference between
            two runs is read call by call. BENCHRESULT therefore keeps the per-round times and the dump
            index of every call, and table.py computes paired statistics over identical calls with a
            bootstrap interval. The two passes of a run give the noise of one call's measurement; the
            run-to-run noise is what the same command under a second TAG shows (an A/A repeat): quote a
            difference between two tables only against that.
Synthetic modes use one static route for all rounds (``--route-samples`` > 1 cycles several). Hidden states
are seeded random BF16 rows quantized here to E4M3 + UE8M0 as TokenSpeed's staging does.

Kernel table (environment variable BENCH_TABLE, applied on every rank before the first layer is built)
  asis    whatever the tree and FlashInfer provide (default)
  none    no table for this shape: every table of signature [5120, 4608, 6, 384] is removed from the dict
          FlashInfer's ``tuned_table._load()`` returns, so the rule-based selection decides
  <path>  JSON in FlashInfer's table schema ({"schema": 1, "tables": [...]}) or in the tuner's output schema
          ({"entries": [...], "signature": ..., "topology": ..., "device": ...}; reduced as the package's
          ``merge_table.py`` does); its tables replace any table of the same (signature, topology, device)
The tables a tree's TokenSpeed wrapper carries (mega_moe_sm120_split_tuned.json) are
added to FlashInfer's first, so the choice applies to the table set the layers see: asis keeps them, none
removes them too, a file replaces the ones of its keys. (The wrapper would otherwise add them at its first use,
after the choice.) BENCHTABLE lists them under "wrapper".
The ranks compare a digest of the resulting table set and stop if it differs. FlashInfer memoizes what it
resolves (``api.select_compile_spec`` and the heuristic under it): both caches are emptied with the edit,
and a run stops if a selection was resolved before it. Per T the configuration FlashInfer resolved for the
bucket is printed (BENCHSPEC, with a line in the format of the package runner's "[SM120 split heuristic]"
line), with the table anchor of the bucket, where that anchor comes from ("file" = BENCH_TABLE's file,
"wrapper" = the tree's wrapper tables, "flashinfer" = FlashInfer's own) and the fields in which the
resolved configuration differs from the anchor's knobs. With a file, the run stops unless at least one
requested bucket takes its anchor from one of the file's tables: a file made for another topology or
device would otherwise be reported under the file's name while something else decides.
A table decides EVERY row bucket of a workspace, not only the benchmarked ones: a bucket takes the anchor
nearest in log space, and a bucket outside the anchor range takes the end anchor with its pinned geometry
(a serving engine warms all buckets up to its capacity, 2048 rows in the DEP16 launch). BENCHCOVER lists,
for the buckets of ``--engine-capacity`` (default 2048), the anchor each one takes and which of them this
run did not time. Before a candidate table is committed, time it against none on all of them (two T per
run):  --capacity 2048 --tokens 7 16 32 64 128 168 256 320 384 512 1024 2048 ; ``--dry-run`` with the
same arguments shows the configurations without GPUs.

Output of rank 0 (one line each, JSON after the keyword):
  BENCHTABLE  the table choice and what it changed
  BENCHSKIP   a T that is not timed, with the reason (no whole decode step in the dump; failed check)
  BENCHSPEC   per T: resolved configuration of its bucket, table anchor and the anchor's source
  BENCHCOVER  buckets of the engine capacity: anchor per bucket, buckets not timed here
  BENCHCHECK  per T with --check: output of one eager launch against a float64 reference for a few rows.
              A T that fails is not timed (no BENCHRESULT) and the run exits with status 3.
  BENCHRESULT per T: T, bucket, route, table, p50_us (max over ranks; primary region), mean_us, tmean_us,
              rank_p50_us, min/p10/p90/max, and the same per region under "regions"; ``--route real``
              (or ``--keep-rounds``): the per-round times ("critical_us") and under "calls" the dump
              index of every selected call (timed round r is call r mod their number)
``--dry-run`` (no GPU, no process group): apply BENCH_TABLE and print BENCHTABLE / BENCHSPEC / BENCHCOVER
for the requested T values of an assumed EP world (default: EP16 as 2 hosts x 2 NUMA nodes x 4 GPUs of
156 SMs).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback

# DeepSeek-V4.1-Flash routed experts (the shape of tokenspeed-kernel's multi-rank MegaMoE test).
HIDDEN, INTER, TOP_K, EXPERTS, LIMIT = 5120, 2304, 6, 384, 10
SIGNATURE = (HIDDEN, 2 * INTER, TOP_K, EXPERTS)
LEGACY_DEVICE = (110, 8, 8)  # FlashInfer tables without a "device" field
# Weight scales 2^-9 .. 2^-6 keep the first projection O(1), inside the clamp (as in the test).
ROUTED_EXPONENTS = (118, 122)
LOG2_E = 1.4426950408889634
DEFAULT_ROUTES_DIR = ""
# MoE layers of the model: one decode step is one call per layer, layers 0..39 in order.
DECODE_LAYERS = 40
# Reference bounds of the multi-rank test for the split kernels: 3% in relative L2, 6% of the output range.
CHECK_L2_BOUND, CHECK_RANGE_BOUND = 0.03, 0.06


# ----------------------------------------------------------------------------------------------- tables


def _table_key(table: dict) -> tuple:
    return (
        tuple(table["signature"]),
        table["topology"],
        tuple(table.get("device", LEGACY_DEVICE)),
    )


def _tuner_entry(entry: dict, max_regress: float = 0.0) -> dict:
    """One anchor of a tuner output as a table entry: merge_table.py's ``entry_of``."""
    knobs = {k: v for k, v in entry["knobs"].items() if v is not None}
    confirm = entry.get("confirm") or {}
    if any(isinstance(v, float) and not math.isfinite(v) for v in confirm.values()):
        confirm = {}  # the confirmation run failed: keep the screening result only
    kept = True
    if confirm.get("best_us") and confirm.get("default_us"):
        kept = confirm["best_us"] <= confirm["default_us"] * (1.0 + max_regress)
    if not kept:
        knobs = {k: v for k, v in knobs.items() if k == "backend"}
    out = dict(
        tokens=entry["tokens"],
        knobs=knobs,
        p50_us=round(entry["p50_us"], 1),
        default_us=round(entry["default_us"], 1),
    )
    if confirm:
        out["confirm"] = {
            k: (round(v, 1) if isinstance(v, float) else v) for k, v in confirm.items()
        }
    if not kept:
        out["note"] = "winner not confirmed on the second seed; default kept"
    return out


def tables_from_file(path: str) -> list:
    """Tables of a JSON file in FlashInfer's table schema or in the tuner's output schema."""
    with open(path) as fh:
        src = json.load(fh)
    if "tables" in src:
        if src.get("schema") != 1:
            raise ValueError(f"{path}: unsupported table schema {src.get('schema')}")
        return [dict(t) for t in src["tables"]]
    if "entries" in src and "signature" in src:
        # merge_table.py: an input without a topology maps its EP size to the default topology, one
        # without a device is a 110-SM table; a 110-SM table carries no "device" field.
        default_topology = {4: "ep4_1numa", 8: "ep8_2numa", 16: "ep16_2x2numa"}
        table = dict(
            model=src.get("model", "bench"),
            signature=list(src["signature"]),
            topology=src.get("topology") or default_topology[src["ep"]],
        )
        device = src.get("device")
        if device is not None and tuple(device) != LEGACY_DEVICE:
            table["device"] = list(device)
        table.update(
            protocol=src.get("protocol", {}),
            entries=[
                _tuner_entry(e)
                for e in sorted(src["entries"], key=lambda e: e["tokens"])
            ],
        )
        return [table]
    raise ValueError(f"{path}: neither a FlashInfer table file nor a tuner output")


def _table_modules():
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import bootstrap_paths

    bootstrap_paths()
    from moe_sm120_mxfp4mxfp8_split import heuristic, tuned_table

    return heuristic, tuned_table


def _tables_digest(tables: list) -> str:
    return hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()


def _wrapper_tables() -> list:
    """Have the tree's TokenSpeed wrapper add its own tuned tables to FlashInfer's now; returns their keys.

    A wrapper that carries tables adds them at its first use. Doing it here, before the choice is applied,
    keeps that from happening after the choice (which would put a removed table back). Checkouts before the table was added
    have no such tables."""
    from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120_split as split

    install = getattr(split, "install_tuned_tables", None)
    return [] if install is None else list(install())


def _selection_caches() -> list:
    """FlashInfer's memoized selections that read the tables: the compile spec the shim asks for
    (``api.select_compile_spec``) and the heuristic under it."""
    heuristic, _ = _table_modules()
    from moe_sm120_mxfp4mxfp8_split import api

    functions = (
        ("api.select_compile_spec", api.select_compile_spec),
        ("heuristic.select_config_and_dispatch", heuristic.select_config_and_dispatch),
    )
    return [
        (name, function)
        for name, function in functions
        if hasattr(function, "cache_clear")
    ]


# Where a table of the edited set comes from, by table key: "file" (BENCH_TABLE) or "wrapper" (the tree's
# wrapper tables); every other table is FlashInfer's own.
_TABLE_SOURCE: dict = {}


def apply_table_choice(choice: str, fresh: bool = False) -> dict:
    """Edit, in place, the dict FlashInfer's ``tuned_table._load()`` caches, and empty FlashInfer's memoized
    selections. Call before any layer exists; ``fresh`` makes that a condition: it fails if a selection
    was resolved before this call (a layer could hold it)."""
    _, tuned_table = _table_modules()
    caches = _selection_caches()
    resolved = {name: function.cache_info().currsize for name, function in caches}
    if fresh and any(resolved.values()):
        raise RuntimeError(
            f"kernel selections were resolved before the table choice was applied: {resolved}"
        )
    wrapper = _wrapper_tables()
    data = tuned_table._load()
    before = [_table_key(t) for t in data.get("tables", [])]
    removed, added = [], []
    _TABLE_SOURCE.clear()
    _TABLE_SOURCE.update(
        {
            (tuple(signature), topology, tuple(device)): "wrapper"
            for signature, topology, device in wrapper
        }
    )
    if choice == "asis":
        pass
    elif choice == "none":
        removed = [k for k in before if k[0] == SIGNATURE]
        data["tables"] = [
            t for t in data.get("tables", []) if tuple(t["signature"]) != SIGNATURE
        ]
    else:
        new = tables_from_file(choice)
        keys = {_table_key(t) for t in new}
        removed = [k for k in before if k in keys]
        added = [
            dict(key=_table_key(t), anchors=[e["tokens"] for e in t["entries"]])
            for t in new
        ]
        data["tables"] = [
            t for t in data.get("tables", []) if _table_key(t) not in keys
        ] + new
        _TABLE_SOURCE.update({key: "file" for key in keys})
    # Selections resolved before the edit must not survive it: both caches, the shim calls the outer one.
    for _, function in caches:
        function.cache_clear()
    return dict(
        choice=choice,
        wrapper=wrapper,
        removed=removed,
        added=added,
        tables=[
            _table_key(t) for t in data["tables"] if tuple(t["signature"]) == SIGNATURE
        ],
        selection_caches=[name for name, _ in caches],
        resolved_before=resolved,
        digest=_tables_digest(data["tables"]),
    )


def current_table_digest() -> str:
    _, tuned_table = _table_modules()
    return _tables_digest(tuned_table._load()["tables"])


def table_anchor(
    bucket: int, world: int, same: int, cross: int, split: bool, device: tuple
):
    """The table entry FlashInfer's heuristic finds for this bucket, or None (rule-based selection)."""
    heuristic, tuned_table = _table_modules()
    shape = heuristic.MegaMoEHeuristicInput(
        tokens_per_rank=bucket,
        hidden=HIDDEN,
        intermediate=2 * INTER,
        num_topk=TOP_K,
        num_total_experts=EXPERTS,
        data_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=world,
        ep_same_numa_peer_count=same + cross if split else same,
        ep_cross_numa_peer_count=0 if split else cross,
        num_sms=device[0],
        sm_min_partition=device[1],
        sm_partition_alignment=device[2],
        ep_numa_split=split,
        ep_split_domain_size=same + 1,
    )
    entry = tuned_table.lookup(shape)
    topology = tuned_table.topology_of(shape)
    # A lookup reads the one table of this key, so the key names the anchor's source.
    source = (
        None
        if entry is None
        else _TABLE_SOURCE.get((SIGNATURE, topology, tuple(device)), "flashinfer")
    )
    return dict(
        topology=topology,
        device=list(device),
        anchor_source=source,
        anchor=(
            None
            if entry is None
            else dict(tokens=entry["tokens"], knobs=entry["knobs"])
        ),
    )


def anchor_mismatch(anchor, kernel) -> list:
    """Fields of a resolved kernel configuration that differ from what a table anchor's knobs ask for, as
    "field: resolved (anchor: wanted)"; [] when the configuration carries the anchor (or there is none).
    Ties the printed anchor, which is a second lookup, to what the layer was built with.
    """
    if not anchor:
        return []
    _, tuned_table = _table_modules()
    differ = []
    for name, want in tuned_table.overrides_for(anchor["knobs"]).items():
        if name in ("comm_backend", "token_back_mode"):
            continue  # the transport choice of a hybrid anchor, not a field of the p2p kernels
        have = _plain(getattr(kernel, name, None))
        if have != _plain(want):
            differ.append(f"{name}: {have} (anchor: {_plain(want)})")
    return differ


def bucket_cover(buckets, timed, anchor_of) -> dict:
    """Which buckets of an engine's workspace a table decides, and which of those this run does not time.
    ``anchor_of(bucket)`` is ``table_anchor`` for the world."""
    rows = []
    for bucket in buckets:
        found = anchor_of(bucket)
        anchor = found["anchor"]
        rows.append(
            [
                bucket,
                None if anchor is None else anchor["tokens"],
                found["anchor_source"],
            ]
        )
    return dict(
        engine_capacity=max(buckets),
        buckets=rows,
        timed=sorted(set(timed)),
        untimed_with_anchor=[
            row[0] for row in rows if row[1] is not None and row[0] not in timed
        ],
    )


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def describe_spec(spec, transport: str) -> dict:
    """A compile spec as JSON plus a line in the format of the package runner's heuristic line."""
    k = spec.kernel

    def g(name):
        return getattr(k, name, None)

    rows = g("expected_rows_per_expert")
    line = (
        "[SM120 split heuristic] "
        f"rows_per_expert={rows if rows is None else format(rows, '.2f')}, backend={g('comm_backend')}, "
        f"K1={g('k1_tile')}, K2={g('k2_tile')}, K2_stages={g('k2_stages')}, "
        f"SMs(K1/TX/RX/K2)={g('k1_sms')}/{g('tx_sms')}/{g('rx_sms')}/{g('k2_sms')}, "
        f"tail_reclaim={g('k2_tail_reclaim')}, dispatch_rank_cache={g('dispatch_rank_cache')}, "
        f"dispatch_rank_cache_debug={g('dispatch_rank_cache_debug')}, rank_combine={g('rank_local_combine')}, "
        f"push_dispatch={g('push_dispatch')}, k1_t0_prefetch={g('k1_t0_prefetch')}, "
        "transport(rows/channels/slots/handoff_local/remote)="
        f"{g('combine_chunk_rows')}/{g('combine_channels')}/{g('combine_slots')}/"
        f"{g('dispatch_local_handoff_windows')}/{g('dispatch_remote_handoff_windows')}"
    )
    # What the runner's line leaves out and the table knobs change.
    extra = {
        name: _plain(g(name))
        for name in (
            "k1_stages",
            "k1_ctas_per_sm",
            "k2_launch_factor",
            "k2_warps",
            "ready_queue_bundle",
            "weight_prefetch_ahead",
            "fc1_utility_postprocess",
            "weight_eviction_policy",
            "activation_eviction_policy",
            "dispatch_chunk_tokens",
        )
    }
    extra["dispatch_warps"] = getattr(
        getattr(spec, "jit", None), "dispatch_warp_count", None
    )
    extra["transport"] = transport
    try:
        kernel = {f.name: _plain(getattr(k, f.name)) for f in dataclasses.fields(k)}
    except TypeError:
        kernel = {}
    return dict(
        line=line + ", " + ", ".join(f"{n}={v}" for n, v in extra.items()),
        extra=extra,
        kernel=kernel,
    )


# ----------------------------------------------------------------------------------------------- buckets


def plan_buckets(
    tokens: list, row_capacity, select_bucket, decode_buckets, capacity=None
) -> tuple:
    """(workspace capacity, {T: bucket}): capacity = the largest requested bucket rounded by the wrapper,
    or the given one (rounded the same way) when it is at least that."""
    largest = max(next((b for b in decode_buckets if t <= b), t) for t in tokens)
    needed = row_capacity(largest)
    capacity = needed if capacity is None else row_capacity(capacity)
    if capacity < needed:
        raise ValueError(
            f"a capacity of {capacity} rows does not hold the largest bucket ({largest})"
        )
    return capacity, {t: select_bucket(t, capacity) for t in tokens}


# ----------------------------------------------------------------------------------------------- routes


def balanced_route_table(
    num_ranks: int, tokens: int, top_k: int, experts: int, seed: int
):
    """int64 [num_ranks, tokens, top_k]: the package runner's ``_generate_topk_idx_balanced`` on
    ``numpy.random.default_rng(seed)`` (see the module docstring)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    padded = (tokens + experts - 1) // experts * experts
    blocks = padded // experts
    permutations = rng.random((num_ranks, blocks, experts)).argsort(axis=-1)
    offsets = rng.random((num_ranks, blocks, experts)).argsort(axis=-1)[..., :top_k]
    token_offsets = np.arange(experts)[None, None, :, None]
    indices = (token_offsets + offsets[:, :, None, :]) % experts
    table = np.take_along_axis(permutations[..., None], indices, axis=2)
    return table.reshape(num_ranks, padded, top_k)[:, :tokens, :].astype(np.int64)


def synthetic_routes(
    mode: str, tokens: int, rank: int, world: int, seed: int, samples: int
):
    """(ids int64 [samples, tokens, TOP_K], weights float32 [samples, tokens, TOP_K]) on the CPU."""
    import torch

    ids, weights = [], []
    for sample in range(samples):
        g = torch.Generator().manual_seed((seed + sample) * 1000003 + rank)
        if mode == "balanced":
            ids.append(
                torch.from_numpy(
                    balanced_route_table(world, tokens, TOP_K, EXPERTS, seed + sample)[
                        rank
                    ]
                )
            )
        elif mode == "uniform":
            # top-k of i.i.d. uniform scores = TOP_K experts without replacement, all equally likely.
            scores = torch.rand(tokens, EXPERTS, generator=g, dtype=torch.float64)
            ids.append(scores.topk(TOP_K, dim=-1).indices)
        else:
            raise ValueError(mode)
        # Positive route weights that sum to one per row (the regime of the reference bounds).
        w = torch.rand(tokens, TOP_K, generator=g, dtype=torch.float32) + 0.5
        weights.append(w / w.sum(dim=-1, keepdim=True))
    return torch.stack(ids).to(torch.int64), torch.stack(weights)


_DUMP_CACHE: dict = {}


def decode_steps(directory: str, rank: int, tokens: int) -> tuple:
    """This rank's whole decode steps with ``tokens`` rows per rank in the route dump, in call order, and
    what the dump holds for that row count.

    A whole step is DECODE_LAYERS consecutive calls of that row count with top-k TOP_K whose layers are
    0..DECODE_LAYERS-1 in order. Returns ([step, ...], found): a step is a list of (dump index, record),
    record = (layer, ids[rows, top-k], weights[rows, top-k]); ``found`` counts the calls with that row
    count, their top-k values and layers (the reason of a skip)."""
    import torch

    key = (directory, rank)
    if key not in _DUMP_CACHE:
        path = os.path.join(directory, f"routes-rank{rank}.pt")
        _DUMP_CACHE[key] = torch.load(path) if os.path.isfile(path) else []
    rows = [
        (index, record)
        for index, record in enumerate(_DUMP_CACHE[key])
        if record[1].shape[0] == tokens
    ]
    usable = [
        (index, record)
        for index, record in rows
        if record[1].dim() == 2 and record[1].shape[1] == TOP_K
    ]
    layers = [int(record[0]) for _, record in usable]
    order = list(range(DECODE_LAYERS))
    steps, at = [], 0
    while at + DECODE_LAYERS <= len(usable):
        if layers[at : at + DECODE_LAYERS] == order:
            steps.append(usable[at : at + DECODE_LAYERS])
            at += DECODE_LAYERS
        else:
            at += 1
    seen = sorted({int(record[0]) for _, record in rows})
    found = dict(
        calls=len(rows),
        top_k=sorted({int(record[1].shape[-1]) for _, record in rows}),
        layers=[seen[0], seen[-1], len(seen)] if seen else [],
        whole_steps=len(steps),
    )
    return steps, found


def spread(available: int, wanted: int) -> list:
    """``wanted`` distinct members of range(available), evenly spread, the first and the last included."""
    if wanted >= available:
        return list(range(available))
    if wanted <= 1:
        return [0] if wanted == 1 else []
    return [
        (2 * k * (available - 1) + (wanted - 1)) // (2 * (wanted - 1))
        for k in range(wanted)
    ]


def skip_reason(tokens: int, directory: str, found: dict) -> str:
    if not found["calls"]:
        return f"the dump in {directory} holds no call with {tokens} rows"
    return (
        f"the dump in {directory} holds no whole decode step (layers 0..{DECODE_LAYERS - 1} in order, top-k "
        f"{TOP_K}) with {tokens} rows: its {found['calls']} calls with {tokens} rows have top-k "
        f"{found['top_k']} and layers {found['layers'][0]}..{found['layers'][1]}"
    )


# ----------------------------------------------------------------------------------------------- inputs


def expert_weights(expert: int, device):
    """One routed expert's packed MXFP4 tensors, the same on every rank (the test's ``_expert_weights``)."""
    import torch

    g = torch.Generator(device=device).manual_seed(1000 + expert)

    def packed(rows, columns):
        return torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, device=device, generator=g
        )

    def scales(rows, columns):
        return torch.randint(
            *ROUTED_EXPONENTS,
            (rows, columns // 32),
            dtype=torch.uint8,
            device=device,
            generator=g,
        )

    return (
        packed(2 * INTER, HIDDEN),
        scales(2 * INTER, HIDDEN),
        packed(HIDDEN, INTER),
        scales(HIDDEN, INTER),
    )


def interleave_gate_up_(weight, granularity: int = 8):
    """In place: checkpoint row order (gate rows, then up rows) to the kernels' order, 16-row groups of 8
    gate rows then 8 up rows. Same permutation as tokenspeed_kernel's ``_interleave_gate_up_``.
    """
    experts, rows = weight.shape[:2]
    blocks = rows // 2 // granularity
    for expert in range(experts):
        source = weight[expert].clone().view(2, blocks, granularity, -1)
        weight[expert].view(blocks, 2, granularity, -1).copy_(source.transpose(0, 1))
    return weight


def quantize_e4m3(x):
    """BF16/FP32 rows [T, H] to (E4M3 codes [T, H], UE8M0 exponent bytes [T, H // 32]): one power-of-two
    scale per 32 values with the input staging's 1e-4 floor on a group's magnitude."""
    import torch

    rows, columns = x.shape
    groups = x.float().view(rows, columns // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    bits = (amax / 448.0).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    inverse = ((254 - exponent) << 23).view(torch.float32)[..., None]
    codes = (groups * inverse).to(torch.float8_e4m3fn).reshape(rows, columns)
    return codes.contiguous(), exponent.to(torch.uint8).contiguous()


# ----------------------------------------------------------------------------------------------- reference


def _decode_fp4(packed, exponents):
    """Low-nibble-first E2M1 [N, K // 2] with UE8M0 [N, K // 32] to float64 [N, K]."""
    import torch

    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float64,
        device=packed.device,
    )
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    return values * torch.exp2(exponents.to(torch.float64) - 127.0).repeat_interleave(
        32, dim=-1
    )


def _fp8_round_trip(values):
    """float64 [M, K]: E4M3 with one power-of-two scale per 32 values (no floor), decoded back."""
    import torch

    rows, columns = values.shape
    groups = values.reshape(rows, columns // 32, 32)
    amax = groups.abs().amax(-1)
    bits = (amax / 448.0).to(torch.float32).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 254)
    exponent = exponent.to(torch.float64)[..., None]
    codes = (
        (groups * torch.exp2(127.0 - exponent))
        .to(torch.float32)
        .to(torch.float8_e4m3fn)
    )
    return (
        codes.to(torch.float32).to(torch.float64) * torch.exp2(exponent - 127.0)
    ).reshape(rows, columns)


def reference_rows(x_fp8, x_scale, ids, weights):
    """float64 reference of the routed experts' output for a few rows, on the numerical boundaries of the
    multi-rank test's reference: BF16 after the first projection, the clamped SwiGLU times the route weight
    requantized to E4M3, BF16 after the second projection, the slot-ordered sum rounded to BF16. Products
    and sums are float64 (the test's are fp32)."""
    import torch

    f64 = torch.float64
    x = x_fp8.to(torch.float32).to(f64) * torch.exp2(
        x_scale.to(f64) - 127.0
    ).repeat_interleave(32, dim=-1)
    partials = torch.zeros(x.shape[0], TOP_K, HIDDEN, dtype=f64, device=x.device)
    for expert in torch.unique(ids).tolist():
        rows, slots = torch.where(ids == expert)
        w13, w13_scale, w2, w2_scale = expert_weights(expert, x.device)
        fc1 = (x[rows] @ _decode_fp4(w13, w13_scale).t()).to(torch.bfloat16).to(f64)
        gate = fc1[:, :INTER].clamp(max=LIMIT)
        up = fc1[:, INTER:].clamp(min=-LIMIT, max=LIMIT)
        sigmoid = 1.0 / (1.0 + torch.exp2(-gate * LOG2_E))
        hidden = _fp8_round_trip(
            gate * sigmoid * up * weights[rows, slots].to(f64)[:, None]
        )
        partials[rows, slots] = (
            (hidden @ _decode_fp4(w2, w2_scale).t()).to(torch.bfloat16).to(f64)
        )
        del w13, w13_scale, w2, w2_scale, fc1, gate, up, sigmoid, hidden
    out = torch.zeros(x.shape[0], HIDDEN, dtype=f64, device=x.device)
    for slot in range(TOP_K):
        out += partials[:, slot]
    return out.to(torch.bfloat16).to(f64)


# ----------------------------------------------------------------------------------------------- statistics


def percentile(values: list, q: float) -> float:
    """numpy.percentile's default (linear interpolation)."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def trimmed_mean(values: list, fraction: float = 0.1) -> float:
    """Mean without the lowest and the highest ``fraction`` of the values (scipy.stats.trim_mean)."""
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    kept = ordered[cut : len(ordered) - cut]
    return sum(kept) / len(kept)


def summarize(per_rank: list) -> dict:
    """per_rank[rank][round] in microseconds -> the package's critical-path statistics (a round's time is
    the maximum over the ranks), plus the mean and a 10% trimmed mean over the rounds.
    """
    rounds = len(per_rank[0])
    critical = [max(times[r] for times in per_rank) for r in range(rounds)]
    return dict(
        p50_us=round(percentile(critical, 50), 2),
        mean_us=round(sum(critical) / rounds, 2),
        tmean_us=round(trimmed_mean(critical), 2),
        min_us=round(min(critical), 2),
        p10_us=round(percentile(critical, 10), 2),
        p90_us=round(percentile(critical, 90), 2),
        max_us=round(max(critical), 2),
        rank_p50_us=[round(percentile(times, 50), 2) for times in per_rank],
        critical_us=[round(v, 2) for v in critical],
    )


# ----------------------------------------------------------------------------------------------- dry run


def file_anchor_error(choice: str, sources: list, where: str):
    """The message a run with a table file stops with when no requested bucket takes an anchor from it."""
    if choice in ("asis", "none") or "file" in sources:
        return None
    return (
        f"BENCH_TABLE={choice}: no requested bucket takes its anchor from this file's tables "
        f"{[list(k) for k, v in _TABLE_SOURCE.items() if v == 'file']}; this world is {where} "
        f"(anchor sources of the requested buckets: {sources})"
    )


def dry_run(args, choice: str) -> int:
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import (
        DECODE_GRAPH_COMPILE_BUCKETS,
        select_graph_compile_bucket,
        workspace_compile_buckets,
    )

    print("BENCHTABLE " + json.dumps(apply_table_choice(choice)), flush=True)
    heuristic, _ = _table_modules()
    from moe_sm120_mxfp4mxfp8_split.api import MegaMoEProblemSpec, select_compile_spec

    def row_capacity(rows):
        return -(-rows // 128) * 128

    capacity, bucket_of = plan_buckets(
        args.tokens,
        row_capacity,
        select_graph_compile_bucket,
        DECODE_GRAPH_COMPILE_BUCKETS,
        args.capacity,
    )
    world, domain = args.dry_world, args.dry_domain
    device = tuple(args.dry_sms)
    same, cross = domain - 1, world - domain
    split = cross > 0
    sources, topology = [], None
    for tokens in args.tokens:
        bucket = bucket_of[tokens]
        problem = MegaMoEProblemSpec(
            tokens_per_rank=bucket,
            num_topk=TOP_K,
            num_total_experts=EXPERTS,
            hidden=HIDDEN,
            intermediate=2 * INTER,
            expert_parallel_size=world,
            expert_parallel_rank=0,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gate_up_clamp=float(LIMIT),
        )
        spec = select_compile_spec(
            problem=problem,
            ep_same_numa_peer_count=same + cross if split else same,
            ep_cross_numa_peer_count=0 if split else cross,
            num_sms=device[0],
            sm_min_partition=device[1],
            sm_partition_alignment=device[2],
            overrides=heuristic.MegaMoEHeuristicOverrides(comm_backend="p2p_direct"),
            ep_numa_split=split,
            ep_split_domain_size=domain,
        )
        record = dict(
            T=tokens,
            bucket=bucket,
            capacity=capacity,
            table=choice,
            dry_run=True,
            **table_anchor(bucket, world, same, cross, split, device),
            **describe_spec(spec, "p2p_split" if split else "p2p_direct"),
        )
        record["anchor_mismatch"] = anchor_mismatch(record["anchor"], spec.kernel)
        sources.append(record["anchor_source"])
        topology = record["topology"]
        print("BENCHSPEC " + json.dumps(record), flush=True)
    cover = bucket_cover(
        workspace_compile_buckets(row_capacity(max(args.engine_capacity, capacity))),
        list(bucket_of.values()),
        lambda bucket: table_anchor(bucket, world, same, cross, split, device),
    )
    print(
        "BENCHCOVER " + json.dumps(dict(cover, table=choice, dry_run=True)), flush=True
    )
    error = file_anchor_error(choice, sources, f"{topology} on {list(device)}")
    if error:
        print("BENCHERROR " + json.dumps(dict(message=error)), flush=True)
        return 4
    return 0


# ----------------------------------------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        required=True,
        help="rows per rank, one result each",
    )
    p.add_argument(
        "--route", choices=("balanced", "uniform", "real"), default="balanced"
    )
    p.add_argument(
        "--routes-dir", default=os.environ.get("BENCH_ROUTES_DIR", DEFAULT_ROUTES_DIR)
    )
    p.add_argument(
        "--route-samples",
        type=int,
        default=None,
        help="distinct routes cycled over the rounds (default: 1 static route for balanced/"
        f"uniform; real: 800 dumped calls, used as whole decode steps of {DECODE_LAYERS} calls "
        "spread over the dump)",
    )
    p.add_argument(
        "--regions",
        default="launch,run",
        help="comma list of launch (kernel launch only) and run (wrapper run()); the first is "
        "the primary number",
    )
    p.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="workspace row capacity (default: the largest requested bucket, rounded to 128 as "
        "the wrapper does); a serving engine sizes it for its largest forward",
    )
    p.add_argument(
        "--engine-capacity",
        type=int,
        default=2048,
        help="row capacity of the serving engine's workspace: BENCHCOVER lists the anchor of each of "
        "its buckets (nothing is built for them)",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="timed rounds (default: 100; real: every selected call twice)",
    )
    p.add_argument(
        "--flush-mb",
        type=int,
        default=512,
        help="L2 flush written before every round (0 = off)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=9341,
        help="route seed (the tuner screens with 9341)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="compare one eager launch with a float64 reference",
    )
    p.add_argument(
        "--check-rows", type=int, default=4, help="rows per rank the reference covers"
    )
    p.add_argument(
        "--keep-rounds",
        action="store_true",
        help="keep the per-round times in BENCHRESULT (always kept with --route real)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="no GPU: table choice and resolved specs only",
    )
    p.add_argument("--dry-world", type=int, default=16)
    p.add_argument(
        "--dry-domain", type=int, default=4, help="GPUs per P2P domain (NUMA node)"
    )
    p.add_argument(
        "--dry-sms",
        type=int,
        nargs=3,
        default=[156, 8, 8],
        metavar=("SMS", "MIN", "ALIGN"),
    )
    args = p.parse_args()
    args.regions = [r for r in args.regions.split(",") if r]
    if not args.regions or any(r not in ("launch", "run") for r in args.regions):
        p.error("--regions takes launch and/or run")
    if min(args.tokens) <= 0:
        p.error("--tokens must be positive")
    if args.route_samples is None:
        args.route_samples = 20 * DECODE_LAYERS if args.route == "real" else 1
    if args.route == "real":
        if args.route_samples < DECODE_LAYERS:
            p.error(
                f"--route real takes whole decode steps: --route-samples must be at least {DECODE_LAYERS}"
            )
        args.route_samples -= args.route_samples % DECODE_LAYERS
        args.keep_rounds = True
    elif args.rounds is None:
        args.rounds = 100
    if (
        args.route_samples < 1
        or (args.rounds is not None and args.rounds < 1)
        or args.warmup < 0
    ):
        p.error("--route-samples and --rounds must be positive")
    if args.engine_capacity < 1:
        p.error("--engine-capacity must be positive")
    return args


def gather_floats(values: list, world: int) -> list:
    """All-gather a list of floats of equal length; returns per_rank[rank][i]."""
    import torch
    import torch.distributed as dist

    local = torch.tensor(values, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return [g.tolist() for g in gathered]


def make_tensors(split, x_fp8, x_scale, ids, weights):
    """FlashInfer's input bundle, as the wrapper's ``run`` builds it."""
    helper = getattr(split, "_tensors", None)
    if helper is not None:
        return helper(x_fp8, x_scale, ids, weights)
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=x_fp8,
        topk_ids=ids,
        topk_weights=weights,
        scales=x_scale.view(torch.float8_e8m0fnu),
    )


def time_region(
    region, layer, split, static, samples, tokens, rounds, args, flush, gate
):
    """Per-round CUDA-event times (microseconds) of this rank for one region. Collective.
    Timed round r runs sample r mod their number (whole passes, in order); the warm-up rounds run the
    last samples."""
    import torch
    import torch.distributed as dist

    x_fp8, x_scale, ids, weights, out = static
    sample_ids, sample_weights = samples
    count = sample_ids.shape[0]
    tensors = make_tensors(split, x_fp8, x_scale, ids, weights)

    def load(index):
        ids.copy_(sample_ids[index % count])
        weights.copy_(sample_weights[index % count])
        if region == "launch":
            # The wrapper's staging, outside the timed graph.
            layer.stage_inputs(tensors, compile_tokens_per_rank=tokens)

    load(0)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if region == "launch":
            layer.compute_staged(output=None)
        else:
            split.run(
                layer,
                x_fp8,
                x_scale,
                ids,
                weights,
                max_tokens_per_rank=tokens,
                output=out,
            )
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for index in range(-args.warmup, rounds):
        if count > 1:
            load(index)
        if flush is not None:
            flush.fill_(1)
        torch.cuda.synchronize()
        dist.barrier()
        # GPU-side gate: the timing stream waits for a one-word all-reduce over the EP ranks, so every
        # rank's start event follows the same collective (the package's stream barrier).
        dist.all_reduce(gate)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if index >= 0:
            times.append(start.elapsed_time(end) * 1e3)
    torch.cuda.synchronize()
    dist.barrier()
    del graph
    return times


def bench(args, choice: str) -> int:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    def say(keyword, record):
        if rank == 0:
            # One write per line: the launcher copies the output while it grows.
            sys.stdout.write(f"{keyword} {json.dumps(record)}\n")
            sys.stdout.flush()

    born = time.monotonic()

    def stage(name, **fields):
        # Progress marks: where a run that does not finish stopped.
        say(
            "BENCHSTAGE",
            dict(stage=name, t_s=round(time.monotonic() - born, 1), **fields),
        )

    stage("process group up", world=world)

    from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120_split as split

    if not split.available():
        raise RuntimeError("the FlashInfer on PYTHONPATH has no split MegaMoE on NCCL")
    if EXPERTS % world or not split.supports_top_k(TOP_K):
        raise RuntimeError(f"unsupported EP size {world} or top-k {TOP_K}")

    # ---- kernel table: the same choice on every rank, before the first layer is built and before
    # FlashInfer resolved any selection.
    table_info = apply_table_choice(choice, fresh=True)
    digests = gather_floats([float(int(table_info["digest"][:12], 16))], world)
    if any(d != digests[0] for d in digests):
        raise RuntimeError(f"the ranks hold different kernel tables: {digests}")
    say("BENCHTABLE", dict(table_info, world=world))

    # ---- real routes: whole decode steps of the dump, the same calls on every rank. A T without one is
    # skipped here, before anything is built.
    timed, routes = list(args.tokens), {}
    if args.route == "real":
        timed = []
        for tokens in args.tokens:
            steps, found = decode_steps(args.routes_dir, rank, tokens)
            counts = [int(v[0]) for v in gather_floats([float(len(steps))], world)]
            if min(counts) == 0:
                reason = (
                    skip_reason(tokens, args.routes_dir, found)
                    if not steps
                    else f"not every rank's dump holds a whole decode step with {tokens} rows: {counts}"
                )
                say(
                    "BENCHSKIP",
                    dict(T=tokens, route="real", reason=reason, found=found),
                )
                continue
            if len(set(counts)) != 1:
                raise RuntimeError(
                    f"the ranks' dumps disagree on the decode steps with {tokens} rows: {counts}"
                )
            chosen = [
                steps[i]
                for i in spread(len(steps), args.route_samples // DECODE_LAYERS)
            ]
            calls = [call for step in chosen for call in step]
            index = [float(at) for at, _ in calls]
            if any(v != index for v in gather_floats(index, world)):
                raise RuntimeError(
                    f"the ranks' dumps disagree on the calls of the decode steps with {tokens} rows"
                )
            routes[tokens] = dict(
                ids=torch.stack([record[1].to(torch.int64) for _, record in calls]),
                weights=torch.stack(
                    [record[2].to(torch.float32) for _, record in calls]
                ),
                calls=dict(
                    steps=len(chosen),
                    available_steps=len(steps),
                    layers=[0, DECODE_LAYERS - 1],
                    first=int(index[0]),
                    last=int(index[-1]),
                    index=[at for at, _ in calls],
                ),
            )
            timed.append(tokens)
        _DUMP_CACHE.clear()
        if not timed:
            dist.destroy_process_group()
            say("BENCHDONE", dict(tokens=[], skipped=list(args.tokens)))
            return 0

    from flashinfer.moe_ep.shims.sm120.split_cutedsl import (
        DECODE_GRAPH_COMPILE_BUCKETS,
        select_graph_compile_bucket,
    )

    capacity, bucket_of = plan_buckets(
        timed,
        split.row_capacity,
        select_graph_compile_bucket,
        DECODE_GRAPH_COMPILE_BUCKETS,
        args.capacity,
    )
    needed = tuple(sorted(set(bucket_of.values())))
    known = split.row_buckets(capacity)
    if any(b not in known for b in needed):
        raise RuntimeError(
            f"buckets {needed} are not buckets of a capacity of {capacity}: {known}"
        )

    # ---- one layer of random MXFP4 weights, as the multi-rank test makes them.
    started = time.monotonic()
    local = EXPERTS // world
    parts = [
        expert_weights(expert, device)
        for expert in range(rank * local, (rank + 1) * local)
    ]
    w13, w13_scale, w2, w2_scale = (
        torch.stack([part[i] for part in parts]) for i in range(4)
    )
    del parts
    interleave_gate_up_(w13)
    transformed = split.transform_weights(
        w13, w13_scale, w2, w2_scale, hidden_size=HIDDEN, intermediate_size=INTER
    )
    layer = split.build_layer(
        process_group=dist.group.WORLD,
        device_index=torch.cuda.current_device(),
        num_experts=EXPERTS,
        max_tokens_per_rank=capacity,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        top_k=TOP_K,
        swiglu_limit=LIMIT,
        transformed_weights=transformed,
        knobs=None,
    )
    stage("layer built", capacity=capacity)
    # One zero row routed to the first experts (the row TokenSpeed warms a layer with).
    warm_x = torch.zeros(1, HIDDEN, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    warm_scale = torch.full((1, HIDDEN // 32), 0x7F, dtype=torch.uint8, device=device)
    warm_ids = (torch.arange(TOP_K, device=device) % EXPERTS).view(1, TOP_K)
    warm_weights = torch.full(
        (1, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device
    )
    split.warmup(layer, warm_x, warm_scale, warm_ids, warm_weights, buckets=needed)
    torch.cuda.synchronize()
    dist.barrier()
    stage("buckets warmed", buckets=list(needed))
    if current_table_digest() != table_info["digest"]:
        say(
            "BENCHWARN",
            dict(message="the kernel table set changed while the layer was built"),
        )

    # ---- what FlashInfer resolved.
    workspace = getattr(layer, "_workspace", None)
    plans = getattr(workspace, "_execution_plans", {}) or {}
    transport = getattr(workspace, "_comm_backend", "unknown")
    topology = getattr(workspace, "_topology", None)

    def anchor_of(bucket):
        # The lookup FlashInfer's heuristic makes for a bucket of this world.
        from moe_sm120_mxfp4mxfp8_split.runtime.green_context import (
            query_sm_resource_info,
        )

        return table_anchor(
            bucket,
            world,
            int(topology.same_numa_peer_count),
            int(topology.cross_numa_peer_count),
            transport == "p2p_split",
            tuple(int(v) for v in query_sm_resource_info()),
        )

    specs = {}
    for tokens in timed:
        bucket = bucket_of[tokens]
        record = dict(T=tokens, bucket=bucket, capacity=capacity, table=choice)
        try:
            record.update(anchor_of(bucket))
        except (
            Exception
        ) as error:  # diagnostics only, unless a table file has to be attributed (below)
            record["anchor_error"] = repr(error)
        if bucket in plans:
            record.update(describe_spec(plans[bucket][0], transport))
            try:
                record["anchor_mismatch"] = anchor_mismatch(
                    record.get("anchor"), plans[bucket][0].kernel
                )
            except Exception as error:  # diagnostics only
                record["anchor_mismatch"] = [f"not compared: {error!r}"]
            if record["anchor_mismatch"]:
                say(
                    "BENCHWARN",
                    dict(
                        T=tokens,
                        bucket=bucket,
                        message="the resolved configuration differs from "
                        "the table anchor's knobs",
                        differ=record["anchor_mismatch"],
                    ),
                )
        # Cross-NUMA/cross-host dispatch of this bucket: packed (only the routed rows are sent to an owner)
        # or the whole input span; None when the transport is not the split one.
        storage = (getattr(workspace, "_storages", {}) or {}).get(bucket)
        layout = getattr(getattr(storage, "xnuma", None), "layout", None)
        record["xnuma_pack"] = getattr(layout, "pack", None)
        specs[tokens] = record
        say("BENCHSPEC", record)
    if choice not in ("asis", "none"):
        # A table file is on trial: its tables must be what decides, or the run is not a run of the file.
        errors = [
            specs[tokens]["anchor_error"]
            for tokens in timed
            if "anchor_error" in specs[tokens]
        ]
        if errors:
            raise RuntimeError(
                f"BENCH_TABLE={choice}: the table anchors of the requested buckets cannot be "
                f"looked up, so the run cannot be attributed to the file: {errors[0]}"
            )
        sources = [specs[tokens]["anchor_source"] for tokens in timed]
        error = file_anchor_error(
            choice,
            sources,
            f"{specs[timed[0]]['topology']} on {specs[timed[0]]['device']}",
        )
        if error:
            raise RuntimeError(error)
        for tokens in timed:
            if specs[tokens]["anchor_source"] != "file":
                say(
                    "BENCHWARN",
                    dict(
                        T=tokens,
                        bucket=bucket_of[tokens],
                        message="this bucket takes no anchor "
                        "from BENCH_TABLE's file",
                        anchor_source=specs[tokens]["anchor_source"],
                    ),
                )
    try:
        cover = bucket_cover(
            split.row_buckets(split.row_capacity(max(args.engine_capacity, capacity))),
            list(bucket_of.values()),
            anchor_of,
        )
        say("BENCHCOVER", dict(cover, table=choice))
    except Exception as error:  # diagnostics only
        say("BENCHWARN", dict(message=f"no bucket cover: {error!r}"))
    say(
        "BENCHINFO",
        dict(
            world=world,
            capacity=capacity,
            warmed_buckets=list(needed),
            transport=transport,
            setup_s=round(time.monotonic() - started, 1),
            allocated_mib=torch.cuda.memory_allocated() >> 20,
        ),
    )

    flush = (
        torch.empty(args.flush_mb << 20, dtype=torch.uint8, device=device)
        if args.flush_mb
        else None
    )
    gate = torch.zeros(1, dtype=torch.int32, device=device)
    failed = []
    for tokens in timed:
        bucket = bucket_of[tokens]
        # ---- routes of this rank.
        calls = None
        if args.route == "real":
            route = routes.pop(tokens)
            sample_ids, sample_weights, calls = (
                route["ids"],
                route["weights"],
                route["calls"],
            )
        else:
            sample_ids, sample_weights = synthetic_routes(
                args.route, tokens, rank, world, args.seed, args.route_samples
            )
        samples = (sample_ids.to(device), sample_weights.to(device))
        count = int(samples[0].shape[0])
        rounds = 2 * count if args.rounds is None else args.rounds
        g = torch.Generator(device=device).manual_seed(
            args.seed * 7919 + 131 * rank + tokens
        )
        x = torch.randn(
            tokens, HIDDEN, device=device, dtype=torch.bfloat16, generator=g
        )
        x_fp8, x_scale = quantize_e4m3(x)
        static = (
            x_fp8,
            x_scale,
            samples[0][0].clone(),
            samples[1][0].clone(),
            torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16, device=device),
        )
        result = dict(
            T=tokens,
            bucket=bucket,
            capacity=capacity,
            world=world,
            route=args.route,
            samples=count,
            table=choice,
            seed=args.seed,
            warmup=args.warmup,
            rounds=rounds,
            flush_mb=args.flush_mb,
            region=args.regions[0],
        )
        if calls is not None:
            result["calls"] = calls

        # ---- optional reference check on one eager launch.
        eager = None
        if args.check:
            split.run(layer, *static[:4], max_tokens_per_rank=tokens, output=static[4])
            torch.cuda.synchronize()
            dist.barrier()
            eager = static[4].clone()
            rows = min(args.check_rows, tokens)
            actual = eager[:rows].to(torch.float64)
            reference = reference_rows(
                static[0][:rows], static[1][:rows], static[2][:rows], static[3][:rows]
            )
            finite = bool(torch.isfinite(actual).all())
            peak = float(reference.abs().max())
            max_rel = (
                float((actual - reference).abs().max()) / peak
                if peak > 0
                else float("nan")
            )
            rel_l2 = float(
                torch.linalg.norm(actual - reference) / torch.linalg.norm(reference)
            )
            del actual, reference
            per_rank = gather_floats([max_rel, rel_l2, 1.0 if finite else 0.0], world)
            check = dict(
                T=tokens,
                route=args.route,
                table=choice,
                rows_per_rank=rows,
                max_rel_err=round(max(v[0] for v in per_rank), 5),
                rel_l2=round(max(v[1] for v in per_rank), 5),
                finite=all(v[2] == 1.0 for v in per_rank),
            )
            check["ok"] = bool(
                check["finite"]
                and check["max_rel_err"] <= CHECK_RANGE_BOUND
                and check["rel_l2"] <= CHECK_L2_BOUND
            )
            say("BENCHCHECK", check)
            result["check"] = check
            if not check["ok"]:
                # Wrong rows: this configuration is no candidate, so its time is not reported; the run
                # goes on with the next T and ends with status 3.
                failed.append(tokens)
                say(
                    "BENCHSKIP",
                    dict(
                        T=tokens, route=args.route, reason="the reference check failed"
                    ),
                )
                del static, samples, eager
                continue
        stage("timing", T=tokens, rounds=rounds, samples=count)

        # ---- timing.
        result["regions"] = {}
        for region in args.regions:
            times = time_region(
                region, layer, split, static, samples, tokens, rounds, args, flush, gate
            )
            stats = summarize(gather_floats(times, world))
            if not args.keep_rounds:
                del stats["critical_us"]
            if region == "run" and eager is not None and samples[0].shape[0] == 1:
                # The last replay wrote the caller's output: it must equal the eager launch bit for bit.
                equal = gather_floats(
                    [1.0 if torch.equal(static[4], eager) else 0.0], world
                )
                stats["replay_equals_eager"] = all(v[0] == 1.0 for v in equal)
            result["regions"][region] = stats
            say(
                "BENCHREGION",
                dict(
                    T=tokens,
                    bucket=bucket,
                    route=args.route,
                    table=choice,
                    region=region,
                    **{k: v for k, v in stats.items() if k != "critical_us"},
                ),
            )
        primary = result["regions"][args.regions[0]]
        result.update(
            {
                k: primary[k]
                for k in (
                    "p50_us",
                    "mean_us",
                    "tmean_us",
                    "min_us",
                    "p10_us",
                    "p90_us",
                    "max_us",
                    "rank_p50_us",
                )
            }
        )
        spec = specs.get(tokens, {})
        result["spec"] = dict(
            line=spec.get("line"),
            anchor=spec.get("anchor"),
            topology=spec.get("topology"),
            anchor_source=spec.get("anchor_source"),
            anchor_mismatch=spec.get("anchor_mismatch"),
            xnuma_pack=spec.get("xnuma_pack"),
        )
        say("BENCHRESULT", result)
        del static, samples, eager

    # ---- collective teardown, as the test's fixtures do; never let it outlive the measurement by much.
    code = 3 if failed else 0  # a failed reference check fails the run
    finished = threading.Event()

    def watchdog():
        if not finished.wait(90):
            print(
                f"BENCHWARN rank {rank}: teardown did not finish within 90 s",
                flush=True,
            )
            os._exit(code)

    threading.Thread(target=watchdog, daemon=True).start()
    stage("teardown")
    del flush
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    split.destroy(layer)
    dist.destroy_process_group()
    finished.set()
    say(
        "BENCHDONE",
        dict(
            tokens=[tokens for tokens in timed if tokens not in failed],
            check_failed=failed,
            skipped=[tokens for tokens in args.tokens if tokens not in timed],
        ),
    )
    return code


def main() -> int:
    args = parse_args()
    choice = os.environ.get("BENCH_TABLE") or "asis"
    if args.dry_run:
        return dry_run(args, choice)
    return bench(args, choice)


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Leave at once: the other ranks wait in collectives and are stopped by the launcher's timeout.
        os._exit(1)
    sys.stdout.flush()
    sys.exit(code)
