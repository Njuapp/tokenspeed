#!/usr/bin/env python3
"""Kernel-level timing of FlashInfer's persistent SM120 MXFP4 x MXFP8 MegaMoE over an EP group, in the round
protocol of bench16.py and on its route tables, so that its numbers stand next to the split kernels'.

One layer of the DeepSeek-V4.1-Flash shape (384 routed experts, no shared expert) on the transport table the
engine uses (``all_gin``), workspace of 2048 rows. Timed: ``layer.compute_staged(output=None)`` captured in a
``torch.cuda.graph``; per round the inputs are staged eagerly (not timed), 512 MB are written (L2 flush), then
a host barrier, a GPU-side gate, start event, graph replay, end event; a round's time is the maximum over the
ranks, the reported number the median over the rounds. Route modes besides ``balanced`` and ``uniform``:

  real       whole decode steps of a route dump (``--routes-dir``), chosen exactly as bench16.py chooses them
             (same calls on every rank, ``--route-samples`` calls spread over the steps the dump holds, every call
             timed twice); ids AND weights come from the dump. A round is one MoE call, so the number that
             counts is the MEAN over the rounds (a decode step pays the sum over its 40 calls).
  power_law  the tuning package runner's ``--route_distribution power_law`` (Zipf exponent 1.0) with the same
             generator calls, so ``--seed 9341`` is the tuner's route table.

"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench16 as b  # noqa: E402


def power_law_routes(
    world: int, tokens: int, top_k: int, experts: int, seed: int, exponent: float = 1.0
):
    """mega_runner._generate_topk_idx_power_law on ``numpy.random.default_rng(seed)`` (its first use there)."""
    rng = np.random.default_rng(seed)
    popularity_rank_to_expert = rng.permutation(experts)
    rank_freq = 1.0 / (np.arange(experts) + 1).astype(np.float64) ** exponent
    probs = np.empty(experts, dtype=np.float64)
    probs[popularity_rank_to_expert] = rank_freq / rank_freq.sum()
    log_probs = np.log(np.maximum(probs, 1e-30))
    uniform_noise = rng.random(size=(world * tokens, experts))
    gumbel_noise = -np.log(-np.log(np.maximum(uniform_noise, 1e-30)) + 1e-30)
    scores = log_probs[None, :] + gumbel_noise
    topk = np.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
    return topk.reshape(world, tokens, top_k).astype(np.int64)


def main() -> int:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_synthetic import (
        Geometry,
        make_layer,
        random_inputs,
        random_weights,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", required=True)
    p.add_argument(
        "--route",
        choices=("balanced", "uniform", "power_law", "real"),
        default="balanced",
    )
    p.add_argument("--routes-dir", default=None)
    p.add_argument("--route-samples", type=int, default=800)
    p.add_argument("--table", default="all_gin")
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--flush-mb", type=int, default=512)
    p.add_argument("--seed", type=int, default=9341)
    args = p.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    # Real routes first (CPU): a T without a whole decode step on every rank is skipped.
    timed, routes = list(args.tokens), {}
    if args.route == "real":
        timed = []
        for tokens in args.tokens:
            steps, found = b.decode_steps(args.routes_dir, rank, tokens)
            counts = [int(v[0]) for v in b.gather_floats([float(len(steps))], world)]
            if min(counts) == 0 or len(set(counts)) != 1:
                if rank == 0:
                    print(
                        "CAKESKIP "
                        + json.dumps(
                            dict(tokens=tokens, whole_steps=counts, found=found)
                        ),
                        flush=True,
                    )
                continue
            chosen = [
                steps[i]
                for i in b.spread(len(steps), args.route_samples // b.DECODE_LAYERS)
            ]
            calls = [call for step in chosen for call in step]
            index = [float(at) for at, _ in calls]
            if any(v != index for v in b.gather_floats(index, world)):
                raise RuntimeError(
                    f"the ranks' dumps disagree on the calls with {tokens} rows"
                )
            routes[tokens] = (
                torch.stack([record[1].to(torch.int64) for _, record in calls]),
                torch.stack([record[2].to(torch.float32) for _, record in calls]),
                dict(
                    steps=len(chosen),
                    available_steps=len(steps),
                    first=int(index[0]),
                    last=int(index[-1]),
                ),
            )
            timed.append(tokens)

    geometry = Geometry(b.HIDDEN, b.INTER, b.EXPERTS, b.TOP_K)
    weights = random_weights(rank, world, geometry, shared=False, seed=args.seed)
    layer = make_layer(
        rank,
        world,
        weights,
        geometry,
        max_tokens_per_rank=args.capacity,
        shared=False,
        transport_table=args.table,
        swiglu_limit=b.LIMIT,
    )
    layer.warmup(
        random_inputs(min(256, args.capacity), geometry, seed=args.seed + rank)
    )
    del weights
    torch.cuda.synchronize()
    dist.barrier()
    flush = (
        torch.empty(args.flush_mb << 20, dtype=torch.uint8, device=device)
        if args.flush_mb
        else None
    )
    gate = torch.zeros(1, dtype=torch.int32, device=device)
    for tokens in timed:
        inputs = random_inputs(tokens, geometry, seed=args.seed + 7 * rank + tokens)
        calls = None
        if args.route == "real":
            sample_ids, sample_weights, calls = routes.pop(tokens)
            sample_ids, sample_weights = sample_ids.to(device), sample_weights.to(
                device
            )
        elif args.route == "power_law":
            table = power_law_routes(world, tokens, b.TOP_K, b.EXPERTS, args.seed)
            sample_ids, sample_weights = (
                torch.from_numpy(table[rank])[None].to(device),
                None,
            )
        else:
            ids, _ = b.synthetic_routes(args.route, tokens, rank, world, args.seed, 1)
            sample_ids, sample_weights = ids[:1].to(device), None
        count = int(sample_ids.shape[0])
        rounds = 2 * count if args.route == "real" else args.rounds

        def load(index):
            inputs.topk_ids.copy_(sample_ids[index % count].to(inputs.topk_ids.dtype))
            if sample_weights is not None:
                inputs.topk_weights.copy_(sample_weights[index % count])
            layer.stage_inputs(inputs, compile_tokens_per_rank=tokens)

        # A few eager launches first: the bucket's first launch is not a graph's business.
        for _ in range(3):
            load(0)
            layer.compute_staged(output=None)
        torch.cuda.synchronize()
        dist.barrier()
        load(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.compute_staged(output=None)
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        for index in range(-args.warmup, rounds):
            load(index)
            if flush is not None:
                flush.fill_(1)
            torch.cuda.synchronize()
            dist.barrier()
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
        stats = b.summarize(b.gather_floats(times, world))
        stats.pop("critical_us", None)
        stats.pop("rank_p50_us", None)
        if rank == 0:
            print(
                "CAKERESULT "
                + json.dumps(
                    dict(
                        tokens=tokens,
                        route=args.route,
                        samples=count,
                        rounds=rounds,
                        table=args.table,
                        calls=calls,
                        **stats,
                    )
                ),
                flush=True,
            )
    torch.cuda.synchronize()
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
