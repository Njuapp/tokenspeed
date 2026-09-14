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

"""TP8 integration smoke through real K3 communication constructors/dispatch.

Synthetic deferred expert rows isolate the communication integration. This is
not proof that a real model's SiTU producer or serving graph uses the candidate.

Run on a configured TP8 fabric (or use torchrun's multi-node rendezvous flags)::

    torchrun --standalone --nproc-per-node=8 test/runtime/smoke_k3_first_half.py \
        --output-dir /tmp/k3-first-half-smoke

This script checks correctness only; its cloned inputs and diagnostics are
excluded from performance claims. Both optional backends must be installed.
"""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend import initialize_comm_backend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.models.kimi_k3_comm import (
    K3MoeTailComm,
    K3MoeTailCommState,
    K3MoETailTier,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def smoke_shapes(value):
    """Return explicit unique eligible integration shapes, before GPU work."""
    shapes = tuple(
        int(x)
        for x in (
            value or "33,64,255,256,896,1024,1025,1280,1536,2048,3072,4096,8192"
        ).split(",")
    )
    if (
        not shapes
        or len(set(shapes)) != len(shapes)
        or any(not 33 <= m <= 8192 for m in shapes)
    ):
        raise ValueError("requires unique integer M in [33,8192]")
    return shapes


def check(actual, expected, name, exact):
    delta = actual.float() - expected.float()
    maximum = float(delta.abs().max())
    relative = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    )
    allowed = 0.0 if exact else 0.046875 + 0.005 * float(expected.float().abs().max())
    good = (
        torch.equal(actual, expected)
        if exact
        else maximum <= allowed and relative <= 0.006
    )
    vote = torch.tensor(int(good), device=actual.device)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    stats = dict(
        name=name,
        max_abs=maximum,
        relative_l2=relative,
        allowed_max=allowed,
        exact=exact,
    )
    if not vote.item():
        raise AssertionError(stats)
    return stats


def materialize(data):
    rows, weights, indices, _, _ = data
    m = weights.shape[0]
    gathered = rows[indices.clamp_min(0).long()].float()
    gathered.mul_(weights.float().unsqueeze(-1))
    gathered.masked_fill_((indices < 0).unsqueeze(-1), 0)
    return gathered.sum(1).to(torch.bfloat16).view(m, 3584)


def make_data(m, rank, device, slot):
    local = torch.Generator(device=device).manual_seed(3100 + rank + 100 * slot)
    common = torch.Generator(device=device).manual_seed(6200 + 100 * slot)
    rows = (
        torch.randn((m, 3584), generator=local, device=device, dtype=torch.bfloat16)
        * 0.125
    )
    weights = (
        torch.rand((m, 16), generator=common, device=device, dtype=torch.bfloat16)
        * 0.125
    )
    indices = torch.arange(m * 16, device=device, dtype=torch.int32).reshape(m, 16) % m
    indices[:, (rank + slot) % 16] = -1
    shared = (
        torch.randn((m, 7168), generator=local, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    residual = (
        torch.randn((m, 7168), generator=common, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    return rows, weights, indices, shared, residual


def invoke(comm, data):
    rows, weights, indices, shared, residual = data
    shared = shared.clone()
    m = shared.shape[0]
    plan = comm.plan(m, shared, is_decode=False)
    routed = (rows, weights, indices) if plan.defer_finalize else materialize(data)
    return comm.run(plan, routed, shared, residual, m, 7168), plan.tier.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", help="Comma-separated token counts in [33,8192]")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shapes = smoke_shapes(args.shapes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    assert dist.get_world_size() == 8
    os.environ["TOKENSPEED_K3_BT_HT"] = "1"
    global_server_args_dict.update(disable_pdl=False)
    mapping = SimpleNamespace(
        moe=SimpleNamespace(
            tp_size=8, ep_size=1, tp_ep_size=8, tp_ep_group=tuple(range(8))
        ),
        attn=SimpleNamespace(dp_size=1, cp_size=1),
        nprocs_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    global_server_args_dict["mapping"] = mapping
    process_group_manager.register_process_group(
        "nccl", tuple(range(8)), dist.group.WORLD
    )
    backend = initialize_comm_backend(use_pynccl=False)
    assert backend.trtllm_ar.configure_group(
        rank=rank,
        group=tuple(range(8)),
        max_token_num=8192,
        hidden_dim=7168,
        use_fp32_lamport=False,
    )
    execution = SimpleNamespace(
        fused_moe_ar=True, use_native=False, join_moe_reduce=False
    )
    candidate = []
    for slot in range(2):
        generator = torch.Generator(device=device).manual_seed(1900 + slot * 100 + rank)
        # Unqualified control buckets use the actual callable norm, not just
        # its metadata. This is the same RMSNorm module as K3's routed expert.
        norm = RMSNorm(3584, eps=1e-5).to(device=device, dtype=torch.bfloat16)
        norm.requires_grad_(False)
        norm.weight.fill_(0.1 + slot * 0.01)
        up = SimpleNamespace(
            shard_group=tuple(range(8)),
            shard_slice=(rank * 896, 896),
            weight=torch.randn(
                (896, 3584), generator=generator, device=device, dtype=torch.bfloat16
            )
            * 0.02,
        )
        candidate.append(
            K3MoeTailComm(
                mapping=mapping,
                hidden_size=7168,
                prefix=f"smoke.layer{slot}",
                layer_index=slot,
                model_scope="first_half_comm_smoke",
                routed_hidden=3584,
                top_k=16,
                routed_norm=norm,
                up_proj=up,
                execution_plan=execution,
                experts_supports_deferred_finalize=True,
            )
        )
    assert all(c.state.mnnvl_bt_deferred is not None for c in candidate)
    assert all(c.state.mnnvl_ht_deferred is not None for c in candidate)
    assert candidate[0].state is candidate[1].state
    # A separate process-wide-equivalent control state keeps its own BT/HT
    # mailboxes. No live singleton or candidate workspace is reconfigured.
    baseline_state = K3MoeTailCommState(
        mapping=mapping,
        hidden_size=7168,
        latent_size=3584,
        top_k=16,
        rms_eps=1e-5,
        allow_latent_tail=False,
        first_half_enabled=False,
        first_half_capacity=0,
    )
    baseline = []
    for c in candidate:
        b = copy.copy(c)
        b.state = baseline_state
        baseline.append(b)
    # Both actual layers use the model's configured epsilon; gamma remains
    # layer-specific and must be read anew on every host invocation/capture.
    record = dict(
        rank=rank,
        status="running",
        shapes=[],
        boundary="real K3MoeTailComm, synthetic expert rows",
    )
    path = args.output_dir / f"comm-rank-{rank}.json"
    path.write_text(json.dumps(record, indent=2))
    try:
        for m in shapes:
            data = [make_data(m, rank, device, slot) for slot in range(2)]
            stats = dict(M=m, checks=[], tiers=[], control_tiers=[])
            control = []
            for c, d in zip(baseline, data):
                value, tier = invoke(c, d)
                control.append(value.clone())
                stats["control_tiers"].append(tier)
            actual = []
            for c, d in zip(candidate, data):
                out, tier = invoke(c, d)
                expected_tier = (
                    K3MoETailTier.MNNVL_BT_DEFERRED
                    if m <= 1024
                    else K3MoETailTier.MNNVL_HT_DEFERRED
                )
                assert tier == expected_tier.name, (m, tier)
                stats["tiers"].append(tier)
                actual.append(out.clone())
            for a, b in zip(actual, control):
                stats["checks"].append(
                    check(
                        a,
                        b,
                        "original main tail / FP32 finalized synthetic rows",
                        exact=False,
                    )
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = [invoke(c, d)[0] for c, d in zip(candidate, data)]
            for _ in range(128):
                graph.replay()
            torch.cuda.synchronize()
            for a, eager in zip(captured, actual):
                stats["checks"].append(
                    check(a, eager, "full comm eager/128 graph replay", exact=True)
                )
            for slot, d in enumerate(data):
                d[0].mul_(0.9375).add_((slot + rank + 1) * 0.0009765625)
                d[3].mul_(0.875)
                d[4].mul_(1.0625)
                candidate[slot].up_proj.weight.mul_(1.03125)
                candidate[slot].routed_norm.weight.mul_(0.984375)
            changed_control = [invoke(c, d)[0].clone() for c, d in zip(baseline, data)]
            changed_eager = [invoke(c, d)[0].clone() for c, d in zip(candidate, data)]
            graph.replay()
            torch.cuda.synchronize()
            for a, eager, b in zip(captured, changed_eager, changed_control):
                stats["checks"].append(
                    check(
                        a, eager, "changed real comm inputs/weights replay", exact=True
                    )
                )
                stats["checks"].append(
                    check(
                        a,
                        b,
                        "changed original main tail / FP32 finalized rows",
                        exact=False,
                    )
                )
            stats["status"] = "pass"
            record["shapes"].append(stats)
            path.write_text(json.dumps(record, indent=2))
            print(f"rank={rank} full comm M={m} passed", flush=True)
            del graph, captured, actual, control, changed_eager, changed_control, data
        record["status"] = "pass"
    except BaseException as exc:
        record["status"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(record, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
