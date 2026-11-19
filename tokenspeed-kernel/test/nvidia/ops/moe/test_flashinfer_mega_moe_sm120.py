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

"""SM120 MXFP4 MegaMoE over an expert-parallel group of RTX-class GPUs.

Every rank holds its own token rows and one shard of the experts; the
registered kernel must return, per rank, the routed (and shared) expert output
for that rank's rows. Normal one-process pytest runs skip this file. Launch one
process per GPU on 4, 8 or 16 SM120 GPUs:

``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``

or across two hosts with ``torchrun --nnodes 2 --nproc-per-node 8
--rdzv_backend c10d --rdzv_endpoint <host>:<port>``. ``MEGA_MOE_TRANSPORT``
selects how the ranks exchange rows (default ``all_gin``).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


if not (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    # 384 experts over at most 128 per rank.
    and _world_size() in {4, 8, 16}
):
    pytest.skip(
        "SM120 MegaMoE needs a torchrun launch on 4, 8 or 16 SM120 GPUs",
        allow_module_level=True,
    )

from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120  # noqa: E402

if not mega_moe_sm120.available():
    pytest.skip(
        "the installed FlashInfer has no SM120 MegaMoE", allow_module_level=True
    )

from tokenspeed_kernel.ops.moe import moe_process_weights  # noqa: E402
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan  # noqa: E402
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (  # noqa: E402
    _stage_mxfp4_mega_moe_inputs_kernel,
)
from tokenspeed_kernel.ops.moe._flashinfer.mega_moe import (  # noqa: E402
    destroy_flashinfer_mxfp4_mega_moe,
    fuse_flashinfer_mxfp4_mega_moe_shared_expert,
)
from utils import assert_no_triton_compile  # noqa: E402

from tokenspeed.runtime.layers.activation import SwigluArg  # noqa: E402

# DeepSeek-V4.1-Flash routed experts.
EXPERTS, HIDDEN, INTER, TOP_K, LIMIT = 384, 5120, 2304, 6, 10.0
# Above 512 rows per rank the kernel switches from 64-row to 128-row tasks.
MAX_TOKENS = 640
# Rows per rank in the reference comparisons; the reference decodes every
# routed expert, so it stays small while the bound selects the task size.
REFERENCE_ROWS = 40
# Weight scales 2^-9 .. 2^-6 keep the first projection O(1), inside the clamp.
ROUTED_EXPONENTS = (118, 122)
SHARED_EXPONENTS = (116, 119)
LOG2_E = 1.4426950408889634


def _transport() -> str:
    return os.environ.get("MEGA_MOE_TRANSPORT", "all_gin")


def _expert_weights(expert: int):
    """One routed expert's packed MXFP4 tensors, the same on every rank."""
    g = torch.Generator(device="cuda").manual_seed(1000 + expert)

    def packed(rows, columns):
        return torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, device="cuda", generator=g
        )

    def scales(rows, columns):
        return torch.randint(
            *ROUTED_EXPONENTS,
            (rows, columns // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )

    return (
        packed(2 * INTER, HIDDEN),
        scales(2 * INTER, HIDDEN),
        packed(HIDDEN, INTER),
        scales(HIDDEN, INTER),
    )


def _shared_weights():
    """The MXFP8 shared expert, the same on every rank."""
    g = torch.Generator(device="cuda").manual_seed(7)

    def codes(rows, columns):
        values = torch.randn(rows, columns, device="cuda", generator=g) * 4.0
        return values.to(torch.float8_e4m3fn)

    def scales(rows, columns):
        return torch.randint(
            *SHARED_EXPONENTS,
            (rows, columns // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )

    return (
        codes(2 * INTER, HIDDEN),
        scales(2 * INTER, HIDDEN),
        codes(HIDDEN, INTER),
        scales(HIDDEN, INTER),
    )


class _Experts(nn.Module):
    """This rank's routed weights and the fields the MegaMoE kernel reads."""

    def __init__(self, rank: int, world_size: int):
        super().__init__()
        local = EXPERTS // world_size
        parts = [
            _expert_weights(expert)
            for expert in range(rank * local, (rank + 1) * local)
        ]

        def stacked(index):
            return nn.Parameter(
                torch.stack([part[index] for part in parts]), requires_grad=False
            )

        self.w13_weight = stacked(0)
        self.w13_weight_scale = stacked(1)
        self.w2_weight = stacked(2)
        self.w2_weight_scale = stacked(3)
        self.num_local_experts = local
        self.num_experts = EXPERTS
        self.hidden_size, self.intermediate_size, self.top_k = HIDDEN, INTER, TOP_K
        self.swiglu_arg = SwigluArg(alpha=None, limit=LIMIT)
        self._moe_backend_state = None


def _scale(exponents):
    return (exponents.to(torch.int32) << 23).view(torch.float32)


def _decode_fp4(packed, exponents):
    """Low-nibble-first E2M1 ``[N, K // 2]`` with UE8M0 ``[N, K // 32]`` to fp32."""
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    return values * _scale(exponents).repeat_interleave(32, dim=-1)


def _decode_fp8(codes, exponents):
    return codes.float() * _scale(exponents).repeat_interleave(32, dim=-1)


def _fp8_round_trip(values, *, floor):
    """E4M3 with one power-of-two scale per 32 values, decoded back to fp32.

    ``floor`` is the input staging's lower bound on a group's magnitude; the
    kernel requantizes its intermediate without one.
    """
    rows, columns = values.shape
    groups = values.float().view(rows, columns // 32, 32)
    amax = groups.abs().amax(-1)
    if floor:
        amax = amax.clamp_min(1e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    exponent = exponent.clamp(1 if floor else 0, 254)
    inverse = ((254 - exponent) << 23).view(torch.float32)[..., None]
    scale = (exponent << 23).view(torch.float32)[..., None]
    codes = (groups * inverse).to(torch.float8_e4m3fn).float()
    return (codes * scale).view(rows, columns)


def _expert_partial(x, w13, w2, route_weight):
    """One expert's BF16 rows on the kernel's numerical boundary."""
    fc1 = (x @ w13.t()).to(torch.bfloat16).float()
    gate = fc1[:, :INTER].clamp(max=LIMIT)
    up = fc1[:, INTER:].clamp(min=-LIMIT, max=LIMIT)
    sigmoid = 1.0 / (1.0 + torch.exp2(-gate * LOG2_E))
    hidden = _fp8_round_trip(gate * sigmoid * up * route_weight[:, None], floor=False)
    return (hidden @ w2.t()).to(torch.bfloat16).float()


def _reference(x, topk_weights, topk_ids, shared):
    """Slot-ordered fp32 sum of the routed BF16 partials, plus the shared one."""
    x = _fp8_round_trip(x, floor=True)
    partials = torch.zeros(x.shape[0], TOP_K, HIDDEN, device=x.device)
    for expert in torch.unique(topk_ids).tolist():
        rows, slots = torch.where(topk_ids == expert)
        w13, w13_scale, w2, w2_scale = _expert_weights(expert)
        partials[rows, slots] = _expert_partial(
            x[rows],
            _decode_fp4(w13, w13_scale),
            _decode_fp4(w2, w2_scale),
            topk_weights[rows, slots],
        )
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for slot in range(TOP_K):
        out += partials[:, slot]
    if shared is not None and x.shape[0]:
        gate_up, gate_up_scale, down, down_scale = shared
        out += _expert_partial(
            x,
            _decode_fp8(gate_up, gate_up_scale),
            _decode_fp8(down, down_scale),
            torch.ones(x.shape[0], device=x.device),
        )
    return out.to(torch.bfloat16)


def _inputs(tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    logits = torch.randn(tokens, EXPERTS, device="cuda", generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, weights.float().contiguous(), ids.to(torch.int32).contiguous()


def _rows(rank: int, world_size: int, bound: int) -> int:
    """Different row counts per rank under one bound; the last rank is idle."""
    if rank == world_size - 1:
        return 0
    return min(bound - (5 * rank) % (bound // 2), REFERENCE_ROWS)


def _assert_close(actual, reference):
    assert torch.isfinite(actual.float()).all()
    if reference.numel() == 0:
        return
    # The fp32 reference and the kernel's tensor-core accumulation order differ
    # on a few E4M3 requantization codes at rounding boundaries; one flipped
    # code moves an output by a fraction of the output range, so the bound is
    # 2% of that range and 2% in relative L2.
    actual, reference = actual.float(), reference.float()
    bound = 0.02 * float(reference.abs().max())
    assert float((actual - reference).abs().max()) <= bound
    rel_l2 = torch.linalg.norm(actual - reference) / torch.linalg.norm(reference)
    assert float(rel_l2) <= 0.02


@pytest.fixture(scope="module")
def ep_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
        )
    yield dist.get_rank(), dist.get_world_size()
    dist.destroy_process_group()


@pytest.fixture(scope="module", params=["routed", "fused_shared"])
def layer(request, ep_group):
    rank, world_size = ep_group
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=world_size,
        ispp=INTER,
        hidden=HIDDEN,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        num_experts=EXPERTS,
        top_k=TOP_K,
        process_group=dist.group.WORLD,
        persistent_max_num_tokens_per_gpu=MAX_TOKENS,
        mega_moe_transport=_transport(),
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="mega_moe",
    )
    assert plan["apply_kernel_name"] == "flashinfer_mxfp4_mega_moe_apply"
    w = _Experts(rank, world_size)
    moe_process_weights(plan, w)
    shared = None
    if request.param == "fused_shared":
        shared = _shared_weights()
        assert fuse_flashinfer_mxfp4_mega_moe_shared_expert(plan, w, *shared)
    plan["warmup"](plan, w)
    yield plan, w, shared, rank, world_size
    torch.cuda.synchronize()
    dist.barrier()
    destroy_flashinfer_mxfp4_mega_moe(w)


def _apply(plan, w, x, weights, ids, bound):
    return moe_apply(
        plan,
        x,
        w,
        None,
        topk_weights=weights,
        topk_ids=ids,
        num_tokens_global=None,
        max_num_tokens_per_gpu=bound,
    )


@pytest.mark.parametrize("bound", [8, 64, MAX_TOKENS])
def test_output_matches_reference(layer, bound):
    plan, w, shared, rank, world_size = layer
    x, weights, ids = _inputs(_rows(rank, world_size, bound), seed=100 * bound + rank)
    out = _apply(plan, w, x, weights, ids, bound)
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    _assert_close(out, _reference(x, weights, ids, shared))


def test_row_counts_exchanged_by_the_kernel_match_a_given_bound(layer):
    """Without a bound the kernel all-gathers the row counts itself."""
    plan, w, _, rank, world_size = layer
    x, weights, ids = _inputs(_rows(rank, world_size, 64), seed=300 + rank)
    exchanged = _apply(plan, w, x, weights, ids, None).clone()
    # The exact largest row count reproduces the exchanged task layout.
    largest = max(_rows(r, world_size, 64) for r in range(world_size))
    bounded = _apply(plan, w, x, weights, ids, largest)
    assert torch.equal(exchanged, bounded)


def test_cuda_graph_replay_matches_eager(layer):
    plan, w, shared, rank, world_size = layer
    bound = 16
    tokens = _rows(rank, world_size, bound)
    static = _inputs(tokens, seed=400 + rank)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _apply(plan, w, *static, bound)
    torch.cuda.synchronize()
    dist.barrier()
    for step in range(3):
        fresh = _inputs(tokens, seed=500 + 10 * step + rank)
        for target, source in zip(static, fresh):
            target.copy_(source)
        graph.replay()
        replayed = captured.clone()
        eager = _apply(plan, w, *fresh, bound)
        assert torch.equal(replayed, eager)
        _assert_close(replayed, _reference(*fresh, shared))


def test_token_count_does_not_recompile_staging(layer):
    plan, w, _, rank, world_size = layer

    def run(bound):
        # The last rank stays idle; an idle rank launches no staging at all.
        tokens = 0 if rank == world_size - 1 else bound
        _apply(plan, w, *_inputs(tokens, seed=600 + bound), bound)

    for bound in (8, 64, 256):
        run(bound)
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for bound in (2, 17, 63, 65, 200):
            run(bound)


def test_more_tokens_than_the_workspace_holds_is_refused(layer):
    plan, w, _, _, _ = layer
    x, weights, ids = _inputs(MAX_TOKENS + 1, seed=800)
    # Refused on the host before any launch, so the ranks stay in step.
    with pytest.raises(ValueError, match="workspace holds"):
        _apply(plan, w, x, weights, ids, MAX_TOKENS)
