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

"""Map contiguous expert partitions without allocating a lookup table."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["COUNT", "WIDTH", "S0", "S1", "START", "EXPERTS"])
def _local_expert_ids(Ids, Out, COUNT, WIDTH, S0, S1, START, EXPERTS, B: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    live = offset < COUNT
    row, col = offset // WIDTH, offset % WIDTH
    expert = tl.load(Ids + row * S0 + col * S1, live, -1).to(tl.int32)
    local = expert - START
    mapped = tl.where((local >= 0) & (local < EXPERTS), local, -1)
    mapped = tl.where(expert < 0, expert, mapped)
    tl.store(Out + offset, mapped, live)


def local_expert_ids(
    topk_ids: torch.Tensor, expert_start: int, num_local_experts: int
) -> torch.Tensor:
    """Map global IDs to a rank's contiguous expert partition.

    Args:
        topk_ids: CUDA int32/int64 [tokens, routes], with arbitrary strides.
            Nonnegative IDs name valid global experts. Negative values are
            existing masked routes and are retained.
        expert_start: First global expert owned by this rank.
        num_local_experts: Number of consecutive experts owned by this rank.

    Returns:
        Fresh contiguous int32 IDs of the same shape. Owned experts become
        zero-based local IDs. Other nonnegative experts become -1.
    """
    if (
        topk_ids.ndim != 2
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_ids.is_cuda
    ):
        raise ValueError("expert IDs must be CUDA int32/int64 [tokens, routes]")
    if expert_start < 0 or num_local_experts <= 0:
        raise ValueError(
            "the expert partition must have a nonnegative start and positive size"
        )
    output = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)
    if topk_ids.numel():
        _local_expert_ids[(triton.cdiv(topk_ids.numel(), 256),)](
            topk_ids,
            output,
            topk_ids.numel(),
            topk_ids.shape[1],
            topk_ids.stride(0),
            topk_ids.stride(1),
            expert_start,
            num_local_experts,
            256,
            num_warps=4,
        )
    return output
