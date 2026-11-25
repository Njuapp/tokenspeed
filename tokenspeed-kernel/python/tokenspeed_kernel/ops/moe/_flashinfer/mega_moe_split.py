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

"""Host side of the SM120 MXFP4 split MegaMoE: weights, staging and launch.

A second MegaMoE for the checkpoint layout of ``mega_moe.py``, on FlashInfer's
CuTe DSL split kernels (``mega_moe_sm120_split``) instead of its persistent
kernel. It is a solution of its own, ``mega_moe_split``: every launch of a
layer built here runs the split kernels, whatever its row count. They run the
routed experts only, so a shared expert stays with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Weights and staged inputs have the persistent kernel's layout.
from tokenspeed_kernel.ops.moe._flashinfer.mega_moe import (
    _check_inputs,
    _get_staging_buffers,
    _order_checkpoint_weights,
    _stage_inputs,
    _stage_warmup_row,
    _swiglu_limit,
)
from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120_split


@dataclass
class _FlashinferSplitMegaMoEState:
    device: torch.device
    # FlashInfer layer; built at warmup.
    layer: object | None


def _state(w: torch.nn.Module) -> _FlashinferSplitMegaMoEState:
    state = w._moe_backend_state
    if not isinstance(state, _FlashinferSplitMegaMoEState):
        raise TypeError("invalid FlashInfer split MegaMoE state")
    return state


def _row_capacity(plan: dict) -> int:
    max_num_tokens = plan.get("persistent_max_num_tokens_per_gpu")
    if max_num_tokens is None or max_num_tokens <= 0:
        raise ValueError(
            "FlashInfer split MegaMoE requires persistent_max_num_tokens_per_gpu"
        )
    return mega_moe_sm120_split.row_capacity(max_num_tokens)


def flashinfer_mxfp4_mega_moe_split_process_weights(
    plan: dict,
    w: torch.nn.Module,
) -> object:
    """Put the routed MXFP4 weights in the kernels' gate/up row order, in place."""
    if not mega_moe_sm120_split.supports_top_k(w.top_k):
        raise ValueError(
            f"The split MegaMoE kernels need an even top-k, got {w.top_k}; run "
            "this model's MoE on the persistent kernel (--moe-backend mega_moe, "
            "or --draft-moe-backend mega_moe when it is the draft model)"
        )
    _row_capacity(plan)
    state = _FlashinferSplitMegaMoEState(
        device=_order_checkpoint_weights(w), layer=None
    )
    w._moe_backend_state = state
    return state


def warmup_flashinfer_mxfp4_mega_moe_split(plan: dict, w: torch.nn.Module) -> None:
    """Build this layer's kernel state and launch every row bucket once. Collective.

    Call on every EP rank, outside CUDA graph capture, after the weights are
    processed. The first layer of a geometry allocates the workspace and the
    kernels' NCCL session and compiles each row bucket; every layer builds one
    launch graph per bucket. A bucket cannot compile under CUDA graph capture,
    which is why all of them are launched here.
    """
    state = _state(w)
    if state.layer is not None:
        return
    process_group = plan.get("process_group")
    if process_group is None:
        raise ValueError("FlashInfer split MegaMoE needs the EP process group")
    capacity = _row_capacity(plan)
    transformed = mega_moe_sm120_split.transform_weights(
        w.w13_weight.data,
        w.w13_weight_scale.data,
        w.w2_weight.data,
        w.w2_weight_scale.data,
        hidden_size=w.hidden_size,
        intermediate_size=w.intermediate_size,
    )
    layer = mega_moe_sm120_split.build_layer(
        process_group=process_group,
        device_index=(
            state.device.index
            if state.device.index is not None
            else torch.cuda.current_device()
        ),
        num_experts=w.num_experts,
        max_tokens_per_rank=capacity,
        hidden_size=w.hidden_size,
        intermediate_size=w.intermediate_size,
        top_k=w.top_k,
        swiglu_limit=_swiglu_limit(w),
        transformed_weights=transformed,
        # FlashInfer's own kernel selection per row bucket.
        knobs=None,
    )
    buffers = _get_staging_buffers(state.device, capacity, w.hidden_size, w.top_k)
    _stage_warmup_row(buffers, w)
    mega_moe_sm120_split.warmup(
        layer,
        buffers.x_fp8[:1],
        buffers.x_scale[:1].view(torch.uint8),
        buffers.topk_ids[:1],
        buffers.topk_weights[:1],
        buckets=mega_moe_sm120_split.row_buckets(capacity),
    )
    state.layer = layer


def destroy_flashinfer_mxfp4_mega_moe_split(w: torch.nn.Module) -> None:
    """Release this layer's kernel state. Collective.

    Call on every EP rank, in the same order, before the EP process group is
    destroyed and once no captured CUDA graph still launches the layer. A
    serving process that exits with its process group needs no call.
    """
    state = _state(w)
    if state.layer is not None:
        mega_moe_sm120_split.destroy(state.layer)
        state.layer = None


def flashinfer_mxfp4_mega_moe_split_run(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    max_num_tokens_per_gpu: int | None,
) -> torch.Tensor:
    """Stage this rank's rows and launch the kernels. Collective.

    Args:
        plan: The layer's MegaMoE plan.
        x: BF16 ``[tokens, hidden]`` rows of this rank; may be empty.
        w: The MoE layer.
        topk_weights: float32 ``[tokens, top_k]`` route weights.
        topk_ids: Global expert ids ``[tokens, top_k]``.
        max_num_tokens_per_gpu: Largest per-rank token count of this forward,
            the same value on every rank. It selects the row bucket the
            kernels run; None runs the bucket of the workspace's capacity.

    Returns:
        BF16 ``[tokens, hidden]``: the routed experts' output for this rank's
        rows.
    """
    state = _state(w)
    capacity = _row_capacity(plan)
    num_tokens = _check_inputs(state.device, capacity, x, w, topk_weights, topk_ids)
    if max_num_tokens_per_gpu is not None and not (
        num_tokens <= max_num_tokens_per_gpu <= capacity
    ):
        raise ValueError(
            f"FlashInfer split MegaMoE got {num_tokens} tokens under a row bound "
            f"of {max_num_tokens_per_gpu}; its workspace holds {capacity}"
        )
    if max_num_tokens_per_gpu == 0:
        # No rank has a row (the bound is the same on every rank) and the
        # kernels have no bucket for it: nothing to launch.
        return torch.empty_like(x, dtype=torch.bfloat16)
    if state.layer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up the FlashInfer split MegaMoE on every EP rank before "
                "CUDA graph capture"
            )
        warmup_flashinfer_mxfp4_mega_moe_split(plan, w)
    buffers = _get_staging_buffers(state.device, capacity, w.hidden_size, w.top_k)
    _stage_inputs(plan, buffers, x, w, topk_weights, topk_ids)
    output = torch.empty_like(x, dtype=torch.bfloat16)
    mega_moe_sm120_split.run(
        state.layer,
        buffers.x_fp8[:num_tokens],
        buffers.x_scale[:num_tokens].view(torch.uint8),
        buffers.topk_ids[:num_tokens],
        buffers.topk_weights[:num_tokens],
        max_tokens_per_rank=max_num_tokens_per_gpu,
        output=output,
        # plan["before_kernel_launch"]: optional callback a caller arms for one
        # forward, run once the inputs are staged and before the kernels are
        # launched (see mega_moe_sm120_split.run). Absent by default.
        before_launch=plan.get("before_kernel_launch"),
    )
    return output
