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

"""Host side of the SM120 MXFP4 MegaMoE: weights, staging and launch."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (
    stage_mxfp4_mega_moe_inputs,
    stage_mxfp4_mega_moe_inputs_placed,
)

# The MXFP4 MegaMoE kernels share one checkpoint layout and one gate/up row
# order, so the layout helpers are DeepGEMM's.
from tokenspeed_kernel.ops.moe.deep_gemm import (
    _MXFP4_BLOCK_SIZE,
    _expected_shapes,
    _interleave_gate_up_,
)
from tokenspeed_kernel.thirdparty.flashinfer import mega_moe_sm120

logger = logging.getLogger(__name__)

# Staged scales are stored four UE8M0 bytes to an int32, one per 128 columns.
_SCALE_WORD_COLUMNS = 128


@dataclass
class _FlashinferMegaMoEState:
    device: torch.device
    # Shared expert the kernel runs with the routed ones, as (gate_up,
    # gate_up_scale, down, down_scale), or None.
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    # FlashInfer layer; built at warmup, once the shared expert is settled.
    layer: object | None


@dataclass(frozen=True)
class _StagingBuffers:
    """One forward's kernel inputs, written by the staging kernel."""

    x_fp8: torch.Tensor  # E4M3 [capacity, hidden]
    x_scale: torch.Tensor  # int32 [capacity, hidden // 128], four UE8M0 bytes each
    topk_ids: torch.Tensor  # int64 [capacity, top_k]
    topk_weights: torch.Tensor  # float32 [capacity, top_k]


# Layers of one geometry on a device stage through the same buffers: the kernel
# copies them into its workspace before the next layer stages.
_staging_buffers: dict[tuple[torch.device, int, int, int], _StagingBuffers] = {}


def _get_staging_buffers(
    device: torch.device, capacity: int, hidden_size: int, top_k: int
) -> _StagingBuffers:
    key = (device, capacity, hidden_size, top_k)
    buffers = _staging_buffers.get(key)
    if buffers is None:
        buffers = _StagingBuffers(
            # Allocated as bytes: torch has no fill for the FP8 dtype.
            x_fp8=torch.zeros(
                capacity, hidden_size, dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn),
            x_scale=torch.zeros(
                capacity,
                hidden_size // _SCALE_WORD_COLUMNS,
                dtype=torch.int32,
                device=device,
            ),
            topk_ids=torch.zeros(capacity, top_k, dtype=torch.int64, device=device),
            topk_weights=torch.zeros(
                capacity, top_k, dtype=torch.float32, device=device
            ),
        )
        _staging_buffers[key] = buffers
    return buffers


def _state(w: torch.nn.Module) -> _FlashinferMegaMoEState:
    state = w._moe_backend_state
    if not isinstance(state, _FlashinferMegaMoEState):
        raise TypeError("invalid FlashInfer MegaMoE state")
    return state


def _row_capacity(plan: dict) -> int:
    max_num_tokens = plan.get("persistent_max_num_tokens_per_gpu")
    if max_num_tokens is None or max_num_tokens <= 0:
        raise ValueError(
            "FlashInfer MegaMoE requires persistent_max_num_tokens_per_gpu"
        )
    return mega_moe_sm120.row_capacity(max_num_tokens)


def _swiglu_limit(w: torch.nn.Module) -> int:
    """The kernel's integer SwiGLU clamp; 0 leaves the activation unclamped."""
    limit = None if w.swiglu_arg is None else w.swiglu_arg.limit
    if limit is None:
        return 0
    if limit <= 0 or float(limit) != int(limit):
        raise ValueError(
            f"FlashInfer MegaMoE takes a positive integer SwiGLU limit, got {limit}"
        )
    return int(limit)


def _order_checkpoint_weights(w: torch.nn.Module) -> torch.device:
    """Check the routed MXFP4 tensors and put gate/up in the kernels' row order.

    In place. The persistent kernel and the split kernels
    (``mega_moe_split.py``) take the same row order.

    Returns:
        The device the tensors are on.
    """
    w13_weight = w.w13_weight.data
    tensors = (
        w13_weight,
        w.w13_weight_scale.data,
        w.w2_weight.data,
        w.w2_weight_scale.data,
    )
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    expected_shapes = _expected_shapes(
        w.num_local_experts, w.hidden_size, w.intermediate_size
    )
    for name, tensor, expected_shape in zip(names, tensors, expected_shapes):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MegaMoE {name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device.type != "cuda":
            raise ValueError(f"MegaMoE {name} must be loaded on CUDA")
        if tensor.device != w13_weight.device:
            raise ValueError("MegaMoE checkpoint tensors must be on the same device")
        if tensor.dtype != torch.uint8:
            raise ValueError(f"MegaMoE {name} must have dtype uint8")
    # Scales stay in checkpoint order; a kernel derives its own scale layout
    # from them when its layer is built.
    _interleave_gate_up_(w13_weight.view(torch.int8), 8)
    return w13_weight.device


def flashinfer_mxfp4_mega_moe_process_weights(
    plan: dict,
    w: torch.nn.Module,
) -> object:
    """Put the routed MXFP4 weights in the kernel's gate/up row order, in place."""
    _row_capacity(plan)
    state = _FlashinferMegaMoEState(
        device=_order_checkpoint_weights(w), shared=None, layer=None
    )
    w._moe_backend_state = state
    return state


def fuse_flashinfer_mxfp4_mega_moe_shared_expert(
    plan: dict,
    w: torch.nn.Module,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
) -> bool:
    """Run one MXFP8 shared expert inside each MegaMoE launch.

    The shared expert must have the routed experts' intermediate size. The
    kernel keeps its own layout of the tensors; they are left unchanged.

    Args:
        plan: The layer's MegaMoE plan.
        w: The MoE layer, with its routed weights already processed.
        gate_up_weight: E4M3 ``[2 * intermediate, hidden]``, gate rows first.
        gate_up_scale: UE8M0 bytes ``[2 * intermediate, hidden // 32]``.
        down_weight: E4M3 ``[hidden, intermediate]``.
        down_scale: UE8M0 bytes ``[hidden, intermediate // 32]``.

    Returns:
        True once the kernel will run the shared expert; False when the shapes
        or dtypes do not fit one routed expert's MXFP8 layout.
    """
    state = _state(w)
    if state.shared is not None:
        raise RuntimeError("MegaMoE already runs a shared expert")
    if state.layer is not None:
        raise RuntimeError("Fuse the shared expert before the MegaMoE warmup")
    hidden, inter = w.hidden_size, w.intermediate_size
    expected = (
        (gate_up_weight, (2 * inter, hidden), torch.float8_e4m3fn),
        (gate_up_scale, (2 * inter, hidden // _MXFP4_BLOCK_SIZE), torch.uint8),
        (down_weight, (hidden, inter), torch.float8_e4m3fn),
        (down_scale, (hidden, inter // _MXFP4_BLOCK_SIZE), torch.uint8),
    )
    if any(
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != state.device
        or not tensor.is_contiguous()
        for tensor, shape, dtype in expected
    ):
        return False
    state.shared = (gate_up_weight, gate_up_scale, down_weight, down_scale)
    return True


def warmup_flashinfer_mxfp4_mega_moe(plan: dict, w: torch.nn.Module) -> None:
    """Build this layer's kernel state and launch it once. Collective.

    Call on every EP rank, outside CUDA graph capture, after the weights are
    processed and the shared expert is fused or left out. The first layer of a
    geometry compiles the kernel and allocates the communication windows.
    """
    state = _state(w)
    if state.layer is not None:
        return
    transport = plan.get("mega_moe_transport")
    if transport is None:
        raise ValueError(
            "FlashInfer MegaMoE needs a transport table: pass mega_moe_transport, "
            f"one of {list(mega_moe_sm120.TRANSPORT_TABLES)}"
        )
    process_group = plan.get("process_group")
    if process_group is None:
        raise ValueError("FlashInfer MegaMoE needs the EP process group")
    capacity = _row_capacity(plan)
    transformed = mega_moe_sm120.transform_weights(
        w.w13_weight.data,
        w.w13_weight_scale.data,
        w.w2_weight.data,
        w.w2_weight_scale.data,
        hidden_size=w.hidden_size,
        intermediate_size=w.intermediate_size,
        shared=state.shared,
    )
    layer = mega_moe_sm120.build_layer(
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
        transport_table=transport,
        transformed_weights=transformed,
        has_shared_expert=state.shared is not None,
    )
    buffers = _get_staging_buffers(state.device, capacity, w.hidden_size, w.top_k)
    # One zero row routed to the first experts: enough to allocate the
    # workspace, load the kernel and encode this layer's weight descriptors.
    _stage_warmup_row(buffers, w)
    mega_moe_sm120.warmup(
        layer,
        buffers.x_fp8[:1],
        buffers.x_scale[:1].view(torch.uint8),
        buffers.topk_ids[:1],
        buffers.topk_weights[:1],
    )
    state.layer = layer


def _stage_warmup_row(buffers: _StagingBuffers, w: torch.nn.Module) -> None:
    """Stage one zero row routed to the layer's first experts."""
    buffers.x_fp8[:1].view(torch.uint8).zero_()
    buffers.x_scale[:1].fill_(0x7F7F7F7F)
    buffers.topk_ids[:1].copy_(
        torch.arange(w.top_k, device=buffers.topk_ids.device) % w.num_experts
    )
    buffers.topk_weights[:1].fill_(1.0 / w.top_k)


def destroy_flashinfer_mxfp4_mega_moe(w: torch.nn.Module) -> None:
    """Release this layer's kernel state. Collective.

    Call on every EP rank, in the same order, before the EP process group is
    destroyed and once no captured CUDA graph still launches the layer. A
    serving process that exits with its process group needs no call.
    """
    state = _state(w)
    if state.layer is not None:
        mega_moe_sm120.destroy(state.layer)
        state.layer = None


def flashinfer_mxfp4_mega_moe_run(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    max_num_tokens_per_gpu: int | None,
) -> torch.Tensor:
    """Stage this rank's rows and launch the kernel. Collective.

    Args:
        plan: The layer's MegaMoE plan.
        x: BF16 ``[tokens, hidden]`` rows of this rank; may be empty.
        w: The MoE layer.
        topk_weights: float32 ``[tokens, top_k]`` route weights.
        topk_ids: Global expert ids ``[tokens, top_k]``.
        max_num_tokens_per_gpu: Largest per-rank token count of this forward,
            the same value on every rank; required under CUDA graph capture.

    Returns:
        BF16 ``[tokens, hidden]``: the routed experts' output for this rank's
        rows, including the shared expert when it is fused.
    """
    state = _state(w)
    capacity = _row_capacity(plan)
    num_tokens = _check_inputs(state.device, capacity, x, w, topk_weights, topk_ids)
    if state.layer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up the FlashInfer MegaMoE on every EP rank before CUDA "
                "graph capture"
            )
        warmup_flashinfer_mxfp4_mega_moe(plan, w)
    buffers = _get_staging_buffers(state.device, capacity, w.hidden_size, w.top_k)
    _stage_inputs(plan, buffers, x, w, topk_weights, topk_ids)
    output = torch.empty_like(x, dtype=torch.bfloat16)
    # plan["before_kernel_launch"]: optional callback a caller arms for one
    # forward, run once the inputs are staged and before the kernel launch
    # (see mega_moe_sm120.run). Absent by default.
    before_launch = plan.get("before_kernel_launch")
    if before_launch is None:
        mega_moe_sm120.run(
            state.layer,
            buffers.x_fp8[:num_tokens],
            buffers.x_scale[:num_tokens].view(torch.uint8),
            buffers.topk_ids[:num_tokens],
            buffers.topk_weights[:num_tokens],
            max_tokens_per_rank=max_num_tokens_per_gpu,
            output=output,
        )
    else:
        mega_moe_sm120.run(
            state.layer,
            buffers.x_fp8[:num_tokens],
            buffers.x_scale[:num_tokens].view(torch.uint8),
            buffers.topk_ids[:num_tokens],
            buffers.topk_weights[:num_tokens],
            max_tokens_per_rank=max_num_tokens_per_gpu,
            output=output,
            before_launch=before_launch,
        )
    return output


def _check_inputs(
    device: torch.device,
    capacity: int,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> int:
    """Check one forward's inputs against the layer.

    Returns:
        The rows of this rank.
    """
    if x.device != device:
        raise ValueError("MegaMoE inputs and processed weights must share a device")
    if x.ndim != 2 or x.shape[1] != w.hidden_size:
        raise ValueError(
            f"FlashInfer MegaMoE input must have shape [tokens, {w.hidden_size}]"
        )
    num_tokens = x.shape[0]
    expected_routing_shape = (num_tokens, w.top_k)
    if topk_weights is None or tuple(topk_weights.shape) != expected_routing_shape:
        raise ValueError(f"topk_weights must have shape {expected_routing_shape}")
    if topk_ids is None or tuple(topk_ids.shape) != expected_routing_shape:
        raise ValueError(f"topk_ids must have shape {expected_routing_shape}")
    if num_tokens > capacity:
        raise ValueError(
            f"FlashInfer MegaMoE got {num_tokens} tokens, but its workspace "
            f"holds {capacity}"
        )
    return num_tokens


def _stage_inputs(
    plan: dict,
    buffers: _StagingBuffers,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> None:
    """Quantize this rank's rows and write the kernel inputs into ``buffers``."""
    num_tokens = x.shape[0]
    # A permuted expert placement (plan["expert_id_map"], int32 logical id ->
    # slot): the weights were loaded into the placed slots, so the ids the
    # kernel receives must name slots. The staging launch maps them.
    expert_id_map = plan.get("expert_id_map")
    # Staging widens ids to int64; one input dtype keeps one compile.
    if expert_id_map is None:
        stage_mxfp4_mega_moe_inputs(
            x,
            topk_weights,
            topk_ids.to(torch.int32),
            buffers.x_fp8[:num_tokens],
            buffers.x_scale[:num_tokens],
            buffers.topk_ids[:num_tokens],
            buffers.topk_weights[:num_tokens],
        )
    else:
        # The map is indexed by logical expert id and names slots; replicas
        # make the slots (w.num_experts) more than the logical experts.
        if expert_id_map.shape[0] > w.num_experts:
            raise ValueError(
                f"expert_id_map covers {expert_id_map.shape[0]} experts, the "
                f"layer holds only {w.num_experts} slots"
            )
        stage_mxfp4_mega_moe_inputs_placed(
            x,
            topk_weights,
            topk_ids.to(torch.int32),
            buffers.x_fp8[:num_tokens],
            buffers.x_scale[:num_tokens],
            buffers.topk_ids[:num_tokens],
            buffers.topk_weights[:num_tokens],
            expert_id_map,
        )
