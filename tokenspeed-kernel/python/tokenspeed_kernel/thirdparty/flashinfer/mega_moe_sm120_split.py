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

FlashInfer imports stay inside the functions, as in ``mega_moe_sm120``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import torch

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
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import workspace_compile_buckets

    return workspace_compile_buckets(row_capacity)


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
    from flashinfer.moe_ep.shims.sm120.split_cutedsl import (
        FP4_DTYPE,
        transform_prequantized_weights,
    )

    (_, fc1_scale), (fc2_weight, fc2_scale) = transform_prequantized_weights(
        w13_interleaved,
        w2,
        w13_scale,
        w2_scale,
        hidden=hidden_size,
        intermediate=intermediate_size,
    )
    # FlashInfer interleaves the gate/up rows it is given. Ours already are,
    # so keep the caller's tensor and drop the twice-permuted copy.
    return (w13_interleaved.view(FP4_DTYPE), fc1_scale), (fc2_weight, fc2_scale)


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
            bucket. None keeps FlashInfer's selection.

    Returns:
        FlashInfer's ``MoEEpMegaLayer``.
    """
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
