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

"""FlashInfer's SM120 MXFP4 x MXFP8 MegaMoE (``sm120_mxfp4_mxfp8_bf16_cake``).

One persistent kernel per EP rank dispatches rows to the experts' ranks, runs
the routed experts and returns the results. Each pair of ranks exchanges data
over NCCL symmetric windows, either by load/store through a peer mapping (LSA,
PCIe or NVLink) or by GPU-initiated networking puts (GIN), so the same kernel
serves a PCIe-only host and an EP group spanning hosts.

FlashInfer imports stay inside the functions: importing ``flashinfer.moe_ep``
loads every MoE-EP backend, which only an SM120 MegaMoE deployment needs.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import torch

# How each pair of EP ranks exchanges data. "all_gin" uses GIN for every pair
# and is the only table valid on every topology; "all_lsa" needs every pair to
# be peer-mapped (one host); "lsa_where_reachable" uses LSA inside a host and
# GIN across hosts; "auto" probes the candidates at startup; "tuned" reads
# FlashInfer's offline knob cache.
TRANSPORT_TABLES = ("all_gin", "all_lsa", "lsa_where_reachable", "auto", "tuned")
# The kernel's row capacity is a multiple of 128 rows, at most 32640.
ROW_ALIGNMENT = 128
MAX_ROWS = 32640

_BACKEND_PACKAGE = "moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cake"


def available() -> bool:
    """Whether the installed FlashInfer ships the SM120 MegaMoE backend.

    Looks for the backend's files without importing FlashInfer.
    """
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        (Path(location) / _BACKEND_PACKAGE / "__init__.py").is_file()
        for location in spec.submodule_search_locations
    )


def row_capacity(max_tokens_per_rank: int) -> int:
    """Rows the kernel is compiled for: ``max_tokens_per_rank`` rounded up to 128.

    Args:
        max_tokens_per_rank: Largest per-rank token count of one forward.

    Returns:
        The kernel's compile-time row capacity.
    """
    if max_tokens_per_rank <= 0:
        raise ValueError(
            f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}"
        )
    capacity = -(-max_tokens_per_rank // ROW_ALIGNMENT) * ROW_ALIGNMENT
    if capacity > MAX_ROWS:
        raise ValueError(
            f"SM120 MegaMoE holds at most {MAX_ROWS} rows per rank, got "
            f"{max_tokens_per_rank}"
        )
    return capacity


def transform_weights(
    w13_interleaved: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> object:
    """Lay this rank's expert weights out for the kernel.

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
        shared: The shared expert the kernel runs with the routed ones, as
            ``(gate_up, gate_up_scale, down, down_scale)``: E4M3 ``[2 *
            intermediate, hidden]`` gate rows first, UE8M0 bytes ``[2 *
            intermediate, hidden // 32]``, E4M3 ``[hidden, intermediate]``,
            UE8M0 bytes ``[hidden, intermediate // 32]``. None for routed
            experts only.

    Returns:
        FlashInfer's kernel-layout weights for ``build_layer``.
    """
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
        transform_prequantized_weights,
    )

    shared_kwargs = {}
    if shared is not None:
        gate_up, gate_up_scale, down, down_scale = shared
        shared_kwargs = {
            "shared_w13": gate_up,
            "shared_w2": down,
            "shared_w13_scale": gate_up_scale,
            "shared_w2_scale": down_scale,
        }
    transformed = transform_prequantized_weights(
        w13_interleaved,
        w2,
        w13_scale,
        w2_scale,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        **shared_kwargs,
    )
    # FlashInfer interleaves the gate/up rows it is given. Ours already are,
    # so keep the caller's tensor and drop the twice-permuted copy.
    return dataclasses.replace(transformed, w1_fp4=w13_interleaved.view(torch.uint8))


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
    transport_table: str,
    transformed_weights: object,
    has_shared_expert: bool,
) -> object:
    """Create this rank's MegaMoE layer over an EP process group.

    Collective: every rank of ``process_group`` calls it with the same
    arguments. Layers of one geometry share one communication workspace.

    Args:
        process_group: torch.distributed NCCL group of the EP ranks. The
            kernel's windows are registered on its communicator.
        device_index: CUDA device this rank runs on.
        num_experts: Routed experts across the EP group.
        max_tokens_per_rank: Largest per-rank token count of one forward.
        hidden_size: Model hidden size.
        intermediate_size: Expert intermediate size.
        top_k: Routed experts per token.
        swiglu_limit: SwiGLU clamp ``gate <= limit``, ``|up| <= limit``; 0
            leaves the activation unclamped.
        transport_table: One of ``TRANSPORT_TABLES``.
        transformed_weights: Result of ``transform_weights``.
        has_shared_expert: Whether ``transformed_weights`` carries a shared
            expert for the kernel to run.

    Returns:
        FlashInfer's ``MoEEpMegaLayer``.
    """
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    if transport_table not in TRANSPORT_TABLES:
        raise ValueError(
            f"transport_table must be one of {list(TRANSPORT_TABLES)}, got "
            f"{transport_table!r}"
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
    config = Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
        intermediate_size=intermediate_size,
        top_k=top_k,
        swiglu_limit=swiglu_limit,
        shared_expert="fp8_in_kernel" if has_shared_expert else None,
        transport_table=transport_table,
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
        scales=x_scale,
    )


def warmup(
    layer: object,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Run one eager forward so later calls can be captured into a CUDA graph.

    Collective, outside graph capture. The first call of a geometry allocates
    its workspace and loads the compiled kernel; each layer's first call
    encodes that layer's weight descriptors.

    Args:
        layer: Result of ``build_layer``.
        x_fp8: E4M3 codes ``[rows, hidden]`` as uint8.
        x_scale: UE8M0 bytes ``[rows, hidden // 32]``.
        topk_ids: int64 global expert ids ``[rows, top_k]``.
        topk_weights: float32 route weights ``[rows, top_k]``.
    """
    layer.warmup(_tensors(x_fp8, x_scale, topk_ids, topk_weights))


def run(
    layer: object,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    max_tokens_per_rank: int | None,
    output: torch.Tensor,
) -> None:
    """Launch the kernel on this rank's rows. Collective over the EP group.

    Args:
        layer: Result of ``build_layer``, already warmed up.
        x_fp8: E4M3 codes ``[rows, hidden]`` as uint8.
        x_scale: UE8M0 bytes ``[rows, hidden // 32]``.
        topk_ids: int64 global expert ids ``[rows, top_k]``; -1 masks a route.
        topk_weights: float32 route weights ``[rows, top_k]``.
        max_tokens_per_rank: Largest per-rank row count of this call, the same
            value on every rank. The call then needs no host synchronization
            and can be captured into a CUDA graph. None makes the kernel
            all-gather the row counts and read them back on the host.
        output: BF16 ``[rows, hidden]`` to receive this rank's rows.
    """
    layer.stage_inputs(
        _tensors(x_fp8, x_scale, topk_ids, topk_weights),
        compile_tokens_per_rank=max_tokens_per_rank,
    )
    layer.compute_staged(output=output)


def destroy(layer: object) -> None:
    """Release a layer's share of the communication windows. Collective.

    Call on every EP rank, in the same order, before the EP process group is
    destroyed and once no captured CUDA graph still launches the layer.

    Args:
        layer: Result of ``build_layer``.
    """
    layer.destroy()
