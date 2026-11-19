# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""Triton DeepSeek V4 selected attention over a dense K/V workspace."""

from __future__ import annotations

import logging
import os

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

logger = logging.getLogger(__name__)

DEEPSEEK_V4_HEAD_DIM = 512

# Three opt-in experiment switches for the SM120 dense-workspace attention,
# read once at import and off by default:
#   TOKENSPEED_SM120_PREFILL_ATTN_H32=1
#       32 heads share each selected K/V tile instead of 16. Regrouping only:
#       the result does not depend on it.
#   TOKENSPEED_SM120_PREFILL_ATTN_PV=tf32x3|bf16x2|bf16
#       PV arithmetic. tf32x3 (default) is today's. bf16x2 splits each FP32
#       probability into two BF16 terms and keeps its precision. bf16 rounds
#       the probabilities to BF16.
#   TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1
#       Requires PV=bf16x2 or PV=bf16. Runs the same arithmetic in the
#       pipelined kernel (32 heads per K/V tile, asynchronous tile prefetch):
#       bit-identical to PV=bf16x2 without it; with PV=bf16 a few stored
#       values in 10^4 round differently (same error). If the kernel
#       cannot be built (it needs nearly all of the device's shared memory),
#       the process warns once and keeps the unpipelined launch of the PV mode.
# With all three at their defaults nothing below the gate is registered and
# every dsv4_prefill call resolves exactly as before.
PREFILL_PV_MODES = ("tf32x3", "bf16x2", "bf16")
PREFILL_VARIANT_KERNEL = "triton_dsv4_prefill_variant_sm120"
# The pipelined kernel addresses the workspace with int32 element offsets.
_PIPELINE_MAX_KV_ROWS = 1 << 22


def _prefill_pipeline_switch(pv: str) -> bool:
    pipeline = os.environ.get("TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE", "0")
    if pipeline not in ("0", "1"):
        raise ValueError(
            f"TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE must be 0 or 1, got {pipeline!r}"
        )
    if pipeline == "1" and pv not in ("bf16x2", "bf16"):
        raise ValueError(
            "TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1 requires "
            f"TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16x2 or bf16, got {pv!r}"
        )
    return pipeline == "1"


def _prefill_variant_switches() -> tuple[bool, str]:
    if "TOKENSPEED_SM120_PREFILL_ATTN_H32_BF16PV" in os.environ:
        raise ValueError(
            "TOKENSPEED_SM120_PREFILL_ATTN_H32_BF16PV was replaced by two switches: "
            "TOKENSPEED_SM120_PREFILL_ATTN_H32=1 and "
            "TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16"
        )
    h32 = os.environ.get("TOKENSPEED_SM120_PREFILL_ATTN_H32", "0")
    if h32 not in ("0", "1"):
        raise ValueError(
            f"TOKENSPEED_SM120_PREFILL_ATTN_H32 must be 0 or 1, got {h32!r}"
        )
    pv = os.environ.get("TOKENSPEED_SM120_PREFILL_ATTN_PV", "tf32x3")
    if pv not in PREFILL_PV_MODES:
        raise ValueError(
            f"TOKENSPEED_SM120_PREFILL_ATTN_PV must be one of {PREFILL_PV_MODES}, "
            f"got {pv!r}"
        )
    return h32 == "1", pv


def _is_sm120() -> bool:
    platform = current_platform()
    return platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)


PREFILL_H32, PREFILL_PV = _prefill_variant_switches()
# The switch as the launch reads it: _pipeline_unavailable turns it off for the
# rest of the process when the pipelined kernel cannot be built.
PREFILL_PIPELINE = _prefill_pipeline_switch(PREFILL_PV)
# The platform is probed only when a switch asks for a variant. The pipeline
# switch implies PV=bf16x2 or PV=bf16, so it never enables anything by itself.
PREFILL_VARIANT_ENABLED = (PREFILL_H32 or PREFILL_PV != "tf32x3") and _is_sm120()

__all__ = [
    "PREFILL_H32",
    "PREFILL_PIPELINE",
    "PREFILL_PV",
    "PREFILL_PV_MODES",
    "PREFILL_VARIANT_ENABLED",
    "PREFILL_VARIANT_KERNEL",
    "triton_dsv4_prefill",
]


@triton.jit
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    num_heads,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    indices_stride_token,
    out_stride_token,
    out_stride_head,
    softmax_scale,
    num_kv_rows,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim = tl.arange(0, BLOCK_DIM)
    dim_mask = dim < HEAD_DIM
    q = tl.load(
        q_ptr + token_idx * q_stride_token + head_idx * q_stride_head + dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    if sink_ptr is not None:
        max_logit = tl.load(sink_ptr + head_idx).to(tl.float32)
        denominator = tl.full((), 1.0, tl.float32)
    else:
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
    accumulator = tl.zeros((BLOCK_DIM,), tl.float32)
    valid_len = tl.minimum(tl.maximum(tl.load(lens_ptr + token_idx), 0), TOPK)
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for start in range(0, TOPK, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        rows = tl.load(
            indices_ptr + token_idx * indices_stride_token + cols,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (rows >= 0) & (rows < num_kv_rows)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(
            kv_ptr + rows[:, None] * kv_stride_row + dim[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(kv * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(max_logit, block_max)
        # Empty shards/tiles must not introduce exp(-inf - -inf).
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(max_logit - safe_max)
        probabilities = tl.exp(logits - safe_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * kv,
            axis=0,
        )
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=0)
        max_logit = next_max

    if lse_ptr is not None:
        logsumexp = tl.where(
            denominator > 0.0, max_logit + tl.log(denominator), -float("inf")
        )
        tl.store(lse_ptr + token_idx * num_heads + head_idx, logsumexp)
    output = tl.where(denominator > 0.0, accumulator / denominator, 0.0)
    tl.store(
        out_ptr + token_idx * out_stride_token + head_idx * out_stride_head + dim,
        output,
        mask=dim_mask,
    )


@register_kernel(
    "attention",
    "dsv4_prefill",
    name="triton_dsv4_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                kv=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"dense_workspace"}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
        "sinks": frozenset({True, False}),
        "return_lse": frozenset({True, False}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run selected shared-KV attention for DeepSeek V4 geometry."""

    if q.dim() != 3 or q.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
    kv_2d = kv.reshape(-1, kv.shape[-1])
    if kv_2d.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected kv rows of width 512, got {tuple(kv.shape)}")
    indices_2d = indices.reshape(indices.shape[0], -1).contiguous()
    lens = lens.reshape(-1).contiguous()
    if indices_2d.shape[0] != q.shape[0] or lens.shape[0] != q.shape[0]:
        raise ValueError("selected-attention metadata must have one row per query")
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    _dsv4_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        lse,
        q.shape[1],
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_scale,
        kv_2d.shape[0],
        TOPK=indices_2d.shape[1],
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        BLOCK_TOPK=16,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
        num_stages=1,
    )
    return output


@register_kernel(
    "attention",
    "dsv4_prefill",
    name="triton_dsv4_prefill_sm120",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                kv=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"dense_workspace"}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
        "sinks": frozenset({True, False}),
        "return_lse": frozenset({True, False}),
    },
    priority=Priority.PERFORMANT,
)
def triton_dsv4_prefill_sm120(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the dsv4_prefill contract with shared K/V tiles on SM120.

    Arguments and returns follow ``dsv4_prefill``. Groups of 16 heads share
    each selected K/V tile. QK uses BF16 tensor cores with FP32 accumulation.
    PV uses TF32x3 to retain FP32 probabilities and the caller's BF16 K/V.
    """
    from tokenspeed_kernel.ops.attention.dsv4._triton.compact import (
        _compact_attention,
    )

    kv_rows = kv.reshape(-1, kv.shape[-1])
    output = torch.empty_like(q) if out is None else out
    head_groups = triton.cdiv(q.shape[1], 16)
    warps = 4 if q.shape[0] * head_groups >= 1024 else 8
    _compact_attention[(q.shape[0], head_groups)](
        q,
        kv_rows,
        indices,
        lens,
        attn_sink,
        output,
        lse,
        *q.stride(),
        *kv_rows.stride(),
        *indices.stride(),
        lens.stride(0),
        0 if attn_sink is None else attn_sink.stride(0),
        *output.stride(),
        indices.shape[1],
        kv_rows.shape[0],
        q.shape[1],
        softmax_scale,
        attn_sink is not None,
        lse is not None,
        32,
        num_warps=warps,
        num_stages=1,
    )
    return output


def _bf16x2_pipelined_serves(q: torch.Tensor, kv_rows: torch.Tensor) -> bool:
    """Whether the pipelined bf16x2 kernel can run a BF16 request.

    It needs whole tiles of 32 heads and a dense workspace of 512-wide rows
    small enough for int32 element offsets.
    """
    return (
        q.shape[1] % 32 == 0
        and kv_rows.stride() == (DEEPSEEK_V4_HEAD_DIM, 1)
        and kv_rows.shape[0] <= _PIPELINE_MAX_KV_ROWS
    )


def _launch_bf16x2_pipelined(
    q: torch.Tensor,
    kv_rows: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    output: torch.Tensor,
    *,
    residual: bool,
) -> None:
    """Launch ``_compact_attention_bf16x2_pipelined`` into ``output``.

    The caller has checked ``_bf16x2_pipelined_serves``. Arguments follow
    ``dsv4_prefill`` with ``kv_rows`` shaped ``[rows, 512]``.

    Args:
        residual: The PV arithmetic. True adds the BF16 residual of every
            probability (PV mode ``bf16x2``); False rounds the probabilities
            to BF16 and multiplies once (PV mode ``bf16``).
    """
    from tokenspeed_kernel.ops.attention.dsv4._triton.compact_variants import (
        _compact_attention_bf16x2_pipelined,
    )

    _compact_attention_bf16x2_pipelined[(q.shape[0], q.shape[1] // 32)](
        q,
        kv_rows,
        indices,
        lens,
        attn_sink,
        output,
        *q.stride(),
        *indices.stride(),
        lens.stride(0),
        0 if attn_sink is None else attn_sink.stride(0),
        *output.stride(),
        indices.shape[1],
        kv_rows.shape[0],
        q.shape[1],
        softmax_scale,
        attn_sink is not None,
        32,
        32,
        DEEPSEEK_V4_HEAD_DIM,
        residual,
        # Eight warps and five stages are part of the kernel's design (see its
        # header); neither depends on the row count.
        num_warps=8,
        num_stages=5,
    )


def _pipeline_unavailable(error: Exception) -> None:
    """Leave the pipelined kernel for the rest of the process, with one warning.

    The kernel takes 100608 of the 101376 bytes of shared memory an SM120
    device offers (100864 with int64 indices), so another Triton release or
    tile constant can make its build fail with OutOfResources at the first
    launch. The unpipelined launch of the same PV mode computes the same
    values (the same bits for bf16x2); the requests the pipelined kernel would
    serve take it from then on, and nothing is retried:
    the module's switch is turned off, so a launch reads no extra state.
    """
    global PREFILL_PIPELINE
    PREFILL_PIPELINE = False
    logger.warning(
        "TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1: the pipelined "
        "attention kernel could not be built or loaded (%s: %s). Prefill "
        "attention uses the unpipelined kernel of the same PV mode for the rest "
        "of this process: the same arithmetic without the speedup.",
        type(error).__name__,
        error,
    )


if PREFILL_VARIANT_ENABLED:

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name=PREFILL_VARIANT_KERNEL,
        solution="triton",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
            "cache_layout": frozenset({"dense_workspace"}),
            "metadata_dtypes": frozenset({torch.int32, torch.int64}),
            "sinks": frozenset({True, False}),
            "return_lse": frozenset({True, False}),
        },
        # Ranked below triton_dsv4_prefill_sm120, which matches the same
        # requests: ranking never picks this kernel. It runs only under an
        # explicit kernel override (dsv4.prefill_attention_override).
        priority=Priority.PORTABLE,
    )
    def triton_dsv4_prefill_variant_sm120(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lens: torch.Tensor,
        attn_sink: torch.Tensor | None,
        softmax_scale: float,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the dsv4_prefill contract with the head grouping, PV arithmetic
        and kernel the environment switches select.

        Arguments and returns follow ``dsv4_prefill``. QK, the softmax and the
        denominator are those of ``triton_dsv4_prefill_sm120``. With 32 heads
        per K/V tile and the default PV the output is bit-identical to it; the
        PV modes ``bf16x2`` and ``bf16`` are not, see the README. The pipeline
        switch changes no bit of the ``bf16x2`` output.

        An override skips signature and trait matching, so requests this
        kernel does not serve are handled here. A partial-attention LSE output
        and inputs that are not BF16 go to ``triton_dsv4_prefill_sm120``. A
        head count that is not a multiple of 32 keeps 16 heads per tile, which
        with the default PV is ``triton_dsv4_prefill_sm120`` again.
        """
        heads_per_tile = 32 if PREFILL_H32 and q.shape[1] % 32 == 0 else 16
        if (
            lse is not None
            or q.dtype != torch.bfloat16
            or kv.dtype != torch.bfloat16
            or (heads_per_tile == 16 and PREFILL_PV == "tf32x3")
        ):
            return triton_dsv4_prefill_sm120(
                q, kv, indices, lens, attn_sink, softmax_scale, out=out, lse=lse
            )
        kv_rows = kv.reshape(-1, kv.shape[-1])
        if PREFILL_PIPELINE and _bf16x2_pipelined_serves(q, kv_rows):
            # The arithmetic of this PV mode's launch below (the same bits for
            # bf16x2); the requests it does not serve take that launch instead.
            output = torch.empty_like(q) if out is None else out
            try:
                _launch_bf16x2_pipelined(
                    q,
                    kv_rows,
                    indices,
                    lens,
                    attn_sink,
                    softmax_scale,
                    output,
                    residual=PREFILL_PV == "bf16x2",
                )
            except (triton.TritonError, RuntimeError) as error:
                # Triton could not build or load the kernel. It raises before
                # anything is launched, so ``output`` is untouched: warn once
                # and take the unpipelined launch below, now and from here on. The
                # try block costs nothing on the path that does not raise.
                _pipeline_unavailable(error)
            else:
                return output
        from tokenspeed_kernel.ops.attention.dsv4._triton.compact_variants import (
            _compact_attention_variant,
        )

        # Tiling per (grouping, PV), fixed for the process; none of it changes
        # the result. tf32x3 keeps the two 256-channel PV tiles of
        # _compact_attention. bf16x2 is fastest split at 16 heads on 4 warps
        # and whole at 32 heads. bf16 always multiplies the whole tile: its
        # split form rounds differently, and the grouping must not change bits.
        split = PREFILL_PV == "tf32x3" or (
            PREFILL_PV == "bf16x2" and heads_per_tile == 16
        )
        warps = 4 if PREFILL_PV == "bf16x2" and heads_per_tile == 16 else 8
        output = torch.empty_like(q) if out is None else out
        grid = (q.shape[0], triton.cdiv(q.shape[1], heads_per_tile))
        _compact_attention_variant[grid](
            q,
            kv_rows,
            indices,
            lens,
            attn_sink,
            output,
            *q.stride(),
            *kv_rows.stride(),
            *indices.stride(),
            lens.stride(0),
            0 if attn_sink is None else attn_sink.stride(0),
            *output.stride(),
            indices.shape[1],
            kv_rows.shape[0],
            q.shape[1],
            softmax_scale,
            attn_sink is not None,
            32,
            heads_per_tile,
            PREFILL_PV,
            split,
            num_warps=warps,
            num_stages=1,
        )
        return output
