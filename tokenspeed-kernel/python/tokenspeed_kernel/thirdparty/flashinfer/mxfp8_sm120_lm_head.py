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

"""MXFP8 copy of a BF16 LM head on SM120 for the DSpark base logits.

A BF16 head of ``V x K`` is read in full for every batch of rows. Its
group-32 MXFP8 copy is half the bytes. Experiment code behind
``TOKENSPEED_SM120_LM_HEAD_MXFP8=draft``: only the draft's base logits use
the copy, and every proposal is still verified by the target's BF16 head.

The verify pass is deliberately not served from the copy. Re-scoring the
tokens near each row's maximum against the BF16 head reproduced today's
argmax in all but 1 of 18,630 synthetic rows, but it is not provably
identical: no usable bound on the MXFP8 logit error exists (the
Cauchy-Schwarz bound is tens of times the observed error), a candidate window
can overflow on flat rows, and the re-scored FP32 sum rounds to BF16
differently from cuBLAS in about 0.08% of logits.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel.platform import pdl_enabled

# One b12x specialization per output dtype serves every row count: the launch
# is bound by reading the weight, so the tile hardly matters.
_TILE = (64, 128)
# (device index, data pointer of the BF16 head) -> (codes [V, K], scales)
_HEADS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _head_key(head: torch.Tensor) -> tuple[int, int]:
    return (head.device.index or 0, head.data_ptr())


def mxfp8_lm_head(head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The MXFP8 copy prepared for this exact BF16 head tensor, or None."""
    return _HEADS.get(_head_key(head)) if head.is_cuda else None


@functools.cache
def _runner(out_dtype: torch.dtype, enable_pdl: bool):
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_runner

    return _b12x_gemm_mxfp8_runner(12, 0, enable_pdl, out_dtype)


def _gemm(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], codes.shape[0]), device=a.device, dtype=out_dtype)
    # A fixed tactic: no tuner lookup, no row-dependent specialization.
    _runner(out_dtype, pdl_enabled()).forward(
        [a, codes.t(), a_scales, scales, out_dtype, out, None],
        tactic=(_TILE, (1, 1), False, False, "sm120", None),
    )
    return out


def _quantize(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.quantization.triton import (
        _quantize_fp8_group32_swizzled,
    )

    if hidden.stride(-1) != 1:
        hidden = hidden.contiguous()
    return _quantize_fp8_group32_swizzled(hidden)


def prepare_mxfp8_lm_head(head: torch.Tensor) -> bool:
    """Quantize a BF16 ``[V, K]`` head once and compile what reads it.

    Call at load time, outside graph capture. The copy is keyed by the head's
    storage, so a draft model sharing the tensor finds it.

    Returns:
        Whether an MXFP8 copy now exists (False for unsupported geometry).
    """
    if (
        head.ndim != 2
        or head.dtype != torch.bfloat16
        or not head.is_cuda
        or not head.is_contiguous()
        or head.shape[0] % 128
        or head.shape[1] % 128
    ):
        return False
    key = _head_key(head)
    codes, scales = _quantize(head.detach())
    if codes.data_ptr() % 32:
        codes = codes.clone(memory_format=torch.contiguous_format)
    existing = _HEADS.get(key)
    if existing is not None and existing[0].shape == codes.shape:
        # Refresh in place after a weight update; captured graphs keep the buffers.
        existing[0].copy_(codes)
        existing[1].copy_(scales)
    else:
        _HEADS[key] = (codes, scales)
    if not torch.cuda.is_current_stream_capturing():
        # Compile the GEMM now: rows are runtime values.
        dummy = torch.zeros((2, head.shape[1]), dtype=torch.bfloat16, device=head.device)
        mxfp8_lm_head_logits(dummy, _HEADS[key], torch.float16)
    return True


def mxfp8_lm_head_logits(
    hidden: torch.Tensor,
    prepared: tuple[torch.Tensor, torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``[M, V]`` logits of BF16 ``hidden`` against the MXFP8 head.

    Hidden states are rounded to E4M3 in groups of 32 (UE8M0 scales), the
    head is its MXFP8 copy; FP32 accumulation, one rounding to ``out_dtype``
    (BF16 or FP16). Two launches.
    """
    codes, scales = prepared
    a, a_scales = _quantize(hidden)
    return _gemm(a, a_scales, codes, scales, out_dtype)
