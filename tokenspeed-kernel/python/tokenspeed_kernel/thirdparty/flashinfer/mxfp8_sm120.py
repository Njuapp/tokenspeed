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

"""Use the SM120 b12x split-output kernel for small MXFP8 projections."""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr
from flashinfer.cute_dsl.utils import get_max_active_clusters
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel,
)
from tokenspeed_kernel._triton import tl, triton

_SPLIT_K = 8


class _SplitKGemm:
    def __init__(self, max_active_clusters: int, *, enable_pdl: bool):
        self.max_active_clusters = max_active_clusters
        self.kernel = Sm120B12xBlockScaledDenseGemmKernel(
            32,
            (16, 128),
            (1, 1),
            mma_k=32,
            tile_k=128,
            split_k_slices=_SPLIT_K,
            split_k_atomic_bf16=False,
            enable_pdl=enable_pdl,
        )

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        sa: cute.Pointer,
        sb: cute.Pointer,
        alpha: cute.Tensor,
        stream,
    ):
        m, k = A.shape
        n = B.shape[0]
        a = cute.make_tensor(
            A.iterator, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            B.iterator, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        # Each K slice writes its own FP32 plane. The stock wrapper exposes
        # one plane, so it cannot represent this kernel's split output.
        c = cute.make_tensor(
            C.iterator, cute.make_ordered_layout((m, n, _SPLIT_K), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sa,
            cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        sfb = cute.make_tensor(
            sb,
            cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        self.kernel(a, b, sfa, sfb, c, alpha, self.max_active_clusters, stream)


@functools.cache
def _compiled_split_k(device: torch.device, enable_pdl: bool):
    # All tensor dimensions are runtime values. Neither rows nor request
    # counts enter the compilation cache key.
    with torch.cuda.device(device):
        m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
        a = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (m, k), stride_order=(1, 0), assumed_align=32
        )
        b = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (n, k), stride_order=(1, 0), assumed_align=32
        )
        c = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (_SPLIT_K, m, n),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        sa = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        sb = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        alpha = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            _SplitKGemm(get_max_active_clusters(1), enable_pdl=enable_pdl),
            a,
            b,
            c,
            cutlass.Int64(1),
            cutlass.Int64(1),
            cutlass.Int64(1),
            sa,
            sb,
            alpha,
            stream,
            options="--opt-level 2 --enable-tvm-ffi",
        )
        return compiled, torch.ones(1, device=device, dtype=torch.float32)


@triton.jit(do_not_specialize=["numel"])
def _reduce_split_k(PART, OUT, numel, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT).to(tl.int64)
    partial = tl.load(
        PART + split[:, None] * numel + index[None, :],
        index[None, :] < numel,
        0.0,
    )
    tl.store(OUT + index, tl.sum(partial, 0), index < numel)


def mm_mxfp8_split_k(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    out: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Compute a small SM120 MXFP8 GEMM with FP32 split-K reduction.

    Args:
        a: Contiguous E4M3 activations [M, 5120], aligned to 32 bytes.
            M must be in 1..16.
        b: Contiguous E4M3 weights [N, 5120], aligned to 32 bytes.
            N must be 576 or 1792.
        a_scales: Contiguous UE8M0 scales in F8_128x4 layout, aligned to
            16 bytes. Each scale covers 32 K elements.
        b_scales: Weight scales with the same format and alignment.
        out: Contiguous BF16 [M, N] output, or None to allocate it.
        enable_pdl: Programmatic dependency policy for the GEMM launch.

    Returns:
        BF16 [M, N] result. Each split accumulates in FP32. The final
        reduction also uses FP32 and rounds once to BF16.
    """
    m, k = a.shape
    n = b.shape[0]
    if not (1 <= m <= 16 and k == 5120 and n in (576, 1792)):
        raise ValueError(f"Unsupported SM120 split-K shape: M={m}, N={n}, K={k}")
    # This bounded domain also bounds upstream CuTe offsets: even the largest
    # weight has fewer than 2**24 bytes. K has 40 full tiles, eight equal slices.
    compiled, alpha = _compiled_split_k(a.device, enable_pdl)
    partial = torch.empty((_SPLIT_K, m, n), device=a.device, dtype=torch.float32)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    compiled(
        a,
        b,
        partial,
        (m + 127) // 128,
        (n + 127) // 128,
        k // 128,
        a_scales.data_ptr(),
        b_scales.data_ptr(),
        alpha,
    )
    _reduce_split_k[(triton.cdiv(m * n, 256),)](
        partial, out, m * n, _SPLIT_K, 256, num_warps=4
    )
    return out
