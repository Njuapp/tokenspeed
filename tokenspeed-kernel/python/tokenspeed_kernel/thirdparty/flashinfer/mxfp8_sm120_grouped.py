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

"""Grouped MXFP8 projection on SM120: one batched b12x launch for all groups.

``out[t, g, :] = dequant(x[t, g, :]) @ dequant(weight[g]).T`` with the
activations quantized to E4M3 in groups of 32 with UE8M0 scales (the reference
dense quantizer) and the weight held as E4M3 codes with 1x32 UE8M0 scales.
The b12x kernel takes the group count as its compile-time batch size; rows are
a runtime value. Experiment code behind ``TOKENSPEED_SM120_WO_A_MXFP8``.
"""

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
from tokenspeed_kernel.platform import pdl_enabled

_fp8_dtype = torch.float8_e4m3fn
# One quantizer specialization for every row count.
_QUANT_GROUPS = 32
_ZERO_BLOCK = 256
# b12x (M, N) tile by row count, measured on RTX 6000D with cold weights
# (8 groups, 4096 -> 1024): one 64-row tile up to 64 rows, one 128-row tile
# up to 128 rows, 64 x 128 tiles above. Three compiled variants in all.
_TILES = ((64, (64, 64)), (128, (128, 64)), (None, (64, 128)))


def _tile_for_rows(rows: int) -> tuple[int, int]:
    for limit, tile in _TILES:
        if limit is None or rows <= limit:
            return tile
    raise AssertionError("unreachable")


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """F8_128x4 layout of canonical ``[rows, K / 32]`` UE8M0 scales."""
    rows, columns = scales.shape
    row_tiles = (rows + 127) // 128
    k_tiles = (columns + 3) // 4
    padded = torch.zeros(
        (row_tiles * 128, k_tiles * 4), dtype=scales.dtype, device=scales.device
    )
    padded[:rows, :columns] = scales
    return (
        padded.view(row_tiles, 4, 32, k_tiles, 4).transpose(1, 3).contiguous().view(-1)
    )


def prepare_grouped_mxfp8_weight(
    codes: torch.Tensor, scales: torch.Tensor, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay a grouped FP8 weight out for :func:`grouped_mxfp8_projection`.

    Args:
        codes: E4M3 ``[groups * N, K]`` weight codes; group ``g`` owns rows
            ``[g * N, (g + 1) * N)``.
        scales: uint8 UE8M0 ``[groups * N, K / 32]`` scales of those rows.
        groups: Number of groups.

    Returns:
        ``[groups, N, K]`` contiguous codes aligned to 32 bytes, and the 1-D
        scales: one F8_128x4 block per group, concatenated.
    """
    if codes.dtype != _fp8_dtype or scales.dtype != torch.uint8:
        raise ValueError("Grouped MXFP8 weight needs E4M3 codes and UE8M0 scales")
    rows, k = codes.shape
    if rows % groups or k % 128 or (rows // groups) % 128:
        raise ValueError("Grouped MXFP8 weight needs N and K multiples of 128")
    if tuple(scales.shape) != (rows, k // 32):
        raise ValueError("Grouped MXFP8 scales must be [groups * N, K / 32]")
    n = rows // groups
    grouped = codes.reshape(groups, n, k).contiguous()
    if grouped.data_ptr() % 32:
        grouped = grouped.clone(memory_format=torch.contiguous_format)
    packed = torch.cat(
        [swizzle_scales(scales[g * n : (g + 1) * n]) for g in range(groups)]
    ).contiguous()
    return grouped, packed


@triton.jit
def _swizzled_scale_offset(row, column, K_TILES: tl.constexpr):
    # Equivalent to [Mtiles,4,32,Ktiles,4].transpose(1,3).contiguous().
    return (
        (row // 128 * K_TILES + column // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + column % 4
    )


@triton.jit(do_not_specialize=["M", "X0"])
def _fp8_group32_ue8m0_quantize_grouped(
    X,
    Q,
    S,
    M,
    X0,
    K: tl.constexpr,
    G: tl.constexpr,
    GROUPS: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Group-32 UE8M0 quantization of ``[M, G, K]``; scales in one F8_128x4
    block per group. Same arithmetic as the dense reference quantizer."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0).to(tl.int64)
    k_tiles = K // 128
    group_columns = K // 32
    row_columns = G * group_columns
    m_tiles = tl.cdiv(M, 128)
    block = m_tiles.to(tl.int64) * (128 * k_tiles * 4)
    if pid * GROUPS < M * row_columns:
        index = pid * GROUPS + tl.arange(0, GROUPS)
        row = index // row_columns
        flat_column = index % row_columns
        group = flat_column // group_columns
        scale_column = flat_column % group_columns
        column = flat_column[:, None] * 32 + tl.arange(0, 32)[None, :]
        values = tl.load(X + row[:, None] * X0 + column, row[:, None] < M, 0.0).to(
            tl.float32
        )
        # Preserve the reference quantizer's exact arithmetic and rounding boundary.
        amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
        raw = amax * (1.0 / 448.0)
        bits = raw.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        quantized = tl.div_rn(values, scale[:, None]).to(tl.float8e4nv)
        tl.store(Q + row[:, None] * (G * K) + column, quantized, row[:, None] < M)
        scale_offset = group * block + _swizzled_scale_offset(
            row, scale_column, k_tiles
        )
        tl.store(S + scale_offset, exponent.to(tl.uint8), row < M)

    # Rows past M inside the last 128-row block. K is a multiple of 128, so
    # there is no column padding. Each program owns a fixed slice.
    if M % 128 != 0:
        padded_columns = k_tiles * 4
        index = pid * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        group = index // block
        within = index % block
        pad_row = within // padded_columns
        pad_column = within % padded_columns
        padding = (index < G * block) & (pad_row >= M)
        offset = group * block + _swizzled_scale_offset(pad_row, pad_column, k_tiles)
        tl.store(S + offset, 0, padding)


def quantize_fp8_group32_grouped(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``[M, G, K]`` for the batched kernel.

    Args:
        x: BF16 activations with ``x.stride(1) == K`` and unit inner stride;
            the row stride is free (a slice of padded rows is valid).

    Returns:
        E4M3 codes ``[M, G, K]`` (contiguous) and uint8 scales: ``G``
        consecutive F8_128x4 blocks of ``ceil(M / 128) * 128 * K / 32`` bytes.
    """
    m, groups, k = x.shape
    quantized = torch.empty((m, groups, k), dtype=_fp8_dtype, device=x.device)
    block = triton.cdiv(m, 128) * 128 * (k // 32)
    scales = torch.empty((groups * block,), dtype=torch.uint8, device=x.device)
    if m:
        programs = max(
            triton.cdiv(m * groups * (k // 32), _QUANT_GROUPS),
            triton.cdiv(groups * block, _ZERO_BLOCK),
        )
        enable_pdl = pdl_enabled()
        _fp8_group32_ue8m0_quantize_grouped[(programs,)](
            x,
            quantized,
            scales,
            m,
            x.stride(0),
            k,
            groups,
            _QUANT_GROUPS,
            _ZERO_BLOCK,
            enable_pdl,
            num_warps=4,
            num_stages=1,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return quantized, scales


class _GroupedGemm:
    def __init__(
        self, tile: tuple[int, int], groups: int, max_active_clusters: int, *, enable_pdl
    ):
        self.groups = groups
        self.max_active_clusters = max_active_clusters
        self.kernel = Sm120B12xBlockScaledDenseGemmKernel(
            32,
            tile,
            (1, 1),
            mma_k=32,
            tile_k=128,
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
        m = A.shape[0]
        k = A.shape[2]
        n = B.shape[1]
        groups = self.groups
        # Operands and result stay token-major [M, G, *] (K or N fastest, then
        # the group, then the row); the weight is group-major [G, N, K].
        a = cute.make_tensor(
            A.iterator, cute.make_ordered_layout((m, k, groups), order=(2, 0, 1))
        )
        b = cute.make_tensor(
            B.iterator, cute.make_ordered_layout((n, k, groups), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            C.iterator, cute.make_ordered_layout((m, n, groups), order=(2, 0, 1))
        )
        sfa = cute.make_tensor(
            sa,
            cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, groups), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        sfb = cute.make_tensor(
            sb,
            cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, groups), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        self.kernel(a, b, sfa, sfb, c, alpha, self.max_active_clusters, stream)


@functools.cache
def _compiled_grouped(
    device: torch.device, tile: tuple[int, int], groups: int, enable_pdl: bool
):
    # Rows, N and K are runtime values; the tile and the group count (a model
    # dimension) are the only compile-time parameters.
    with torch.cuda.device(device):
        m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
        a = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (m, groups, k), stride_order=(2, 1, 0), assumed_align=32
        )
        b = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (groups, n, k), stride_order=(2, 1, 0), assumed_align=32
        )
        c = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, groups, n), stride_order=(2, 1, 0), assumed_align=16
        )
        sa = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        sb = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        alpha = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            _GroupedGemm(
                tile, groups, get_max_active_clusters(1), enable_pdl=enable_pdl
            ),
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


def warm_grouped_mxfp8(device: torch.device, groups: int) -> None:
    """Compile every tile variant the projection can launch on ``device``."""
    enable_pdl = pdl_enabled()
    for _, tile in _TILES:
        _compiled_grouped(device, tile, groups, enable_pdl)


def grouped_mxfp8_gemm(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    tile: tuple[int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One batched b12x launch: ``out[m, g] = a[m, g] @ weight[g].T``.

    Args:
        a: Contiguous E4M3 ``[M, G, K]`` codes aligned to 32 bytes, M >= 1.
        a_scales: Scales from :func:`quantize_fp8_group32_grouped`.
        weight: Codes from :func:`prepare_grouped_mxfp8_weight`.
        weight_scales: Scales from :func:`prepare_grouped_mxfp8_weight`.
        tile: b12x ``(M, N)`` tile; None picks it from the row count.
        out: Optional contiguous BF16 ``[M, G, N]`` result.

    Returns:
        BF16 ``[M, G, N]``; FP32 accumulation, one rounding.
    """
    m, groups, k = a.shape
    n = weight.shape[1]
    if tile is None:
        tile = _tile_for_rows(m)
    compiled, alpha = _compiled_grouped(a.device, tile, groups, pdl_enabled())
    if out is None:
        out = torch.empty((m, groups, n), device=a.device, dtype=torch.bfloat16)
    compiled(
        a,
        weight,
        out,
        (m + 127) // 128,
        (n + 127) // 128,
        k // 128,
        a_scales.data_ptr(),
        weight_scales.data_ptr(),
        alpha,
    )
    return out


def grouped_mxfp8_projection(
    x: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor
) -> torch.Tensor:
    """Grouped projection of BF16 activations by a grouped MXFP8 weight.

    Args:
        x: BF16 ``[M, G, K]`` activations, M >= 1, K a multiple of 128.
        weight: ``[G, N, K]`` codes from :func:`prepare_grouped_mxfp8_weight`.
        weight_scales: Its scales.

    Returns:
        BF16 ``[M, G, N]``. Two launches: the group-32 quantizer and one
        batched GEMM. Activations are rounded to E4M3 with UE8M0 group scales,
        which the BF16 grouped projection does not do.
    """
    if x.ndim != 3 or x.dtype != torch.bfloat16:
        raise ValueError("Grouped MXFP8 projection needs BF16 [M, G, K] activations")
    if x.shape[1:] != (weight.shape[0], weight.shape[2]):
        raise ValueError("Grouped MXFP8 operands must share groups and input_dim")
    if x.stride(2) != 1 or x.stride(1) != x.shape[2]:
        x = x.contiguous()
    codes, scales = quantize_fp8_group32_grouped(x)
    return grouped_mxfp8_gemm(codes, scales, weight, weight_scales)
