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

"""Pipelined SM120 prefill attention against the bf16x2 and bf16 kernels it replaces.

Self-contained: needs only the kernel package on PYTHONPATH and an SM120 GPU.
Both kernels are launched directly, so neither
``TOKENSPEED_SM120_PREFILL_ATTN_PV`` nor
``TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE`` need be set. The pipelined kernel
must reproduce the 16-head bf16x2 launch bit for bit. With the residual
product left out it must stay within rounding of the bf16 launch.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv4 import triton as implementation
from tokenspeed_kernel.ops.attention.dsv4._triton.compact_variants import (
    _compact_attention_bf16x2_pipelined,
    _compact_attention_variant,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform

_platform = current_platform()
_sm120 = _platform.is_nvidia and _platform.arch_version == ArchVersion(12, 0)
pytestmark = pytest.mark.skipif(not _sm120, reason="requires NVIDIA SM120")

_SCALE = 512**-0.5
_KV_ROWS = 3000


def _bf16x2(q, kv, indices, lens, sink):
    """The launch of triton_dsv4_prefill_variant_sm120 for PV=bf16x2, 16 heads."""
    out = torch.full_like(q, 3.0)
    _compact_attention_variant[(q.shape[0], -(-q.shape[1] // 16))](
        q,
        kv,
        indices,
        lens,
        sink,
        out,
        *q.stride(),
        *kv.stride(),
        *indices.stride(),
        lens.stride(0),
        0 if sink is None else sink.stride(0),
        *out.stride(),
        indices.shape[1],
        kv.shape[0],
        q.shape[1],
        _SCALE,
        sink is not None,
        32,
        16,
        "bf16x2",
        True,
        num_warps=4,
        num_stages=1,
    )
    return out


def _bf16(q, kv, indices, lens, sink):
    """The launch of triton_dsv4_prefill_variant_sm120 for PV=bf16, 16 heads."""
    out = torch.full_like(q, 3.0)
    _compact_attention_variant[(q.shape[0], -(-q.shape[1] // 16))](
        q,
        kv,
        indices,
        lens,
        sink,
        out,
        *q.stride(),
        *kv.stride(),
        *indices.stride(),
        lens.stride(0),
        0 if sink is None else sink.stride(0),
        *out.stride(),
        indices.shape[1],
        kv.shape[0],
        q.shape[1],
        _SCALE,
        sink is not None,
        32,
        16,
        "bf16",
        False,
        num_warps=8,
        num_stages=1,
    )
    return out


def _pipelined(q, kv, indices, lens, sink, *, residual):
    assert implementation._bf16x2_pipelined_serves(q, kv)
    out = torch.full_like(q, 5.0)
    implementation._launch_bf16x2_pipelined(
        q, kv, indices, lens, sink, _SCALE, out, residual=residual
    )
    return out


def _inputs(rows, heads, width, seed, q_scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(rows, heads, 512, generator=generator, device="cuda") * q_scale).to(
        torch.bfloat16
    )
    kv = torch.randn(_KV_ROWS, 512, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    indices = torch.randint(
        0, _KV_ROWS, (rows, width), generator=generator, device="cuda", dtype=torch.int32
    )
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    sink = torch.randn(heads, generator=generator, device="cuda")
    return q, kv, indices, lens, sink, generator


def _reference(q, kv, indices, lens, sink):
    """FP64 joint softmax, with the sink when given, over the valid entries."""
    result = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for token in range(q.shape[0]):
        row = indices[token, : int(lens[token])]
        row = row[(row >= 0) & (row < kv.shape[0])].long()
        if not row.numel():
            continue
        keys = kv[row].double()
        logits = q[token].double() @ keys.T * _SCALE
        if sink is not None:
            logits = torch.cat((logits, sink.double()[:, None]), dim=1)
            result[token] = torch.softmax(logits, dim=1)[:, :-1] @ keys
        else:
            result[token] = torch.softmax(logits, dim=1) @ keys
    return result


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("has_sink", [True, False])
@pytest.mark.parametrize(
    "rows,width",
    [(1, 640), (3, 1), (5, 31), (7, 32), (9, 33), (40, 128), (33, 640), (200, 96)],
)
def test_pipelined_without_residual_matches_bf16(heads, has_sink, rows, width):
    """PV=bf16 under the pipeline: probabilities rounded to BF16, one product.

    A row that fits one 32-row tile equals the bf16 launch bit for bit; longer
    rows may round a few values in 10^4 differently (measured at most 4e-4 of
    the values and a relative L2 gap of 9.3e-5).
    """
    q, kv, indices, lens, sink, _ = _inputs(rows, heads, width, seed=rows * 1000 + width)
    sink = sink if has_sink else None
    expected = _bf16(q, kv, indices, lens, sink)
    actual = _pipelined(q, kv, indices, lens, sink, residual=False)
    if width <= 32:
        assert torch.equal(actual, expected)
    else:
        assert (actual != expected).float().mean() < 1e-3
        gap = (actual.double() - expected.double()).square().sum().sqrt()
        assert gap < 2e-4 * expected.double().square().sum().sqrt()
    # The mode's own error (bf16x2 is at 1.7e-3 on these inputs).
    reference = _reference(q, kv, indices, lens, sink)
    error = (actual.double() - reference).square().sum().sqrt()
    assert error <= 2.3e-3 * reference.square().sum().sqrt()


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("has_sink", [True, False])
@pytest.mark.parametrize(
    "rows,width",
    [(1, 640), (3, 1), (5, 31), (7, 32), (9, 33), (40, 128), (33, 640), (200, 96)],
)
def test_pipelined_matches_bf16x2_bitwise(heads, has_sink, rows, width):
    q, kv, indices, lens, sink, _ = _inputs(rows, heads, width, seed=rows * 1000 + width)
    sink = sink if has_sink else None
    expected = _bf16x2(q, kv, indices, lens, sink)
    actual = _pipelined(q, kv, indices, lens, sink, residual=True)
    assert torch.equal(actual, expected)
    reference = _reference(q, kv, indices, lens, sink)
    error = (actual.double() - reference).square().sum().sqrt()
    assert error < 3e-3 * reference.square().sum().sqrt()


@pytest.mark.parametrize("has_sink", [True, False])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_pipelined_matches_bf16x2_on_invalid_entries(has_sink, index_dtype):
    q, kv, indices, lens, sink, generator = _inputs(96, 64, 640, seed=17)
    sink = sink if has_sink else None
    # Negative and out-of-range rows anywhere, and every length from an empty
    # row through tile boundaries to the full width.
    indices[torch.rand(indices.shape, generator=generator, device="cuda") < 0.05] = -1
    indices[torch.rand(indices.shape, generator=generator, device="cuda") < 0.01] = (
        _KV_ROWS + 5
    )
    lens = torch.randint(
        0, 641, (96,), generator=generator, device="cuda", dtype=torch.int32
    )
    lens[:10] = torch.tensor(
        [0, 1, 31, 32, 33, 63, 64, 639, 640, 900], dtype=torch.int32, device="cuda"
    )
    indices = indices.to(index_dtype)
    expected = _bf16x2(q, kv, indices, lens, sink)
    actual = _pipelined(q, kv, indices, lens, sink, residual=True)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual.float()).all()
    # A row without a valid entry and without a sink has no mass: zeros.
    if not has_sink:
        assert not actual[0].any()


@pytest.mark.parametrize("q_scale", [0.25, 4.0, 16.0])
def test_pipelined_matches_bf16x2_on_flat_and_peaked_rows(q_scale):
    q, kv, indices, lens, sink, _ = _inputs(64, 64, 640, seed=23, q_scale=q_scale)
    assert torch.equal(
        _pipelined(q, kv, indices, lens, sink, residual=True), _bf16x2(q, kv, indices, lens, sink)
    )


def test_pipelined_writes_strided_output_and_reads_strided_queries():
    q, kv, indices, lens, sink, _ = _inputs(24, 64, 640, seed=29)
    wide_q = torch.zeros(24, 2, 64, 512, dtype=torch.bfloat16, device="cuda")
    wide_q[:, 1] = q
    expected = _bf16x2(q, kv, indices, lens, sink)
    out = torch.full((24, 2, 64, 512), 9.0, dtype=torch.bfloat16, device="cuda")
    view = out[:, 1]
    implementation._launch_bf16x2_pipelined(
        wide_q[:, 1], kv, indices, lens, sink, _SCALE, view, residual=True
    )
    assert torch.equal(view, expected)
    assert bool((out[:, 0] == 9.0).all())


def test_pipelined_serves_only_dense_rows_and_whole_head_tiles():
    q, kv, _, _, _, _ = _inputs(2, 64, 4, seed=31)
    assert implementation._bf16x2_pipelined_serves(q, kv)
    assert not implementation._bf16x2_pipelined_serves(q[:, :16], kv)
    assert not implementation._bf16x2_pipelined_serves(q[:, :48], kv)
    assert not implementation._bf16x2_pipelined_serves(q, kv[:, ::2])
    assert not implementation._bf16x2_pipelined_serves(q, kv[::2])


def test_pipelined_row_counts_and_widths_are_runtime_values():
    def specializations():
        return sum(
            len(cache[0])
            for cache in _compact_attention_bf16x2_pipelined.device_caches.values()
        )

    q, kv, indices, lens, sink, _ = _inputs(4, 64, 640, seed=37)
    _pipelined(q, kv, indices, lens, sink, residual=True)
    before = specializations()
    for rows, width in ((1, 640), (17, 5), (300, 640), (64, 129), (2, 1)):
        q, kv, indices, lens, sink, _ = _inputs(rows, 64, width, seed=41)
        _pipelined(q, kv, indices, lens, sink, residual=True)
    assert specializations() == before
