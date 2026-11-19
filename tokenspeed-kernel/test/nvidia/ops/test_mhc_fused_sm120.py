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

"""Fused SM120 mHC boundary against the kernels it replaces.

Self-contained: needs only the kernel package on PYTHONPATH and an SM120 GPU.
The fused entry points are called directly, so ``TOKENSPEED_SM120_MHC_FUSED``
need not be set.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from tokenspeed_kernel.ops.residual import mhc_apply_pre, mhc_mixes, mhc_post
from tokenspeed_kernel.ops.residual import triton as implementation
from tokenspeed_kernel.platform import ArchVersion, current_platform

_platform = current_platform()
_sm120 = _platform.is_nvidia and _platform.arch_version == ArchVersion(12, 0)
pytestmark = pytest.mark.skipif(not _sm120, reason="requires NVIDIA SM120")

_HIDDEN = 5120
_RMS_EPS = 1e-20
_HC_EPS = 1e-6
_SINKHORN = 20


def _inputs(tokens: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(*shape):
        return torch.randn(shape, generator=generator, device="cuda")

    residual = normal(tokens, 4, _HIDDEN).to(torch.bfloat16)
    scale = torch.rand(3, generator=generator, device="cuda") + 0.5
    base = normal(24) * 0.5
    # Coefficients of the sublayer in front, as the model holds them.
    pre, post, comb = mhc_mixes(
        residual,
        normal(24, 4 * _HIDDEN) * 0.01,
        scale,
        base,
        _RMS_EPS,
        _HC_EPS,
        _SINKHORN,
    )
    return {
        "x": normal(tokens, _HIDDEN).to(torch.bfloat16),
        "residual": residual,
        "pre": pre,
        "post": post,
        "comb": comb,
        "weight": normal(24, 4 * _HIDDEN) * 0.01,
        "scale": scale,
        "base": base,
        "norm_weight": (1.0 + 0.1 * normal(_HIDDEN)).to(torch.bfloat16),
    }


def _separate(d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    residual = mhc_post(d["x"], d["residual"], d["post"].unsqueeze(-1), d["comb"])
    layer_input = mhc_apply_pre(
        residual, d["pre"], norm_weight=d["norm_weight"], norm_eps=_RMS_EPS
    )
    return residual, layer_input


def _fused(d: dict[str, torch.Tensor]):
    return implementation.mhc_post_input_norm_hc4(
        d["x"],
        d["residual"],
        d["pre"],
        d["post"],
        d["comb"],
        d["norm_weight"],
        _RMS_EPS,
    )


@pytest.mark.parametrize("tokens", [1, 5, 17, 55, 66, 300, 2048, 2500])
def test_post_input_norm_matches_separate_kernels_bitwise(tokens):
    d = _inputs(tokens, tokens)
    residual, layer_input = _separate(d)
    fused = _fused(d)
    assert fused is not None
    assert torch.equal(fused[0], residual)
    assert torch.equal(fused[1], layer_input)


def test_other_layouts_decline():
    d = _inputs(8, 1)
    assert _fused(dict(d, x=d["x"].float())) is None
    assert _fused(dict(d, residual=d["residual"][:, :, :4096].contiguous())) is None
    assert _fused(dict(d, pre=d["pre"][::2])) is None


def test_cuda_graph_replay_with_changed_inputs():
    d = _inputs(66, 7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _fused(d)
    fresh = _inputs(66, 8)
    for name, tensor in fresh.items():
        d[name].copy_(tensor)
    graph.replay()
    residual, layer_input = _separate(d)
    assert torch.equal(captured[0], residual)
    assert torch.equal(captured[1], layer_input)


def test_row_counts_share_one_binary():
    kernel = implementation._mhc_post_input_norm_hc4_kernel
    _fused(_inputs(66, 11))
    before = sum(len(cache[0]) for cache in kernel.device_caches.values())
    for tokens in (1, 17, 55, 97, 513, 2036, 2048):
        _fused(_inputs(tokens, 100 + tokens))
    assert sum(len(cache[0]) for cache in kernel.device_caches.values()) == before == 1


def _mixes(d: dict[str, torch.Tensor], min_splits: int):
    return implementation._mhc_mixes_tensor_sm120_min_splits(
        d["residual"],
        d["weight"],
        d["scale"],
        d["base"],
        _RMS_EPS,
        _HC_EPS,
        _SINKHORN,
        min_splits,
    )


@pytest.mark.parametrize("tokens", [17, 55, 66, 128])
def test_min_64_splits_is_the_unfused_count_up_to_128_rows(tokens):
    d = _inputs(tokens, 2000 + tokens)
    expected = mhc_mixes(
        d["residual"], d["weight"], d["scale"], d["base"], _RMS_EPS, _HC_EPS, _SINKHORN
    )
    for a, b in zip(_mixes(d, 64), expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("tokens", [129, 300, 2048, 2500])
def test_min_64_splits_keeps_the_coefficients_close(tokens):
    d = _inputs(tokens, 3000 + tokens)
    expected = mhc_mixes(
        d["residual"], d["weight"], d["scale"], d["base"], _RMS_EPS, _HC_EPS, _SINKHORN
    )
    for a, b in zip(_mixes(d, 64), expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
