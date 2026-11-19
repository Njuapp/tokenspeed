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

"""Opt-in FlashInfer paged decode attention on NVIDIA compute capability 12.0.

Self-contained: needs only the kernel package on PYTHONPATH. The native
tests run with ``TOKENSPEED_SM120_DSV41_ATTENTION_FLASHINFER=1`` and are
skipped otherwise; the registration test starts its own interpreters.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import flashinfer as implementation
from tokenspeed_kernel.ops.attention.dsv41 import triton as portable
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry

_NAME = "flashinfer_dsv41_selected_attention_sm120"
_TRITON = "triton_dsv41_selected_attention_sm120"
_SCALE = 512**-0.5
_STREAMS, _SWA_PAGES, _GLOBAL_PAGES = 4, 4, 16
_platform = current_platform()
_sm120 = _platform.is_nvidia and _platform.arch_version == ArchVersion(12, 0)
pytestmark = pytest.mark.skipif(not _sm120, reason="requires NVIDIA SM120")
native = pytest.mark.skipif(
    KernelRegistry.get().get_by_name(_NAME) is None,
    reason=f"set {implementation.SM120_ATTENTION_ENV}=1 to register the kernel",
)


def _selected_kernel(gate: str | None) -> str:
    env = {
        k: v for k, v in os.environ.items() if k != implementation.SM120_ATTENTION_ENV
    }
    if gate is not None:
        env[implementation.SM120_ATTENTION_ENV] = gate
    script = (
        "import torch\n"
        "from tokenspeed_kernel.ops.attention import dsv41\n"
        "from tokenspeed_kernel.selection import select_kernel\n"
        "from tokenspeed_kernel.signature import dense_tensor_format, format_signature\n"
        "sig = format_signature(x=dense_tensor_format(torch.bfloat16))\n"
        "print(select_kernel('attention', 'dsv41_selected_attention', sig, "
        "traits={'flashmla_eligible': True}).name)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("gate", [None, "0", "true"])
def test_switch_off_keeps_the_triton_reader(gate):
    assert _selected_kernel(gate) == _TRITON


def test_switch_on_selects_the_flashinfer_reader():
    assert _selected_kernel("1") == _NAME


@pytest.fixture(scope="module")
def fields():
    """Page-planar SWA and global fields written by the cache writer, as strided views."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for pages, row_bytes, stride, layout in (
        (_STREAMS * _SWA_PAGES, 528, 64 * 528 + 4096, "swa"),
        (_STREAMS * _GLOBAL_PAGES, 288, 23040, "global"),
    ):
        buffer = torch.zeros(pages * stride, dtype=torch.uint8, device="cuda")
        field = buffer.as_strided((pages, 64, row_bytes), (stride, row_bytes, 1))
        rows = torch.randn(pages * 64, 512, device="cuda", generator=generator)
        slots = torch.arange(pages * 64, device="cuda", dtype=torch.int32)
        portable.cache_scatter(rows.to(torch.bfloat16), field, slots, layout)
        out.append(field)
    return out


def _case(name: str, tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    stream = torch.arange(tokens, device="cuda") % _STREAMS
    first = torch.randint(
        0, _SWA_PAGES * 64 - 128, (tokens,), device="cuda", generator=generator
    )
    columns = torch.arange(128, device="cuda")
    swa = (stream * _SWA_PAGES * 64 + first)[:, None] + columns[None, :]
    order = torch.rand(tokens, _GLOBAL_PAGES * 64, device="cuda", generator=generator)
    glob = (
        order.topk(512, dim=1).indices.sort(dim=1).values
        + stream[:, None] * _GLOBAL_PAGES * 64
    )
    swa, glob = swa.to(torch.int32), glob.to(torch.int32)
    swa_len = torch.full((tokens,), 128, device="cuda", dtype=torch.int32)
    glob_len = torch.full((tokens,), 512, device="cuda", dtype=torch.int32)
    if name == "short":
        # History shorter than the window: a valid prefix, then -1 padding.
        swa_len = torch.randint(
            0, 129, (tokens,), device="cuda", generator=generator
        ).int()
        glob_len = torch.randint(
            0, 513, (tokens,), device="cuda", generator=generator
        ).int()
        swa_len[0], glob_len[0] = 0, 0
        swa = swa.masked_fill(columns[None, :] >= swa_len[:, None], -1)
        glob = glob.masked_fill(
            torch.arange(512, device="cuda")[None, :] >= glob_len[:, None], -1
        )
    elif name == "holes":
        swa = swa.masked_fill(
            torch.rand(swa.shape, device="cuda", generator=generator) < 0.3, -1
        )
        glob = glob.masked_fill(
            torch.rand(glob.shape, device="cuda", generator=generator) < 0.3, -1
        )
        swa_len, glob_len = swa_len.fill_(100), glob_len.fill_(300)
    elif name == "duplicates":
        swa, glob = swa[:, columns % 7], glob[:, torch.arange(512, device="cuda") % 33]
    elif name == "no_global_rows":
        glob_len = glob_len.zero_()
    sink = torch.randn(64, device="cuda", generator=generator) * 0.5
    q = torch.randn(tokens, 64, 512, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    return q, swa, swa_len, glob, glob_len, sink


def _reference(q, swa_field, swa, swa_len, global_field, glob, glob_len, sink):
    """FP64 joint softmax with one zero-valued sink over the selected rows."""
    rows, masks = [], []
    for field, slots, lengths, layout in (
        (swa_field, swa, swa_len, "swa"),
        (global_field, glob, glob_len, "global"),
    ):
        if field is None:
            continue
        valid = (
            torch.arange(slots.shape[1], device="cuda")[None, :] < lengths[:, None]
        ) & (slots >= 0)
        rows.append(
            portable.cache_gather(
                field, slots.masked_fill(~valid, -1), layout, None
            ).double()
        )
        masks.append(valid)
    kv, valid = torch.cat(rows, dim=1), torch.cat(masks, dim=1)
    logits = torch.einsum("thd,twd->thw", q.double(), kv) * _SCALE
    logits = logits.masked_fill(~valid[:, None, :], -math.inf)
    top = torch.maximum(logits.max(dim=-1).values, sink.double()[None, :])
    weights = torch.exp(logits - top[..., None])
    denominator = weights.sum(-1) + torch.exp(sink.double()[None, :] - top)
    return torch.einsum("thw,twd->thd", weights, kv) / denominator[..., None]


def _op(q, swa_field, swa, swa_len, global_field, glob, glob_len, sink, schedule):
    return dsv41.selected_attention(
        q, swa_field, swa, swa_len, global_field, glob, glob_len, sink,
        _SCALE, None, 256, schedule, None, None, lse=None,
    )  # fmt: skip


def _triton(q, swa_field, swa, swa_len, global_field, glob, glob_len, sink):
    return portable.triton_dsv41_selected_attention_sm120(
        q, swa_field, swa, swa_len, global_field, glob, glob_len, sink,
        _SCALE, None, 256, None, None, None,
    )  # fmt: skip


@native
@pytest.mark.parametrize("with_global", [True, False])
@pytest.mark.parametrize("tokens", [1, 6, 66])
@pytest.mark.parametrize(
    "case", ["full", "short", "holes", "duplicates", "no_global_rows"]
)
def test_native_decode_matches_reference_and_triton(fields, with_global, tokens, case):
    swa_field, global_field = fields
    q, swa, swa_len, glob, glob_len, sink = _case(case, tokens, 17 * tokens + len(case))
    selection = (global_field, glob, glob_len) if with_global else (None, None, None)
    reported = set(implementation._reported_fallbacks)
    out = _op(
        q, swa_field, swa, swa_len, *selection, sink, dsv41.new_attention_schedule()
    )
    # Served natively: the device and head count are warm and nothing fell back.
    assert (q.get_device(), 64) in implementation._warmed_decode
    assert implementation._reported_fallbacks == reported
    expected = _reference(q, swa_field, swa, swa_len, *selection, sink)
    # Output and probabilities round to BF16: 2**-8 relative per element.
    scale = expected.norm(dim=-1, keepdim=True)
    live = scale.squeeze(-1) > 0
    error = (out.double() - expected).norm(dim=-1)
    assert bool((error[live] <= 1e-2 * scale.squeeze(-1)[live]).all())
    assert bool((out[~live] == 0).all())
    triton = _triton(q, swa_field, swa, swa_len, *selection, sink).double()
    assert float((out.double() - triton).norm()) <= 6e-3 * float(triton.norm()) + 1e-12


@native
@pytest.mark.parametrize("with_global", [True, False])
@pytest.mark.parametrize("tokens", [6, 66])
def test_graph_replay_reads_live_slots_lengths_and_rows(fields, with_global, tokens):
    swa_field, global_field = fields
    buffers = [t.clone() for t in _case("full", tokens, 5)]
    q, swa, swa_len, glob, glob_len, sink = buffers
    selection = (global_field, glob, glob_len) if with_global else (None, None, None)
    schedule = dsv41.new_attention_schedule()
    _op(q, swa_field, swa, swa_len, *selection, sink, schedule)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _op(q, swa_field, swa, swa_len, *selection, sink, schedule)
    for step, case in enumerate(("short", "holes", "duplicates")):
        for buffer, value in zip(buffers, _case(case, tokens, 100 + step), strict=True):
            buffer.copy_(value)
        graph.replay()
        eager = _op(q, swa_field, swa, swa_len, *selection, sink, schedule)
        torch.testing.assert_close(out, eager, rtol=0, atol=0)


@native
def test_calls_the_family_does_not_serve_run_the_triton_reader(fields):
    swa_field, global_field = fields
    q, swa, swa_len, glob, glob_len, sink = _case("short", 6, 3)
    paged = (swa_field, swa, swa_len, global_field, glob, glob_len)
    schedule = dsv41.new_attention_schedule()
    expected = _triton(q, *paged, sink)
    # No schedule: slots are not promised to be runtime-produced.
    torch.testing.assert_close(_op(q, *paged, sink, None), expected, rtol=0, atol=0)
    # A head count outside the generated family.
    q24, sink24 = q[:, :24].contiguous(), sink[:24].contiguous()
    torch.testing.assert_close(
        _op(q24, *paged, sink24, schedule), _triton(q24, *paged, sink24), rtol=0, atol=0
    )
    # A page stride that is not a multiple of 16 bytes.
    odd = torch.zeros(4 * (64 * 528 + 8), dtype=torch.uint8, device="cuda")
    odd = odd.as_strided((4, 64, 528), (64 * 528 + 8, 528, 1))
    rows = torch.randn(4 * 64, 512, device="cuda", dtype=torch.bfloat16)
    portable.cache_scatter(
        rows, odd, torch.arange(4 * 64, device="cuda", dtype=torch.int32), "swa"
    )
    local = (swa % (4 * 64)).masked_fill(swa < 0, -1)
    torch.testing.assert_close(
        _op(q, odd, local, swa_len, None, None, None, sink, schedule),
        _triton(q, odd, local, swa_len, None, None, None, sink),
        rtol=0,
        atol=0,
    )
    reported = " ".join(implementation._reported_fallbacks)
    assert "no decode schedule" in reported
    assert "24 query heads" in reported
    assert "SWA field" in reported
