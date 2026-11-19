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

"""Opt-in grouped index scoring on NVIDIA compute capability 12.0.

Self-contained: needs only the kernel package on PYTHONPATH. The switch
``TOKENSPEED_SM120_DSV41_INDEX_GROUPED`` is read once at import, so the
equality tests flip the module constant to run both paths in one process;
the switch test starts its own interpreters.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv41 import flashinfer as implementation
from tokenspeed_kernel.ops.attention.dsv41 import triton as portable
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry

_NAME = "flashinfer_dsv41_index_topk"
_platform = current_platform()
_sm120 = _platform.is_nvidia and _platform.arch_version == ArchVersion(12, 0)
pytestmark = pytest.mark.skipif(
    not _sm120 or KernelRegistry.get().get_by_name(_NAME) is None,
    reason="requires NVIDIA SM120 with FlashInfer's graph-safe TopK",
)
_PAGES = 96
_WIDTH = _PAGES * 64


def _switch(value: str | None) -> str:
    name = implementation.SM120_INDEX_GROUPED_ENV
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    script = (
        "from tokenspeed_kernel.ops.attention.dsv41 import flashinfer\n"
        "print(flashinfer._INDEX_GROUPED)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_switch_is_off_unless_one(value):
    assert _switch(value) == "False"


def test_switch_on():
    assert _switch("1") == "True"


@pytest.fixture(scope="module")
def cache():
    """A strided index field written by the cache writer: 12 streams of 96 pages."""
    device = torch.device("cuda")
    pages, stride = 12 * _PAGES + 1, 4608
    generator = torch.Generator(device=device).manual_seed(0)
    arena = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
    field = arena.as_strided((pages, 64, 68), (stride, 68, 1))
    rows = torch.randn(pages * 64, 128, device=device, generator=generator)
    slots = torch.arange(pages * 64, device=device, dtype=torch.int32)
    portable.cache_scatter(rows.to(torch.bfloat16), field, slots, "index")
    return field


def _queries(count, dtype, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(count, 32, 128, device=device, generator=generator)
    w = torch.rand(count, 32, device=device, generator=generator) - 0.3
    return q.to(torch.bfloat16), (w / 64).to(dtype)


def _paged(streams, rows_per_stream):
    """One table row per query row, as the runtime's selection table builds them."""
    device = torch.device("cuda")
    count = streams * rows_per_stream
    stream = torch.arange(count, device=device) // rows_per_stream
    table = 1 + stream[:, None] * _PAGES + torch.arange(_PAGES, device=device)[None, :]
    visible = _WIDTH - 40 + torch.arange(count, device=device) % rows_per_stream
    return table.to(torch.int32).contiguous(), visible.to(torch.int32)


def _broadcast(count):
    device = torch.device("cuda")
    row = 1 + torch.randperm(12 * _PAGES, device=device)[:_PAGES]
    visible = _WIDTH - torch.arange(count, device=device) // 2
    return row.to(torch.int32)[None, :].expand(count, -1), visible.to(torch.int32)


def _run(monkeypatch, grouped, cache, q, w, table, visible, candidate_topk, out=None):
    """Selections plus the score matrices handed to the selector."""
    monkeypatch.setattr(implementation, "_INDEX_GROUPED", grouped)
    seen = []
    select = implementation.topk_small_indices

    def recording(scores, k):
        seen.append(scores.clone())
        return select(scores, k)

    monkeypatch.setattr(implementation, "topk_small_indices", recording)
    result = implementation.flashinfer_dsv41_index_topk(
        q,
        w,
        cache,
        table,
        visible,
        None,
        512,
        candidate_topk,
        8,
        1024,
        256,
        None,
        out,
        None,
    )
    monkeypatch.setattr(implementation, "topk_small_indices", select)
    return [tensor.clone() for tensor in result], seen


def _layout(name):
    if name == "broadcast":
        return _broadcast(41)
    if name == "paged-6":
        return _paged(11, 6)
    if name == "paged-5":
        return _paged(9, 5)
    if name == "paged-1":
        return _paged(12, 1)
    table, visible = _paged(11, 6)
    table, visible = table.clone(), visible.clone()
    table[7] = -1  # a padding row
    visible[7] = 0
    table[20, 30:] = -1  # a table that ends early
    table[33, 5] = 10**6  # a page outside the field
    visible[40] = 300  # fewer rows than the selection holds
    return table, visible


@pytest.mark.parametrize(
    "layout", ["broadcast", "paged-6", "paged-5", "paged-1", "holes"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("candidate_topk", [0, 2048])
def test_grouped_scores_and_selections_are_identical(
    monkeypatch, cache, layout, dtype, candidate_topk
):
    table, visible = _layout(layout)
    q, w = _queries(table.shape[0], dtype, seed=3)
    want, want_scores = _run(
        monkeypatch, False, cache, q, w, table, visible, candidate_topk
    )
    got, got_scores = _run(
        monkeypatch, True, cache, q, w, table, visible, candidate_topk
    )
    for expected, actual in zip(want, got, strict=True):
        assert torch.equal(expected, actual)
    assert len(want_scores) == len(got_scores) == (2 if candidate_topk else 1)
    for expected, actual in zip(want_scores, got_scores, strict=True):
        assert expected.dtype == torch.float32
        # 16-bit rows exactly when the weights, and so the scores, are BF16.
        assert actual.dtype == dtype
        assert torch.equal(expected.view(torch.int32), actual.float().view(torch.int32))


def test_grouped_replay_reads_live_tables_and_queries(monkeypatch, cache):
    table, visible = _paged(11, 6)
    q, w = _queries(66, torch.bfloat16, seed=5)
    shapes = ((66, 512), (66,), (66, 2048), (66,))
    out = tuple(torch.empty(s, dtype=torch.int32, device=q.device) for s in shapes)
    _run(monkeypatch, True, cache, q, w, table, visible, 2048, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        implementation.flashinfer_dsv41_index_topk(
            q, w, cache, table, visible, None, 512, 2048, 8, 1024, 256, None, out, None
        )
    fresh_q, fresh_w = _queries(66, torch.bfloat16, seed=6)
    q.copy_(fresh_q)
    w.copy_(fresh_w)
    visible.sub_(500)
    # Every third pair stops sharing its pages; one stream leaves the batch.
    table[1::6] = table[1::6].roll(1, dims=1)
    table[12:18] = -1
    visible[12:18] = 0
    graph.replay()
    torch.cuda.synchronize()
    got = [tensor.clone() for tensor in out]
    want, _ = _run(monkeypatch, False, cache, q, w, table, visible, 2048)
    for expected, actual in zip(want, got, strict=True):
        assert torch.equal(expected, actual)
