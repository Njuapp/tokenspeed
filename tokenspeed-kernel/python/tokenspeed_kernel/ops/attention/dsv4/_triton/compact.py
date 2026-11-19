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

"""Grouped selected attention over BF16 K/V rows on SM120."""

from __future__ import annotations

from tokenspeed_kernel._triton import triton, tl


@triton.jit(do_not_specialize=["WIDTH", "ROWS", "IS0", "IS1"])
def _compact_attention(
    Q,
    KV,
    Indices,
    Lens,
    Sink,
    Out,
    Lse,
    QS0,
    QS1,
    QS2,
    KS0,
    KS1,
    IS0,
    IS1,
    LS,
    SS,
    OS0,
    OS1,
    OS2,
    WIDTH,
    ROWS,
    HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LSE: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.arange(0, 512)
    q = tl.load(
        Q + token * QS0 + heads[:, None] * QS1 + d[None, :] * QS2,
        heads[:, None] < HEADS,
        0,
    )
    if HAS_SINK:
        maximum = tl.load(Sink + heads * SS, heads < HEADS, -float("inf")).to(
            tl.float32
        )
        denom = tl.full((16,), 1.0, tl.float32)
    else:
        maximum = tl.full((16,), -float("inf"), tl.float32)
        denom = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    length = tl.minimum(tl.maximum(tl.load(Lens + token * LS), 0), WIDTH)
    for start in range(0, length, B):
        col = start + tl.arange(0, B)
        rows = tl.load(Indices + token * IS0 + col * IS1, col < length, -1).to(tl.int64)
        valid = (col < length) & (rows >= 0) & (rows < ROWS)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(KV + rows[:, None] * KS0 + d[None, :] * KS1, valid[:, None], 0)
        logits = tl.dot(q, tl.trans(kv)) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.where(valid[None, :], tl.exp(logits - safe_max[:, None]), 0.0)
        kv_even, kv_odd = tl.split(kv.to(tl.float32).reshape(B, 256, 2))
        even = tl.dot(p, kv_even, input_precision="tf32x3")
        odd = tl.dot(p, kv_odd, input_precision="tf32x3")
        pv = tl.join(even, odd).reshape(16, 512)
        acc = acc * alpha[:, None] + pv
        denom = denom * alpha + tl.sum(p, 1)
        maximum = next_max
    value = tl.where(denom[:, None] > 0, acc / denom[:, None], 0)
    tl.store(
        Out + token * OS0 + heads[:, None] * OS1 + d[None, :] * OS2,
        value,
        heads[:, None] < HEADS,
    )
    if HAS_LSE:
        tl.store(
            Lse + token * HEADS + heads,
            tl.where(denom > 0, maximum + tl.log(denom), -float("inf")),
            heads < HEADS,
        )
