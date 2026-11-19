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

"""Opt-in variants of the SM120 selected attention over BF16 K/V rows."""

from __future__ import annotations

from tokenspeed_kernel._triton import triton, tl


@triton.jit(do_not_specialize=["WIDTH", "ROWS", "IS0", "IS1"])
def _compact_attention_variant(
    Q,
    KV,
    Indices,
    Lens,
    Sink,
    Out,
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
    B: tl.constexpr,
    HG: tl.constexpr,
    PV: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # The loop, masking, QK product, softmax and denominator are those of
    # _compact_attention. Two things are selectable, both fixed per process:
    #   HG  heads sharing one selected K/V tile (16 there).
    #   PV  how probabilities meet the K/V tile:
    #       "tf32x3"  FP32 probabilities, three TF32 passes (as there);
    #       "bf16x2"  each FP32 probability as a BF16 high part plus a BF16
    #                 residual, two BF16 passes into one FP32 accumulator (the
    #                 split of the V4.1 paged decode kernel);
    #       "bf16"    probabilities rounded to BF16, one pass.
    # SPLIT multiplies the even and odd value channels separately, as
    # _compact_attention does; it is a tiling choice of the launcher.
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HG + tl.arange(0, HG)
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
        denom = tl.full((HG,), 1.0, tl.float32)
    else:
        maximum = tl.full((HG,), -float("inf"), tl.float32)
        denom = tl.full((HG,), 0.0, tl.float32)
    acc = tl.full((HG, 512), 0.0, tl.float32)
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
        if PV == "tf32x3":
            kv_even, kv_odd = tl.split(kv.to(tl.float32).reshape(B, 256, 2))
            even = tl.dot(p, kv_even, input_precision="tf32x3")
            odd = tl.dot(p, kv_odd, input_precision="tf32x3")
            pv = tl.join(even, odd).reshape(HG, 512)
        elif PV == "bf16x2":
            # K/V is BF16 already, so only the probability needs two terms.
            p_high = p.to(tl.bfloat16)
            p_low = (p - p_high.to(tl.float32)).to(tl.bfloat16)
            if SPLIT:
                kv_even, kv_odd = tl.split(kv.reshape(B, 256, 2))
                even = tl.dot(p_low, kv_even, tl.dot(p_high, kv_even))
                odd = tl.dot(p_low, kv_odd, tl.dot(p_high, kv_odd))
                pv = tl.join(even, odd).reshape(HG, 512)
            else:
                pv = tl.dot(p_high, kv)
                pv = tl.dot(p_low, kv, pv)
        else:
            pv = tl.dot(p.to(tl.bfloat16), kv)
        acc = acc * alpha[:, None] + pv
        # The denominator keeps the FP32 probabilities in every mode.
        denom = denom * alpha + tl.sum(p, 1)
        maximum = next_max
    value = tl.where(denom[:, None] > 0, acc / denom[:, None], 0)
    tl.store(
        Out + token * OS0 + heads[:, None] * OS1 + d[None, :] * OS2,
        value,
        heads[:, None] < HEADS,
    )


@triton.jit(do_not_specialize=["WIDTH", "ROWS", "IS0", "IS1"])
def _compact_attention_bf16x2_pipelined(
    Q,
    KV,
    Indices,
    Lens,
    Sink,
    Out,
    QS0,
    QS1,
    QS2,
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
    B: tl.constexpr,
    HG: tl.constexpr,
    KS0: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    # RESIDUAL selects the PV arithmetic: True is PV == "bf16x2" of
    # _compact_attention_variant (BF16 high part plus BF16 residual), False is
    # PV == "bf16" (probabilities rounded to BF16, one product, a third less
    # tensor-core arithmetic). With RESIDUAL it is that kernel's arithmetic,
    # value for value and in the same order, so the output is bit-identical to
    # it. Without it a row of several tiles can round differently from that
    # kernel's bf16 launch: a few stored values in 10^4, a relative L2 gap
    # below 1e-4, against the mode's own 2e-3 error. Shared by both modes:
    # the same 32-row tiles, QK chains, running-maximum rescale, high-plus-
    # residual BF16 probabilities and FP32 denominator. Only where the work
    # runs differs, and three constructs below exist for that reason alone.
    #
    # 1. HG heads (32) share each selected K/V tile, so a query row fetches
    #    its tiles twice instead of four times. The fetch is bound by the
    #    path from L2 to the SM, about 0.9 ms of the 16-head kernel at 2048
    #    rows.
    # 2. The PV products sit in their own region (the always-true `if`). Dots
    #    of one region share one warp layout, and PV's [1, 8] would make the
    #    eight warps compute the four QK column tiles twice. On its own QK
    #    gets [2, 4]: one chain per warp, no duplicate.
    # 3. The launcher asks for five pipeline stages. With the dependent
    #    index load that gives the K/V tile two shared-memory buffers filled
    #    by asynchronous copies issued one iteration ahead, so the fetch
    #    overlaps the previous tile's arithmetic and the tile never passes
    #    through registers. Q (32 KiB) and the two buffers (64 KiB) leave
    #    3072 bytes of the device's 101376: the residual probabilities are
    #    therefore formed after the high product, which lets both BF16
    #    operands use one 2048-byte slot.
    #
    # KS0 is the workspace row stride in elements, a compile-time constant
    # (512 for the dense workspace); a workspace row index times KS0 plus a
    # channel stays below 2**31 for the row counts the launcher admits, so
    # int32 indices keep their width.
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HG + tl.arange(0, HG)
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
        denom = tl.full((HG,), 1.0, tl.float32)
    else:
        maximum = tl.full((HG,), -float("inf"), tl.float32)
        denom = tl.full((HG,), 0.0, tl.float32)
    acc = tl.full((HG, 512), 0.0, tl.float32)
    length = tl.minimum(tl.maximum(tl.load(Lens + token * LS), 0), WIDTH)
    for start in range(0, length, B):
        col = start + tl.arange(0, B)
        rows = tl.load(Indices + token * IS0 + col * IS1, col < length, -1)
        valid = (col < length) & (rows >= 0) & (rows < ROWS)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(KV + (rows[:, None] * KS0 + d[None, :]), valid[:, None], 0)
        logits = tl.dot(q, tl.trans(kv)) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.where(valid[None, :], tl.exp(logits - safe_max[:, None]), 0.0)
        p_high = p.to(tl.bfloat16)
        if start < length:
            pv = tl.dot(p_high, kv)
            if RESIDUAL:
                p_low = (p - p_high.to(tl.float32)).to(tl.bfloat16)
                pv = tl.dot(p_low, kv, pv)
        else:
            pv = tl.zeros((HG, 512), tl.float32)
        acc = acc * alpha[:, None] + pv
        # The denominator keeps the FP32 probabilities.
        denom = denom * alpha + tl.sum(p, 1)
        maximum = next_max
    value = tl.where(denom[:, None] > 0, acc / denom[:, None], 0)
    tl.store(
        Out + token * OS0 + heads[:, None] * OS1 + d[None, :] * OS2,
        value,
        heads[:, None] < HEADS,
    )
