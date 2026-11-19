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

"""Direct readers for V4.1 selected attention with FP32 partial softmax."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41._triton.nvidia_fp4 import decode_e2m1_pair
from tokenspeed_kernel.ops.attention.dsv41.triton import _planar_offset


@triton.jit
def _selected_segment(
    q,
    maximum,
    denominator,
    acc,
    Cache,
    Slots,
    Lens,
    token,
    part,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    SS0,
    SS1,
    LS,
    CAPACITY,
    WIDTH,
    SCALE: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    FP4: tl.constexpr,
):
    d = tl.arange(0, 512)
    span = tl.cdiv(WIDTH, PARTS * B) * B
    length = tl.minimum(tl.maximum(tl.load(Lens + token * LS), 0), WIDTH)
    begin = part * span
    end = tl.minimum(begin + span, length)
    for start in range(begin, end, B):
        col = start + tl.arange(0, B)
        slot = tl.load(Slots + token * SS0 + col * SS1, col < end, -1).to(tl.int64)
        valid = (col < end) & (slot >= 0) & (slot < CAPACITY)
        slot = tl.where(valid, slot, 0)
        base = Cache + slot // 64 * CP
        row = slot % 64
        if FP4:
            byte = tl.load(
                base[:, None]
                + _planar_offset(row[:, None] * 256 + d[None, :] // 2, CR, CB, 288),
                valid[:, None],
                0,
            )
            scale = (
                tl.load(
                    base[:, None]
                    + _planar_offset(
                        64 * 256 + row[:, None] * 32 + d[None, :] // 16, CR, CB, 288
                    ),
                    valid[:, None],
                    0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            value = decode_e2m1_pair(byte, d[None, :] % 2)
        else:
            byte = tl.load(
                base[:, None]
                + _planar_offset(row[:, None] * 512 + d[None, :], CR, CB, 528),
                valid[:, None],
                0,
            )
            scale_byte = tl.load(
                base[:, None]
                + _planar_offset(
                    64 * 512 + row[:, None] * 16 + d[None, :] // 32, CR, CB, 528
                ),
                valid[:, None],
                0,
            )
            scale = tl.where(
                scale_byte == 0,
                2.0**-127,
                (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
            )
            value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Match the BF16 gather boundary before any attention arithmetic.
        kv = (value * scale).to(tl.bfloat16).to(tl.float32)
        logits = tl.sum(kv * q[None, :], 1) * SCALE
        logits = tl.where(valid, logits, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 0))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(maximum - safe_max)
        probabilities = tl.where(valid, tl.exp(logits - safe_max), 0.0)
        acc = acc * previous_scale + tl.sum(probabilities[:, None] * kv, 0)
        denominator = denominator * previous_scale + tl.sum(probabilities, 0)
        maximum = next_max
    return maximum, denominator, acc


@triton.jit(
    do_not_specialize=[
        "SS0",
        "SS1",
        "SLS",
        "GS0",
        "GS1",
        "GLS",
        "SCAP",
        "GCAP",
        "SWIDTH",
        "GWIDTH",
    ]
)
def _selected_attention_parts(
    Q,
    SCache,
    SSlots,
    SLens,
    GCache,
    GSlots,
    GLens,
    Sink,
    Partial,
    QS0,
    QS1,
    QS2,
    SP,
    SR: tl.constexpr,
    SB: tl.constexpr,
    SS0,
    SS1,
    SLS,
    GP,
    GR: tl.constexpr,
    GB: tl.constexpr,
    GS0,
    GS1,
    GLS,
    SCAP,
    GCAP,
    SWIDTH,
    GWIDTH,
    HEADS,
    SINK_STRIDE,
    HAS_GLOBAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCALE: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    part = tl.program_id(2)
    d = tl.arange(0, 512)
    q = tl.load(Q + token * QS0 + head * QS1 + d * QS2).to(tl.float32)
    if HAS_SINK:
        # The joint softmax includes the zero-valued sink exactly once.
        maximum = tl.where(part == 0, tl.load(Sink + head * SINK_STRIDE), -float("inf"))
        denominator = tl.where(part == 0, 1.0, 0.0)
    else:
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
    acc = tl.full((512,), 0.0, tl.float32)
    maximum, denominator, acc = _selected_segment(
        q,
        maximum,
        denominator,
        acc,
        SCache,
        SSlots,
        SLens,
        token,
        part,
        SP,
        SR,
        SB,
        SS0,
        SS1,
        SLS,
        SCAP,
        SWIDTH,
        SCALE,
        PARTS,
        B,
        False,
    )
    if HAS_GLOBAL:
        maximum, denominator, acc = _selected_segment(
            q,
            maximum,
            denominator,
            acc,
            GCache,
            GSlots,
            GLens,
            token,
            part,
            GP,
            GR,
            GB,
            GS0,
            GS1,
            GLS,
            GCAP,
            GWIDTH,
            SCALE,
            PARTS,
            B,
            True,
        )
    base = Partial + ((token * HEADS + head) * PARTS + part) * 514
    tl.store(base + d, acc)
    tl.store(base + 512, maximum)
    tl.store(base + 513, denominator)


@triton.jit
def _selected_segment_grouped(
    q,
    maximum,
    denominator,
    acc,
    Cache,
    Slots,
    Lens,
    token,
    part,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    SS0,
    SS1,
    LS,
    CAPACITY,
    WIDTH,
    SCALE: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    FP4: tl.constexpr,
):
    d = tl.arange(0, 512)
    span = tl.cdiv(WIDTH, PARTS * B) * B
    length = tl.minimum(tl.maximum(tl.load(Lens + token * LS), 0), WIDTH)
    begin = part * span
    end = tl.minimum(begin + span, length)
    for start in range(begin, end, B):
        col = start + tl.arange(0, B)
        slot = tl.load(Slots + token * SS0 + col * SS1, col < end, -1).to(tl.int64)
        valid = (col < end) & (slot >= 0) & (slot < CAPACITY)
        slot = tl.where(valid, slot, 0)
        base = Cache + slot // 64 * CP
        row = slot % 64
        if FP4:
            byte = tl.load(
                base[:, None]
                + _planar_offset(row[:, None] * 256 + d[None, :] // 2, CR, CB, 288),
                valid[:, None],
                0,
            )
            scale = (
                tl.load(
                    base[:, None]
                    + _planar_offset(
                        64 * 256 + row[:, None] * 32 + d[None, :] // 16, CR, CB, 288
                    ),
                    valid[:, None],
                    0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            value = decode_e2m1_pair(byte, d[None, :] % 2)
        else:
            byte = tl.load(
                base[:, None]
                + _planar_offset(row[:, None] * 512 + d[None, :], CR, CB, 528),
                valid[:, None],
                0,
            )
            scale_byte = tl.load(
                base[:, None]
                + _planar_offset(
                    64 * 512 + row[:, None] * 16 + d[None, :] // 32, CR, CB, 528
                ),
                valid[:, None],
                0,
            )
            scale = tl.where(
                scale_byte == 0,
                2.0**-127,
                (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
            )
            value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        kv = (value * scale).to(tl.bfloat16).to(tl.float32)
        logits = tl.dot(q.to(tl.bfloat16), tl.trans(kv.to(tl.bfloat16))) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(maximum - safe_max)
        probabilities = tl.where(
            valid[None, :], tl.exp(logits - safe_max[:, None]), 0.0
        )
        # Keep the probability's BF16 high and residual components.
        # KV is already exactly representable in BF16.
        probability_high = probabilities.to(tl.bfloat16)
        probability_low = (probabilities - probability_high.to(tl.float32)).to(
            tl.bfloat16
        )
        pv = tl.dot(probability_high, kv.to(tl.bfloat16))
        pv = tl.dot(probability_low, kv.to(tl.bfloat16), pv)
        acc = acc * previous_scale[:, None] + pv
        denominator = denominator * previous_scale + tl.sum(probabilities, 1)
        maximum = next_max
    return maximum, denominator, acc


@triton.jit(
    do_not_specialize=[
        "SS0",
        "SS1",
        "SLS",
        "GS0",
        "GS1",
        "GLS",
        "SCAP",
        "GCAP",
        "SWIDTH",
        "GWIDTH",
    ]
)
def _selected_attention_grouped_parts(
    Q,
    SCache,
    SSlots,
    SLens,
    GCache,
    GSlots,
    GLens,
    Sink,
    Partial,
    QS0,
    QS1,
    QS2,
    SP,
    SR: tl.constexpr,
    SB: tl.constexpr,
    SS0,
    SS1,
    SLS,
    GP,
    GR: tl.constexpr,
    GB: tl.constexpr,
    GS0,
    GS1,
    GLS,
    SCAP,
    GCAP,
    SWIDTH,
    GWIDTH,
    HEADS,
    SINK_STRIDE,
    HAS_GLOBAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCALE: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * 16 + tl.arange(0, 16)
    part = tl.program_id(2)
    d = tl.arange(0, 512)
    q = tl.load(
        Q + token * QS0 + head[:, None] * QS1 + d[None, :] * QS2,
        head[:, None] < HEADS,
        0,
    ).to(tl.float32)
    if HAS_SINK:
        maximum = tl.where(
            part == 0,
            tl.load(Sink + head * SINK_STRIDE, head < HEADS, -float("inf")),
            -float("inf"),
        )
        denominator = tl.full((16,), 0.0, tl.float32) + tl.where(part == 0, 1.0, 0.0)
    else:
        maximum = tl.full((16,), -float("inf"), tl.float32)
        denominator = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    maximum, denominator, acc = _selected_segment_grouped(
        q,
        maximum,
        denominator,
        acc,
        SCache,
        SSlots,
        SLens,
        token,
        part,
        SP,
        SR,
        SB,
        SS0,
        SS1,
        SLS,
        SCAP,
        SWIDTH,
        SCALE,
        PARTS,
        B,
        False,
    )
    if HAS_GLOBAL:
        maximum, denominator, acc = _selected_segment_grouped(
            q,
            maximum,
            denominator,
            acc,
            GCache,
            GSlots,
            GLens,
            token,
            part,
            GP,
            GR,
            GB,
            GS0,
            GS1,
            GLS,
            GCAP,
            GWIDTH,
            SCALE,
            PARTS,
            B,
            True,
        )
    base = Partial + ((token * HEADS + head) * PARTS + part) * 514
    tl.store(base[:, None] + d[None, :], acc, head[:, None] < HEADS)
    tl.store(base + 512, maximum, head < HEADS)
    tl.store(base + 513, denominator, head < HEADS)


@triton.jit
def _selected_attention_merge(
    Partial,
    Out,
    Lse,
    OS0,
    OS1,
    HEADS,
    PARTS: tl.constexpr,
    P: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    p = tl.arange(0, P)
    d = tl.arange(0, 512)
    base = Partial + ((token * HEADS + head) * PARTS + p) * 514
    maxima = tl.load(base + 512, p < PARTS, -float("inf"))
    denoms = tl.load(base + 513, p < PARTS, 0.0)
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    scales = tl.exp(maxima - safe_max)
    denom = tl.sum(denoms * scales, 0)
    acc = tl.load(base[:, None] + d[None, :], p[:, None] < PARTS, 0.0)
    value = tl.sum(acc * scales[:, None], 0)
    tl.store(
        Out + token * OS0 + head * OS1 + d, tl.where(denom > 0.0, value / denom, 0.0)
    )
    if HAS_LSE:
        tl.store(
            Lse + token * HEADS + head,
            tl.where(denom > 0.0, maximum + tl.log(denom), -float("inf")),
        )


def launch_selected_attention(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    lse,
    query_chunk_size,
):
    """Launch paged readers and a merge with at most 128 MiB of scratch."""
    tokens, heads, _ = q.shape
    if not tokens:
        return
    vector = tokens == 1 and heads <= 16
    head_groups = heads if vector else triton.cdiv(heads, 16)
    if vector:
        parts, tile = 32, 8
        kernel = _selected_attention_parts
    else:
        # Bucket the occupancy target to four split counts. New request counts
        # do not create a new binary once these startup tactics are warm.
        target = max(4, min(32, 128 // (head_groups * min(tokens, 32))))
        if head_groups == 1 and tokens <= 16:
            target = max(target, 16)
        parts, tile = 1 << (target.bit_length() - 1), 16
        kernel = _selected_attention_grouped_parts
    scratch_per_query = heads * parts * 514 * 4
    chunk = min(query_chunk_size, 32, max(1, (128 << 20) // scratch_per_query))
    partial = torch.empty(
        (min(tokens, chunk), heads, parts, 514), device=q.device, dtype=torch.float32
    )
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        query = q[start:end]
        ss, sl = swa_slots[start:end], swa_lens[start:end]
        gs = None if global_slots is None else global_slots[start:end]
        gl = None if global_lens is None else global_lens[start:end]
        global_strides = (
            (*global_cache.stride(), *gs.stride(), gl.stride(0))
            if global_cache is not None
            else (0, 0, 0, 0, 0, 0)
        )
        kernel[(end - start, head_groups, parts)](
            query,
            swa_cache,
            ss,
            sl,
            global_cache,
            gs,
            gl,
            attn_sink,
            partial,
            *query.stride(),
            *swa_cache.stride(),
            *ss.stride(),
            sl.stride(0),
            *global_strides,
            swa_cache.shape[0] * 64,
            global_cache.shape[0] * 64 if global_cache is not None else 0,
            ss.shape[1],
            gs.shape[1] if gs is not None else 0,
            heads,
            0 if attn_sink is None else attn_sink.stride(0),
            global_cache is not None,
            attn_sink is not None,
            softmax_scale,
            parts,
            tile,
            num_warps=4,
            num_stages=1,
        )
        _selected_attention_merge[(end - start, heads)](
            partial,
            out[start:end],
            None if lse is None else lse[start:end],
            out.stride(0),
            out.stride(1),
            heads,
            parts,
            parts,
            lse is not None,
            num_warps=4,
        )
