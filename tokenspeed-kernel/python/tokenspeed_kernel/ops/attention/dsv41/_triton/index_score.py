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

"""Score packed MXFP4 queries and history on SM120 tensor cores."""

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import _planar_offset


@triton.jit(do_not_specialize=["PAGES", "TABLE_WIDTH", "WIDTH"])
def _index_score_tiles(
    Q,
    W,
    Cache,
    Table,
    Visible,
    Candidates,
    Rows,
    Blocks,
    Shared,
    QS0,
    QS1,
    QS2,
    WS0,
    WS1,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    TS0,
    TS1,
    VS,
    CS0,
    CS1,
    PAGES,
    TABLE_WIDTH,
    WIDTH,
    HEADS: tl.constexpr,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    MAKE_BLOCKS: tl.constexpr,
    HAS_CANDIDATES: tl.constexpr,
):
    shared = False
    if Shared is not None:
        pair = tl.program_id(0).to(tl.int64) // 2
        tile_count = tl.cdiv(WIDTH, B)
        shared = tl.load(Shared + pair * tile_count + tl.program_id(1)) != 0
    if not shared:
        query = tl.program_id(0).to(tl.int64)
        start = tl.program_id(1).to(tl.int64) * B
        col = start + tl.arange(0, B)
        visible = tl.minimum(
            tl.maximum(tl.load(Visible + query * VS), 0), TABLE_WIDTH.to(tl.int64) * 64
        )
        if HAS_CANDIDATES:
            block = tl.load(
                Candidates + query * CS0 + col // 8 * CS1, col < WIDTH, -1
            ).to(tl.int64)
            logical = tl.where(block >= 0, block * 8 + col % 8, -1)
        else:
            logical = col
        valid = (col < WIDTH) & (logical >= 0) & (logical < visible)
        page = tl.load(Table + query * TS0 + logical // 64 * TS1, valid, -1).to(
            tl.int64
        )
        valid = valid & (page >= 0) & (page < PAGES)
        h = tl.arange(0, H)
        packed_d = tl.arange(0, 64)
        scale_d = tl.arange(0, 4)
        q_codes = tl.load(
            Q + query * QS0 + h[:, None] * QS1 + packed_d[None, :] * QS2,
            h[:, None] < HEADS,
            0,
        )
        q_scales = tl.load(
            Q + query * QS0 + h[:, None] * QS1 + (64 + scale_d[None, :]) * QS2,
            h[:, None] < HEADS,
            127,
        )
        weights = tl.load(W + query * WS0 + h * WS1, h < HEADS, 0)
        base = Cache + page * CP
        row = logical % 64
        key_codes = tl.load(
            base[None, :]
            + _planar_offset(row[None, :] * 64 + packed_d[:, None], CR, CB, 68),
            valid[None, :],
            0,
        )
        key_scales = tl.load(
            base[:, None]
            + _planar_offset(64 * 64 + row[:, None] * 4 + scale_d[None, :], CR, CB, 68),
            valid[:, None],
            127,
        )
        dots = (
            tl.dot_scaled(
                q_codes,
                q_scales,
                "e2m1",
                key_codes,
                key_scales,
                "e2m1",
                fast_math=False,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        products = (
            (tl.maximum(dots, 0.0) * weights[:, None].to(tl.float32))
            .to(W.dtype.element_ty)
            .to(tl.float32)
        )
        # Reduce each contiguous head shard once, then retain its rounding
        # boundary and the existing rank-order sum.
        partials = tl.sum(products.reshape(SHARDS, SHARD_HEADS, B), 1)
        partials = partials.to(W.dtype.element_ty).to(tl.float32)
        rank_ids = tl.arange(0, SHARDS)
        scores = tl.full((B,), 0.0, tl.float32)
        for rank in tl.static_range(SHARDS):
            partial = tl.sum(tl.where(rank_ids[:, None] == rank, partials, 0.0), 0)
            scores = scores + partial
        scores = scores.to(W.dtype.element_ty).to(tl.float32)
        scores = tl.where(valid, scores, -float("inf"))
        # Match the portable selector: NaN and -inf entries are invalid.
        scores = tl.where(scores > -float("inf"), scores, -float("inf"))
        tl.store(Rows + query * WIDTH + col, scores, col < WIDTH)
        if MAKE_BLOCKS:
            block_scores = tl.max(scores.reshape(B // 8, 8), 1)
            ids = start // 8 + tl.arange(0, B // 8)
            block_scores = tl.where(
                (ids == (visible - 1) // 8) & (block_scores > -float("inf")),
                float("inf"),
                block_scores,
            )
            tl.store(
                Blocks + query * (WIDTH // 8) + ids, block_scores, ids < WIDTH // 8
            )


@triton.jit(do_not_specialize=["N", "PAGES", "TABLE_WIDTH", "WIDTH", "TS0"])
def _index_score_shared_history_tiles(
    Q,
    W,
    Cache,
    Table,
    Visible,
    Rows,
    Blocks,
    Shared,
    N,
    QS0,
    QS1,
    QS2,
    WS0,
    WS1,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    TS0,
    TS1,
    VS,
    PAGES,
    TABLE_WIDTH,
    WIDTH,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    BQ: tl.constexpr,
    B: tl.constexpr,
    MAKE_BLOCKS: tl.constexpr,
):
    tl.static_assert(BQ == 2)
    same = True
    if Shared is not None:
        pair_query = tl.program_id(0).to(tl.int64) * BQ + tl.arange(0, BQ)
        pair_col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
        pair_visible = tl.load(Visible + pair_query * VS, pair_query < N, 0)
        pair_page = tl.load(
            Table + pair_query[:, None] * TS0 + pair_col[None, :] // 64 * TS1,
            (pair_query[:, None] < N) & (pair_col[None, :] < WIDTH),
            -1,
        ).to(tl.int64)
        page0, page1 = tl.split(pair_page.trans())
        active = (pair_query[:, None] < N) & (pair_col[None, :] < pair_visible[:, None])
        _, active1 = tl.split(active.trans())
        same = tl.sum(((page0 != page1) & active1).to(tl.int32), 0) == 0
        tile_count = tl.cdiv(WIDTH, B)
        tl.store(
            Shared + tl.program_id(0).to(tl.int64) * tile_count + tl.program_id(1),
            same,
        )
    if same:
        query = tl.program_id(0).to(tl.int64) * BQ + tl.arange(0, BQ)
        row = tl.arange(0, BQ * 32)
        qidx = tl.program_id(0).to(tl.int64) * BQ + row // 32
        head = row % 32
        col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
        visible = tl.load(Visible + query * VS, query < N, 0)
        visible = tl.minimum(tl.maximum(visible, 0), TABLE_WIDTH.to(tl.int64) * 64)
        page = tl.load(
            Table + tl.program_id(0).to(tl.int64) * BQ * TS0 + col // 64 * TS1,
            (col < WIDTH) & (col < tl.max(visible, 0)),
            -1,
        ).to(tl.int64)
        page_ok = (page >= 0) & (page < PAGES)
        page = tl.where(page_ok, page, 0)
        valid = (
            (col[None, :] < WIDTH)
            & (col[None, :] < visible[:, None])
            & page_ok[None, :]
        )
        base = Cache + page * CP
        kr = col % 64
        packed_d = tl.arange(0, 64)
        scale_d = tl.arange(0, 4)
        key_codes = tl.load(
            base[None, :]
            + _planar_offset(kr[None, :] * 64 + packed_d[:, None], CR, CB, 68),
            page_ok[None, :],
            0,
        )
        key_scales = tl.load(
            base[:, None]
            + _planar_offset(64 * 64 + kr[:, None] * 4 + scale_d[None, :], CR, CB, 68),
            page_ok[:, None],
            127,
        )
        q_codes = tl.load(
            Q + qidx[:, None] * QS0 + head[:, None] * QS1 + packed_d[None, :] * QS2,
            qidx[:, None] < N,
            0,
        )
        q_scales = tl.load(
            Q
            + qidx[:, None] * QS0
            + head[:, None] * QS1
            + (64 + scale_d[None, :]) * QS2,
            qidx[:, None] < N,
            127,
        )
        weights = tl.load(W + qidx * WS0 + head * WS1, qidx < N, 0)
        dots = (
            tl.dot_scaled(
                q_codes,
                q_scales,
                "e2m1",
                key_codes,
                key_scales,
                "e2m1",
                fast_math=False,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        products = (
            (tl.maximum(dots, 0.0) * weights[:, None].to(tl.float32))
            .to(W.dtype.element_ty)
            .to(tl.float32)
        )
        partials = tl.sum(products.reshape(BQ, SHARDS, SHARD_HEADS, B), 2)
        partials = partials.to(W.dtype.element_ty).to(tl.float32)
        rank_ids = tl.arange(0, SHARDS)
        scores = tl.full((BQ, B), 0.0, tl.float32)
        for rank in tl.static_range(SHARDS):
            partial = tl.sum(
                tl.where(rank_ids[None, :, None] == rank, partials, 0.0), 1
            )
            scores = scores + partial
        scores = scores.to(W.dtype.element_ty).to(tl.float32)
        scores = tl.where(valid & (scores > -float("inf")), scores, -float("inf"))
        tl.store(
            Rows + query[:, None] * WIDTH + col[None, :],
            scores,
            (query[:, None] < N) & (col[None, :] < WIDTH),
        )
        if MAKE_BLOCKS:
            block_scores = tl.max(scores.reshape(BQ, B // 8, 8), 2)
            ids = tl.program_id(1).to(tl.int64) * (B // 8) + tl.arange(0, B // 8)
            block_scores = tl.where(
                (ids[None, :] == (visible[:, None] - 1) // 8)
                & (block_scores > -float("inf")),
                float("inf"),
                block_scores,
            )
            tl.store(
                Blocks + query[:, None] * (WIDTH // 8) + ids[None, :],
                block_scores,
                (query[:, None] < N) & (ids[None, :] < WIDTH // 8),
            )
