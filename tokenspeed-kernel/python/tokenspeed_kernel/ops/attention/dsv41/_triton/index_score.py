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


# ---------------------------------------------------------------------------
# Grouped scoring (opt-in through flashinfer.py). A score tile is computed
# exactly as above, operand shapes and reduction order included, so scores
# are the same bits. What changes is the work around a tile: a program keeps
# its queries and walks several history tiles, a tile's two pages come from
# two table entries instead of one per column, and the per-query launch
# visits only programs that have a tile left to score.
# ---------------------------------------------------------------------------


@triton.jit
def _load_query_pair(Q, W, Visible, first, N, QS0, QS1, QS2, WS0, WS1, VS, TABLE_WIDTH):
    # Rows 0..31 are the heads of query `first`, rows 32..63 of `first + 1`.
    row = tl.arange(0, 64)
    qidx = first + row // 32
    head = row % 32
    packed_d = tl.arange(0, 64)
    scale_d = tl.arange(0, 4)
    q_codes = tl.load(
        Q + qidx[:, None] * QS0 + head[:, None] * QS1 + packed_d[None, :] * QS2,
        qidx[:, None] < N,
        0,
    )
    q_scales = tl.load(
        Q + qidx[:, None] * QS0 + head[:, None] * QS1 + (64 + scale_d[None, :]) * QS2,
        qidx[:, None] < N,
        127,
    )
    weights = tl.load(W + qidx * WS0 + head * WS1, qidx < N, 0)
    limit = TABLE_WIDTH.to(tl.int64) * 64
    visible_a = tl.load(Visible + first * VS, first < N, 0).to(tl.int64)
    visible_b = tl.load(Visible + (first + 1) * VS, first + 1 < N, 0).to(tl.int64)
    visible_a = tl.minimum(tl.maximum(visible_a, 0), limit)
    visible_b = tl.minimum(tl.maximum(visible_b, 0), limit)
    return q_codes, q_scales, weights, visible_a, visible_b


@triton.jit
def _score_query_pair(
    q_codes,
    q_scales,
    weights,
    visible_a,
    visible_b,
    key_codes,
    key_scales,
    ok0,
    ok1,
    tile,
    WIDTH,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    B: tl.constexpr,
):
    # Column masks come from scalars, so no per-column tensor changes layout.
    i = tl.arange(0, B)
    col = tile * B + i
    visible = tl.where(tl.arange(0, 2) == 0, visible_a, visible_b)
    page_ok = tl.where(i < 64, ok0, ok1)
    valid = (col[None, :] < WIDTH) & (col[None, :] < visible[:, None]) & page_ok[None, :]
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
        .to(weights.dtype)
        .to(tl.float32)
    )
    partials = tl.sum(products.reshape(2, SHARDS, SHARD_HEADS, B), 2)
    partials = partials.to(weights.dtype).to(tl.float32)
    rank_ids = tl.arange(0, SHARDS)
    scores = tl.full((2, B), 0.0, tl.float32)
    for rank in tl.static_range(SHARDS):
        partial = tl.sum(tl.where(rank_ids[None, :, None] == rank, partials, 0.0), 1)
        scores = scores + partial
    scores = scores.to(weights.dtype).to(tl.float32)
    return tl.where(valid & (scores > -float("inf")), scores, -float("inf"))


@triton.jit
def _pair_block_scores(scores, visible_a, visible_b, tile, B: tl.constexpr):
    block_scores = tl.max(scores.reshape(2, B // 8, 8), 2)
    ids = tile * (B // 8) + tl.arange(0, B // 8)
    visible = tl.where(tl.arange(0, 2) == 0, visible_a, visible_b)
    return tl.where(
        (ids[None, :] == (visible[:, None] - 1) // 8) & (block_scores > -float("inf")),
        float("inf"),
        block_scores,
    )


@triton.jit(do_not_specialize=["N", "PAGES", "TABLE_WIDTH", "WIDTH", "TS0", "TL"])
def _index_score_pair_history_tiles(
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
    TL,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    B: tl.constexpr,
    MAKE_BLOCKS: tl.constexpr,
):
    """Score TL history tiles for one query pair.

    Program (p, t) owns queries 2 * p and 2 * p + 1 and tiles
    [TL * t, TL * (t + 1)). Pages come from the first query's table row.
    Without Shared the caller passes a table whose rows are one row. With
    Shared (TL <= 64) the pair is scored when its two rows agree on every
    page the second query can see in these tiles; Shared[p, tile] records
    that for the per-query launch, which scores the rest.
    """
    # A tile is two 64-row pages, and WIDTH is a whole number of pages.
    tl.static_assert(B == 128)
    pair = tl.program_id(0).to(tl.int64)
    first = pair * 2
    second = first + 1
    tile_count = tl.cdiv(WIDTH, B).to(tl.int64)
    t_begin = tl.program_id(1).to(tl.int64) * TL
    t_end = tl.minimum(t_begin + TL, tile_count)
    same = True
    if Shared is not None:
        # Unclamped, as the per-tile check reads it; 0 for a missing query.
        raw_b = tl.load(Visible + second * VS, second < N, 0)
        # One answer covers the program's tiles, so the scoring loop below
        # has no branch and keeps its query operands across tiles.
        same = first < N
        for tile in range(t_begin, t_end):
            c0 = tile * B
            c1 = c0 + 64
            a0 = tl.load(Table + first * TS0 + (c0 // 64) * TS1, c0 < WIDTH, -1)
            a1 = tl.load(Table + first * TS0 + (c1 // 64) * TS1, c1 < WIDTH, -1)
            b0 = tl.load(
                Table + second * TS0 + (c0 // 64) * TS1,
                (second < N) & (c0 < WIDTH),
                -1,
            )
            b1 = tl.load(
                Table + second * TS0 + (c1 // 64) * TS1,
                (second < N) & (c1 < WIDTH),
                -1,
            )
            same = same & ((a0 == b0) | (c0 >= raw_b)) & ((a1 == b1) | (c1 >= raw_b))
        offset = tl.arange(0, 64)
        tl.store(
            Shared + pair * tile_count + t_begin + offset,
            tl.full((64,), 0, tl.uint8) + same.to(tl.uint8),
            (offset < TL) & (t_begin + offset < tile_count),
        )
    if same:
        packed_d = tl.arange(0, 64)
        scale_d = tl.arange(0, 4)
        i = tl.arange(0, B)
        kr = i % 64
        q_codes, q_scales, weights, visible_a, visible_b = _load_query_pair(
            Q, W, Visible, first, N, QS0, QS1, QS2, WS0, WS1, VS, TABLE_WIDTH
        )
        bound = tl.maximum(visible_a, visible_b)
        query = first + tl.arange(0, 2)
        for tile in range(t_begin, t_end):
            c0 = tile * B
            c1 = c0 + 64
            p0 = tl.load(
                Table + first * TS0 + (c0 // 64) * TS1,
                (c0 < WIDTH) & (c0 < bound),
                -1,
            ).to(tl.int64)
            p1 = tl.load(
                Table + first * TS0 + (c1 // 64) * TS1,
                (c1 < WIDTH) & (c1 < bound),
                -1,
            ).to(tl.int64)
            ok0 = (p0 >= 0) & (p0 < PAGES)
            ok1 = (p1 >= 0) & (p1 < PAGES)
            p0 = tl.where(ok0, p0, 0)
            p1 = tl.where(ok1, p1, 0)
            key_codes = tl.load(
                Cache
                + tl.where(i[None, :] < 64, p0, p1) * CP
                + _planar_offset(kr[None, :] * 64 + packed_d[:, None], CR, CB, 68),
                tl.where(i[None, :] < 64, ok0, ok1),
                0,
            )
            key_scales = tl.load(
                Cache
                + tl.where(i[:, None] < 64, p0, p1) * CP
                + _planar_offset(
                    64 * 64 + kr[:, None] * 4 + scale_d[None, :], CR, CB, 68
                ),
                tl.where(i[:, None] < 64, ok0, ok1),
                127,
            )
            scores = _score_query_pair(
                q_codes, q_scales, weights, visible_a, visible_b, key_codes,
                key_scales, ok0, ok1, tile, WIDTH, SHARD_HEADS, SHARDS, B,
            )
            cols = c0 + i
            tl.store(
                Rows + query[:, None] * WIDTH + cols[None, :],
                scores,
                (query[:, None] < N) & (cols[None, :] < WIDTH),
            )
            if MAKE_BLOCKS:
                block_ids = tile * (B // 8) + tl.arange(0, B // 8)
                tl.store(
                    Blocks + query[:, None] * (WIDTH // 8) + block_ids[None, :],
                    _pair_block_scores(scores, visible_a, visible_b, tile, B),
                    (query[:, None] < N) & (block_ids[None, :] < WIDTH // 8),
                )


@triton.jit(do_not_specialize=["PAGES", "TABLE_WIDTH", "WIDTH", "TS0", "TL"])
def _index_score_looped_tiles(
    Q,
    W,
    Cache,
    Table,
    Visible,
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
    PAGES,
    TABLE_WIDTH,
    WIDTH,
    TL,
    HEADS: tl.constexpr,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    MAKE_BLOCKS: tl.constexpr,
):
    """Full _index_score_tiles for the tiles a paged launch left unscored.

    Program (q, t) owns query q and tiles [TL * t, TL * (t + 1)), TL <= 64.
    It scores the tiles whose Shared flag is zero with that kernel's tile
    arithmetic and returns at once when its tiles have none.
    """
    tl.static_assert(B == 128)
    query = tl.program_id(0).to(tl.int64)
    tile_count = tl.cdiv(WIDTH, B).to(tl.int64)
    t_begin = tl.program_id(1).to(tl.int64) * TL
    t_end = tl.minimum(t_begin + TL, tile_count)
    offset = tl.arange(0, 64)
    flags = tl.load(
        Shared + (query // 2) * tile_count + t_begin + offset,
        (offset < TL) & (t_begin + offset < tile_count),
        1,
    )
    if tl.min(flags.to(tl.int32), 0) == 0:
        visible = tl.minimum(
            tl.maximum(tl.load(Visible + query * VS), 0), TABLE_WIDTH.to(tl.int64) * 64
        )
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
        for tile in range(t_begin, t_end):
            if tl.load(Shared + (query // 2) * tile_count + tile) == 0:
                start = tile * B
                col = start + tl.arange(0, B)
                valid = (col < WIDTH) & (col < visible)
                page = tl.load(Table + query * TS0 + col // 64 * TS1, valid, -1).to(
                    tl.int64
                )
                valid = valid & (page >= 0) & (page < PAGES)
                base = Cache + page * CP
                row = col % 64
                key_codes = tl.load(
                    base[None, :]
                    + _planar_offset(row[None, :] * 64 + packed_d[:, None], CR, CB, 68),
                    valid[None, :],
                    0,
                )
                key_scales = tl.load(
                    base[:, None]
                    + _planar_offset(
                        64 * 64 + row[:, None] * 4 + scale_d[None, :], CR, CB, 68
                    ),
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
                partials = tl.sum(products.reshape(SHARDS, SHARD_HEADS, B), 1)
                partials = partials.to(W.dtype.element_ty).to(tl.float32)
                rank_ids = tl.arange(0, SHARDS)
                scores = tl.full((B,), 0.0, tl.float32)
                for rank in tl.static_range(SHARDS):
                    partial = tl.sum(
                        tl.where(rank_ids[:, None] == rank, partials, 0.0), 0
                    )
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
                        Blocks + query * (WIDTH // 8) + ids,
                        block_scores,
                        ids < WIDTH // 8,
                    )
