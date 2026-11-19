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

"""SM120 paged index scoring with exact native TopK selection, and opt-in
paged decode attention on FlashInfer's mixed-cache kernels."""

from __future__ import annotations

import logging
import os

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.dsv41._triton.index_score import (
    _index_score_looped_tiles,
    _index_score_pair_history_tiles,
    _index_score_shared_history_tiles,
    _index_score_tiles,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _finish_topk,
    _index_gather_heads,
    _index_topk_outputs,
    cache_pack,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    index_topk as portable_index_topk,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    triton_dsv41_selected_attention_sm120,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.dsv41_mixed_decode import (
    MAX_SPLITS,
    Dsv41MixedDecode,
    dsv41_mixed_decode_chunks,
    dsv41_mixed_decode_splits,
    is_dsv41_mixed_decode_available,
    load_dsv41_mixed_decode,
)
from tokenspeed_kernel.thirdparty.flashinfer.topk import (
    is_graph_safe_topk_available,
    topk_small_indices,
)

logger = logging.getLogger(__name__)

_SCORE_BUDGET_BYTES = 128 * 1024 * 1024
_MAX_SCORE_COLUMNS = 1 << 21
_platform = current_platform()

# "1" turns on grouped index scoring and 16-bit score rows (README, "SM120
# grouped index scoring"); unset or any other value leaves this kernel as it
# was. Read once, at import.
SM120_INDEX_GROUPED_ENV = "TOKENSPEED_SM120_DSV41_INDEX_GROUPED"
_INDEX_GROUPED = os.environ.get(SM120_INDEX_GROUPED_ENV, "0") == "1"
# A program walks several tiles with its queries loaded once, as long as the
# launch keeps about _GROUP_PROGRAMS programs for the SMs. The bounds are for
# a broadcast page table, for rows with their own tables, and for the
# per-query launch, whose flag window is 64 tiles.
_GROUP_PROGRAMS = 2000
_GROUP_MAX_TILES = 16
_PAGED_MAX_TILES = 8
_LOOPED_MAX_TILES = 16
assert max(_PAGED_MAX_TILES, _LOOPED_MAX_TILES) <= 64


def _tiles_per_program(programs_per_tile: int, tiles: int, limit: int) -> int:
    """Tiles one program walks. A launch argument, so no shape recompiles."""
    count = 1
    while (
        count < limit
        and programs_per_tile * triton.cdiv(tiles, 2 * count) >= _GROUP_PROGRAMS
    ):
        count *= 2
    return count


def _score_grouped(
    q, w, index_cache, table, visible, rows, blocks, shard_heads, shards
):
    """Fill rows and blocks with the per-tile Full kernels' scores, bit for bit.

    q is the packed [queries, 32, 68] tile; rows is [queries, width] and
    blocks [queries, width // 8] or empty, in the weight dtype or FP32. A
    broadcast page table takes one launch. Rows with their own tables take
    two: the second scores, query by query, the tiles whose pair of rows read
    different pages, and is otherwise a few thousand programs that return
    after one flag read. Grids and tile counts follow shapes only, so a
    captured graph replays them on live tables, lengths and queries.
    """
    queries, width = rows.shape
    tiles = triton.cdiv(width, 128)
    pairs = triton.cdiv(queries, 2)
    make_blocks = blocks.shape[1] > 0
    strides = (
        *q.stride(),
        *w.stride(),
        *index_cache.stride(),
        *table.stride(),
        visible.stride(0),
    )
    flags = None
    if table.stride(0) != 0:
        flags = torch.empty((pairs, tiles), dtype=torch.uint8, device=q.device)
    count = _tiles_per_program(
        pairs, tiles, _GROUP_MAX_TILES if flags is None else _PAGED_MAX_TILES
    )
    _index_score_pair_history_tiles[(pairs, triton.cdiv(tiles, count))](
        q,
        w,
        index_cache,
        table,
        visible,
        rows,
        blocks,
        flags,
        queries,
        *strides,
        index_cache.shape[0],
        table.shape[1],
        width,
        count,
        shard_heads,
        shards,
        128,
        make_blocks,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if flags is None:
        return
    count = _tiles_per_program(queries, tiles, _LOOPED_MAX_TILES)
    _index_score_looped_tiles[(queries, triton.cdiv(tiles, count))](
        q,
        w,
        index_cache,
        table,
        visible,
        rows,
        blocks,
        flags,
        *strides,
        index_cache.shape[0],
        table.shape[1],
        width,
        count,
        q.shape[1],
        shard_heads,
        shards,
        max(16, triton.next_power_of_2(q.shape[1])),
        128,
        make_blocks,
        enable_fp_fusion=False,
    )


if (
    _platform.is_nvidia
    and _platform.arch_version == ArchVersion(12, 0)
    and is_graph_safe_topk_available()
):

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="flashinfer_dsv41_index_topk",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
        traits={
            "index_heads": frozenset({32}),
            "index_k_format": frozenset({"mxfp4"}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_dsv41_index_topk(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
        index_metadata,
    ):
        """Implement the family's index_topk contract with bounded score tiles.

        Arguments and returns follow ``dsv41.index_topk``. Scoring retains the
        portable implementation's BF16 dot and per-TP-shard rounding. Native
        selection breaks boundary ties by the smaller logical row or block ID.
        Score storage is limited to 128 MiB per query tile. Metadata remains
        owned by the caller; no scores or cache contents survive this call.
        """
        out = _index_topk_outputs(
            index_q,
            weights,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            candidate_block_size,
            query_chunk_size,
            score_chunk_size,
            process_group,
            out,
        )
        width = (
            page_table.shape[1] * 64
            if candidate_blocks is None
            else candidate_blocks.shape[1] * 8
        )
        if width > _MAX_SCORE_COLUMNS or score_chunk_size < 16:
            # CUB's per-row column bound and the MMA tile minimum are tighter
            # than the family contract. Preserve all arguments on fallback.
            return portable_index_topk(
                index_q,
                weights,
                index_cache,
                page_table,
                visible_lens,
                candidate_blocks,
                topk,
                candidate_topk,
                candidate_block_size,
                query_chunk_size,
                score_chunk_size,
                process_group,
                out,
                index_metadata,
            )
        row_out, row_lens, block_out, block_lens = out
        row_out.fill_(-1)
        row_lens.zero_()
        block_out.fill_(-1)
        block_lens.zero_()
        tokens = index_q.shape[0]
        if not tokens or not page_table.shape[1] or not width:
            return out
        score_columns = width + (width // 8 if candidate_topk else 0)
        chunk = min(
            query_chunk_size, 256, max(1, _SCORE_BUDGET_BYTES // (4 * score_columns))
        )
        tile_bound = min(128, 1 << (score_chunk_size.bit_length() - 1))
        for start in range(0, tokens, chunk):
            end = min(start + chunk, tokens)
            q, w, shards = _index_gather_heads(
                index_q[start:end], weights[start:end], process_group
            )
            # Keep the existing query codec, then consume its packed values
            # directly. Each row contains 64 E2M1 bytes and four UE8M0 scales.
            q = cache_pack(q.reshape(-1, 128), "index", None).view(end - start, 32, 68)
            table = page_table[start:end]
            visible = visible_lens[start:end]
            candidates = None
            if candidate_blocks is not None:
                # Inputs may be unsorted. Sorting makes the native selector's
                # smaller-column tie rule equivalent to smaller logical IDs.
                candidates = candidate_blocks[start:end].sort(dim=-1).values
            # Scores are rounded to the weight dtype, so BF16 weights make
            # 16-bit rows exact; selection reads half the bytes.
            score_dtype = (
                torch.bfloat16
                if _INDEX_GROUPED and w.dtype == torch.bfloat16
                else torch.float32
            )
            rows = torch.empty((end - start, width), dtype=score_dtype, device=q.device)
            blocks = torch.empty(
                (end - start, width // 8 if candidate_topk else 0),
                dtype=score_dtype,
                device=q.device,
            )
            tile = min(64 if end - start == 1 else 128, tile_bound)
            # Full scoring of eight or more queries. Reindex, fewer queries
            # and smaller score tiles keep their kernels.
            grouped = (
                _INDEX_GROUPED
                and candidates is None
                and tile == 128
                and end - start >= 8
            )
            if grouped:
                _score_grouped(
                    q,
                    w,
                    index_cache,
                    table,
                    visible,
                    rows,
                    blocks,
                    index_q.shape[1],
                    shards,
                )
            shared_history = (
                not grouped
                and candidates is None
                and end - start >= 2
                and tile == 128
                and (table.stride(0) == 0 or end - start >= 8)
            )
            shared_flags = None
            if shared_history and table.stride(0) != 0:
                shared_flags = torch.empty(
                    (triton.cdiv(end - start, 2), triton.cdiv(width, 128)),
                    dtype=torch.uint8,
                    device=q.device,
                )
            if shared_history:
                # Two queries reuse one history tile. This tile also keeps
                # the portable score reduction order for FP32 head weights.
                _index_score_shared_history_tiles[
                    (triton.cdiv(end - start, 2), triton.cdiv(width, 128))
                ](
                    q,
                    w,
                    index_cache,
                    table,
                    visible,
                    rows,
                    blocks,
                    shared_flags,
                    end - start,
                    *q.stride(),
                    *w.stride(),
                    *index_cache.stride(),
                    *table.stride(),
                    visible.stride(0),
                    index_cache.shape[0],
                    table.shape[1],
                    width,
                    index_q.shape[1],
                    shards,
                    2,
                    128,
                    bool(candidate_topk),
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
            if not grouped and (not shared_history or shared_flags is not None):
                _index_score_tiles[(end - start, triton.cdiv(width, tile))](
                    q,
                    w,
                    index_cache,
                    table,
                    visible,
                    candidates,
                    rows,
                    blocks,
                    shared_flags,
                    *q.stride(),
                    *w.stride(),
                    *index_cache.stride(),
                    *table.stride(),
                    visible.stride(0),
                    *(candidates.stride() if candidates is not None else (0, 0)),
                    index_cache.shape[0],
                    table.shape[1],
                    width,
                    q.shape[1],
                    index_q.shape[1],
                    shards,
                    max(16, triton.next_power_of_2(q.shape[1])),
                    tile,
                    bool(candidate_topk),
                    candidates is not None,
                    enable_fp_fusion=False,
                )
            values, ids = topk_small_indices(rows, min(topk, width))
            _finish_topk(
                values, ids, row_out[start:end], row_lens[start:end], candidates
            )
            if candidate_topk:
                values, ids = topk_small_indices(
                    blocks, min(candidate_topk, width // 8)
                )
                _finish_topk(values, ids, block_out[start:end], block_lens[start:end])
        return out


# ---------------------------------------------------------------------------
# Paged decode attention on FlashInfer's SM120 mixed-cache kernels (opt-in).
# ---------------------------------------------------------------------------

# "1" registers flashinfer_dsv41_selected_attention_sm120 above the Triton
# SM120 reader; unset or any other value leaves the registry as it was.
SM120_ATTENTION_ENV = "TOKENSPEED_SM120_DSV41_ATTENTION_FLASHINFER"
_SWA_ROW_BYTES = 528
_GLOBAL_ROW_BYTES = 288
# (device index, heads) whose kernels have launched once outside a capture.
_warmed_decode: set[tuple[int, int]] = set()
# Reasons already reported for running the Triton kernel instead.
_reported_fallbacks: set[str] = set()


def _reads_in_place(cache: torch.Tensor, row_bytes: int) -> bool:
    """Whether FlashInfer can address this page-planar field without a copy."""
    return (
        cache.ndim == 3
        and cache.dtype == torch.uint8
        and cache.shape[0] >= 1
        and cache.shape[1:] == (64, row_bytes)
        and cache.stride(2) == 1
        and cache.stride(1) == row_bytes
        and cache.stride(0) >= 64 * row_bytes
        and cache.stride(0) % 16 == 0
        and cache.data_ptr() % 16 == 0
    )


def _selects(slots, lengths, tokens: int) -> bool:
    return (
        slots is not None
        and lengths is not None
        and slots.ndim == 2
        and slots.shape[0] == tokens
        and slots.shape[1] >= 1
        and slots.dtype in (torch.int32, torch.int64)
        and lengths.shape == (tokens,)
        and lengths.dtype in (torch.int32, torch.int64)
    )


def _decode_fallback_reason(
    decode: Dsv41MixedDecode,
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    out,
    schedule,
    lse,
) -> str | None:
    """Name why FlashInfer cannot serve this paged call, or None when it can.

    Only shapes, dtypes, strides and addresses are inspected, so the answer is
    fixed for a captured graph and nothing waits on the device.
    """
    if schedule is None:
        # A schedule marks slots the runtime produced, which are bounded by
        # the field's pages. FlashInfer does not scan slots against the field.
        return "the call carries no decode schedule"
    if lse is not None:
        return "a partial-attention LSE is requested"
    if q.ndim != 3 or q.shape[2] != 512 or q.dtype != torch.bfloat16 or not q.is_cuda:
        return "the query is not CUDA BF16 [tokens, heads, 512]"
    tokens, heads, _ = q.shape
    if tokens == 0:
        return "the call has no query rows"
    if heads not in decode.head_counts:
        return f"{heads} query heads are not one of {sorted(decode.head_counts)}"
    tensors = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
    )
    if any(tensor is not None and tensor.device != q.device for tensor in tensors):
        return "the tensors are on different devices"
    if q.is_contiguous() and q.data_ptr() % 16:
        return "the query is not 16-byte aligned"
    if swa_cache is None or not _reads_in_place(swa_cache, _SWA_ROW_BYTES):
        return (
            "the SWA field is not an aligned page-planar V4.1 FP8 [pages, 64, 528] view"
        )
    if not _selects(swa_slots, swa_lens, tokens):
        return "the SWA selection is not integer [tokens, width >= 1] with [tokens] lengths"
    global_width = 0
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            return "a global selection comes without a global field"
    else:
        if not _reads_in_place(global_cache, _GLOBAL_ROW_BYTES):
            return "the global field is not an aligned page-planar V4.1 FP4 [pages, 64, 288] view"
        if not _selects(global_slots, global_lens, tokens):
            return "the global selection is not integer [tokens, width >= 1] with [tokens] lengths"
        global_width = global_slots.shape[1]
    if dsv41_mixed_decode_chunks(swa_slots.shape[1], global_width) > decode.max_chunks:
        return "the selection is wider than FlashInfer's split plan"
    if attn_sink is not None and (
        attn_sink.ndim != 1
        or attn_sink.shape[0] < heads
        or attn_sink.dtype != torch.float32
        or attn_sink.stride(0) != 1
    ):
        return "the sink is not contiguous FP32 [heads]"
    if out is not None and (
        out.shape != q.shape
        or out.dtype != q.dtype
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        return "out is not an aligned contiguous buffer shaped like the query"
    return None


def _launch_decode(
    decode: Dsv41MixedDecode,
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
):
    """One FlashInfer decode launch, plus its merge launch when the plan splits."""
    tokens, heads, _ = q.shape
    query = q.contiguous()
    indices = swa_slots.to(torch.int32).contiguous()
    lengths = swa_lens.to(torch.int32).contiguous()
    extra_indices = extra_lengths = None
    global_width = 0
    if global_cache is not None:
        extra_indices = global_slots.to(torch.int32).contiguous()
        extra_lengths = global_lens.to(torch.int32).contiguous()
        global_width = extra_indices.shape[1]
    # The split count is a function of shapes and the device only. Pinning it
    # on the launch keeps the kernel's plan and this scratch in agreement.
    splits = dsv41_mixed_decode_splits(
        tokens, heads, indices.shape[1], global_width, True, q.get_device()
    )
    mid_out = mid_lse = None
    if splits > 1:
        # Invocation-owned split partials: under capture they come from the
        # graph's pool and every replay reuses them.
        mid_out = torch.empty(
            (tokens, heads, splits, 512), dtype=torch.bfloat16, device=q.device
        )
        mid_lse = torch.empty(
            (tokens, heads, splits), dtype=torch.float32, device=q.device
        )
    # FlashInfer always writes an LSE; this op does not return it.
    lse = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    output = torch.empty_like(query) if out is None else out
    decode.run(
        query,
        swa_cache,
        indices,
        output,
        lse,
        softmax_scale,
        topk_length=lengths,
        attn_sink=attn_sink,
        extra_kv_cache=global_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        mid_out=mid_out,
        mid_lse=mid_lse,
        num_splits=splits,
        max_splits=MAX_SPLITS,
        enable_pdl=pdl_enabled(),
    )
    return output


def _warm_decode(decode: Dsv41MixedDecode, device: torch.device, heads: int) -> None:
    """Launch every kernel this head count uses once, outside a graph capture.

    The family is prebuilt, so nothing compiles here; the launches make the
    driver load the window-only reader, the window+global reader and the
    split merge (one query row splits both layouts) before a capture records
    them. All inputs are zero rows owned by this call.
    """
    q = torch.zeros((1, heads, 512), dtype=torch.bfloat16, device=device)
    swa = torch.zeros((1, 64, _SWA_ROW_BYTES), dtype=torch.uint8, device=device)
    full = torch.zeros((1, 64, _GLOBAL_ROW_BYTES), dtype=torch.uint8, device=device)
    swa_slots = torch.zeros((1, 128), dtype=torch.int32, device=device)
    global_slots = torch.zeros((1, 512), dtype=torch.int32, device=device)
    length = torch.ones((1,), dtype=torch.int32, device=device)
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    for global_cache in (None, full):
        _launch_decode(
            decode,
            q,
            swa,
            swa_slots,
            length,
            global_cache,
            None if global_cache is None else global_slots,
            None if global_cache is None else length,
            sink,
            1.0,
            None,
        )


def flashinfer_dsv41_selected_attention_sm120(
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
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
    *,
    lse=None,
):
    """Paged V4.1 decode attention reading the packed fields in place.

    Arguments and returns follow ``dsv41.selected_attention``. One launch
    covers every query row, plus a merge launch when FlashInfer's plan splits
    a row's candidates across CTAs; ``query_chunk_size`` does not apply.
    Negative slots and slots past a row's length are masked on device.
    Nonnegative slots must address their field: they are not scanned, which
    is why only calls that carry a ``schedule`` (the runtime's paged decode,
    whose slot producers bound every slot by the field's pages) are served.

    Arithmetic is BF16 Q/K/V with FP32 accumulation and softmax, as in the
    Triton reader, except that probabilities are rounded to BF16 before PV,
    so outputs agree to BF16 rounding, not bit for bit.

    Everything else runs ``triton_dsv41_selected_attention_sm120`` with the
    same arguments: compact workspaces (prefill, the DSpark window), calls
    without a schedule, a requested LSE, head counts or cache views the
    family does not read, and a first use that falls inside a graph capture.
    Each reason other than a workspace is logged once.
    """
    reason = None
    workspace = prefill_kv is not None or prefill_indices is not None
    if not workspace:
        decode = load_dsv41_mixed_decode()
        reason = _decode_fallback_reason(
            decode,
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            out,
            schedule,
            lse,
        )
        if reason is None:
            key = (q.get_device(), q.shape[1])
            if key not in _warmed_decode:
                if torch.cuda.is_current_stream_capturing():
                    reason = (
                        "FlashInfer's kernels were first requested inside a "
                        "graph capture, before any eager decode launched them"
                    )
                else:
                    _warm_decode(decode, q.device, q.shape[1])
                    _warmed_decode.add(key)
                    logger.info(
                        "V4.1 SM120 decode attention: FlashInfer kernels launched "
                        f"on {q.device} for {q.shape[1]} query heads"
                    )
    if workspace or reason is not None:
        if reason is not None and reason not in _reported_fallbacks:
            _reported_fallbacks.add(reason)
            logger.warning(
                f"V4.1 SM120 decode attention runs the Triton kernel: {reason}"
            )
        return triton_dsv41_selected_attention_sm120(
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
            query_chunk_size,
            schedule,
            prefill_kv,
            prefill_indices,
            lse=lse,
        )
    return _launch_decode(
        decode,
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        None if attn_sink is None else attn_sink[: q.shape[1]],
        softmax_scale,
        out,
    )


def _register_sm120_selected_attention() -> None:
    """Register the FlashInfer reader when the switch asks for it.

    The family is built or loaded here, at import, so no request and no graph
    capture ever waits for nvcc. Without the switch, off SM120, or when the
    family is missing or fails to build, nothing is registered and selection
    stays on ``triton_dsv41_selected_attention_sm120``.
    """
    if os.environ.get(SM120_ATTENTION_ENV, "0") != "1":
        return
    if not (_platform.is_nvidia and _platform.arch_version == ArchVersion(12, 0)):
        return
    if not is_dsv41_mixed_decode_available():
        logger.warning(
            f"{SM120_ATTENTION_ENV}=1 but FlashInfer does not ship the V4.1 "
            "mixed-cache decode family; paged decode attention stays on Triton"
        )
        return
    try:
        decode = load_dsv41_mixed_decode()
    except Exception as error:
        logger.warning(
            f"{SM120_ATTENTION_ENV}=1 but FlashInfer's V4.1 mixed-cache decode "
            f"family failed to load ({error!r}); paged decode attention stays "
            "on Triton"
        )
        return
    register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="flashinfer_dsv41_selected_attention_sm120",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
        # Above the Triton SM120 reader (PERFORMANT), which this kernel calls
        # for everything it does not serve.
        priority=Priority.SPECIALIZED,
    )(flashinfer_dsv41_selected_attention_sm120)
    logger.info(
        f"{SM120_ATTENTION_ENV}=1: V4.1 paged decode attention uses FlashInfer's "
        f"SM120 mixed-cache kernels (heads {sorted(decode.head_counts)})"
    )


_register_sm120_selected_attention()
