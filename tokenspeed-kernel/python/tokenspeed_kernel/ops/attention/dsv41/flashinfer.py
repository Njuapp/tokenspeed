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

"""SM120 paged index scoring with exact native TopK selection."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.dsv41._triton.index_score import (
    _index_score_shared_history_tiles,
    _index_score_tiles,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _finish_topk,
    _index_gather_heads,
    _index_topk_outputs,
    cache_pack,
    index_topk as portable_index_topk,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.topk import (
    is_graph_safe_topk_available,
    topk_small_indices,
)

_SCORE_BUDGET_BYTES = 128 * 1024 * 1024
_MAX_SCORE_COLUMNS = 1 << 21
_platform = current_platform()


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
            rows = torch.empty(
                (end - start, width), dtype=torch.float32, device=q.device
            )
            blocks = torch.empty(
                (end - start, width // 8 if candidate_topk else 0),
                dtype=torch.float32,
                device=q.device,
            )
            tile = min(64 if end - start == 1 else 128, tile_bound)
            shared_history = (
                candidates is None
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
            if not shared_history or shared_flags is not None:
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
