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

"""FlashInfer's SM120 DeepSeek-V4.1 mixed-cache sparse-MLA decode family.

The family reads V4.1 page-planar caches in place: the 528-byte FP8 SWA rows
(512 E4M3 bytes per row, then the page's 16 UE8M0 scale bytes per row) and the
288-byte FP4 global rows (256 E2M1 bytes per row, then 32 E4M3 scale bytes per
row). It is generated CUDA built once by FlashInfer's JIT; nothing is compiled
per shape.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import torch

# FlashInfer's bound on the split-K CTAs of one head block.
MAX_SPLITS = 16


class Dsv41MixedDecode(NamedTuple):
    """Entry points of the loaded decode family."""

    # cake_sparse_mla_sm120_dsv41_mixed_decode: allocation-free launch.
    run: Callable
    # Query head counts the generated kernels serve.
    head_counts: frozenset[int]
    # Largest candidate-chunk count a row may have (index table x splits).
    max_chunks: int


def is_dsv41_mixed_decode_available() -> bool:
    """Whether FlashInfer ships the generated family; nothing is built or loaded."""
    try:
        from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_available,
        )
    except ImportError:
        return False
    return cake_sparse_mla_sm120_dsv41_mixed_available()


@functools.cache
def load_dsv41_mixed_decode() -> Dsv41MixedDecode:
    """Build (first use) or load the family and return its entry points.

    The first call compiles the generated translation units with nvcc unless
    FlashInfer's JIT cache already holds them, so it belongs to startup, never
    to a request or a graph capture. Raises whatever FlashInfer raises when
    the family is absent or the build fails.
    """
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as family

    family.get_cake_sparse_mla_sm120_dsv41_mixed_module()
    geometry = family.kernel_geometry()
    return Dsv41MixedDecode(
        run=family.cake_sparse_mla_sm120_dsv41_mixed_decode,
        head_counts=frozenset(geometry.head_counts),
        max_chunks=geometry.max_chunks_per_block * MAX_SPLITS,
    )


def dsv41_mixed_decode_chunks(swa_width: int, global_width: int) -> int:
    """Candidate chunks of one query row: the upper bound on its split count."""
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as family

    return family.cake_sparse_mla_sm120_dsv41_mixed_num_chunks(swa_width, global_width)


@functools.lru_cache(maxsize=None)
def _device_facts(device_index: int) -> tuple[int, bool]:
    props = torch.cuda.get_device_properties(device_index)
    # Compute capability 12.1 is the unified-memory part FlashInfer plans apart.
    return int(props.multi_processor_count), (props.major, props.minor) == (12, 1)


@functools.lru_cache(maxsize=None)
def dsv41_mixed_decode_splits(
    tokens: int,
    heads: int,
    swa_width: int,
    global_width: int,
    ragged: bool,
    device_index: int,
) -> int:
    """Split-K CTAs per (query row, head block) FlashInfer plans for a call.

    A pure function of the call's shapes and the device, so a caller can size
    the split scratch and pin the same count on the launch. One split writes
    the output directly; more need ``[tokens, heads, splits, 512]`` BF16 and
    ``[tokens, heads, splits]`` FP32 scratch and add a merge launch.

    Args:
        tokens: Query rows.
        heads: Query heads, one of the family's head counts.
        swa_width: Columns of the SWA selection.
        global_width: Columns of the global selection, 0 without a global field.
        ragged: Whether per-row lengths are passed to the launch.
        device_index: CUDA device index of the call.
    """
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as family

    num_sms, unified_memory = _device_facts(device_index)
    tiles = family.cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(
        num_tokens=tokens,
        num_heads=heads,
        topk=swa_width,
        extra_topk=global_width,
        num_sms=num_sms,
    )
    splits, _ = family.cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
        num_tokens=tokens,
        num_heads=heads,
        topk=swa_width,
        extra_topk=global_width,
        num_sms=num_sms,
        max_splits=MAX_SPLITS,
        head_tiles=tiles,
        unified_memory=unified_memory,
        ragged=ragged,
    )
    return int(splits)
