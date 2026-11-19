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

"""Optional graph-safe TopK with an explicit smaller-index boundary tie rule."""

from __future__ import annotations

import inspect

import torch


def is_graph_safe_topk_available() -> bool:
    """Whether FlashInfer exposes graph-safe selection and index tie breaking."""
    try:
        from flashinfer.topk import TopKTieBreak, top_k
    except ImportError:
        return False
    parameters = inspect.signature(top_k).parameters
    return (
        "dsa_graph_safe" in parameters
        and "tie_break" in parameters
        and TopKTieBreak.SMALL == 1
    )


def topk_small_indices(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select values and column indices, preferring smaller indices at ties.

    Args:
        scores: Contiguous FP32 rows with at most 2**21 columns.
        k: Number of columns to select, between one and the row width.

    Returns:
        FP32 values and int64 indices. Their order is unspecified. A caller
        that needs deterministic position order must sort the selected IDs.
    """
    from flashinfer.topk import TopKTieBreak, top_k

    return top_k(
        scores,
        k,
        sorted=False,
        deterministic=False,
        tie_break=TopKTieBreak.SMALL,
        dsa_graph_safe=True,
    )
