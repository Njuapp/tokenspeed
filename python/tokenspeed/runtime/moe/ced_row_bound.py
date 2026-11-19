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

"""Row bound of the fused MoE launches that follow a CED narrowing.

A fused expert-parallel MoE launch takes a rank-uniform bound on the rows any
rank holds (``max_num_tokens_per_gpu``): the kernel picks its row schedule and
sizes its per-launch setup from it, so every EP rank must pass the same value
and it must not be below any rank's rows. Under attention data parallelism the
layers size it from the forward's per-rank token counts, which the event loop
gathers before the forward. Behind the causal encoder-decoder narrowing a rank
holds only the last window of the prompts that end in the chunk, far fewer
rows, but which prompts end on a rank is not part of the gathered metadata.

The gathered metadata still bounds the narrowed rows: a rank keeps at most one
window per request of its batch and never more rows than it forwards. That
bound is the same on every rank, needs no exchange, and lets the kernel run the
schedule built for small launches.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

# Opt-in experiment switch, off by default: "1" makes the layers behind the CED
# narrowing of a pure-prefill attention-DP forward bound their fused MegaMoE
# launch by ``ced_decoder_row_bound`` instead of the forward's token counts.
# The kernel's output does not depend on the bound, only its schedule does.
# Unset or any other value keeps the token-count bound.
MEGA_MOE_DECODER_ROW_BOUND_ENV = "TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND"

# Read once, at import: the EP ranks of a launch must agree on its bound, so
# the choice cannot change while a process serves.
_REQUESTED = os.environ.get(MEGA_MOE_DECODER_ROW_BOUND_ENV, "") == "1"


def mega_moe_decoder_row_bound_requested() -> bool:
    """Whether the process was started with the narrowed row bound switched on."""
    return _REQUESTED


def ced_decoder_row_bound(
    global_num_tokens: Sequence[int],
    global_batch_size: Sequence[int],
    rows_per_request: int,
) -> int:
    """Most rows any rank can hold behind the CED narrowing of a pure-prefill forward.

    The narrowing keeps, per prefill request of a rank's batch, at most the
    prompt's last window (``rows_per_request`` rows; a chunk that does not end
    its prompt keeps one row or none), and only rows the rank forwards.

    Args:
        global_num_tokens: Tokens each rank forwards, as the event loop
            gathered them (``ForwardContext.global_num_tokens``).
        global_batch_size: Requests in each rank's batch, same index
            (``ForwardContext.global_bs``). Every request must be a prefill
            request: decode rows are not narrowed.
        rows_per_request: The model's ``max_decoder_rows_per_request``.

    Returns:
        A bound that is identical on every rank that holds the same metadata.
    """
    if rows_per_request < 1:
        raise ValueError(f"rows_per_request must be positive, got {rows_per_request}")
    if len(global_num_tokens) != len(global_batch_size):
        raise ValueError(
            f"{len(global_num_tokens)} token counts for {len(global_batch_size)} "
            "batch sizes"
        )
    return max(
        (
            min(int(tokens), int(requests) * rows_per_request)
            for tokens, requests in zip(global_num_tokens, global_batch_size)
        ),
        default=0,
    )
