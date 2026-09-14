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

"""Pure, rank-uniform routing policy for the Kimi-K3 MoE tail.

No device probing or optional kernel imports belong here. The communication
layer negotiates capabilities, owns storage, and supplies the resulting booleans.
"""

from __future__ import annotations

from enum import IntEnum


class K3MoETailTier(IntEnum):
    """How the K3 MoE tail combines routed/shared partials, best first.

    The first four entries retain the original selector's priority. The two
    deferred entries are opt-in overrides applied by ``select_bt_ht_first_half``.
    Values never escape the process (identity comparisons, no serialization).
    """

    TAIL_FUSION = 0  # fused decode kernel (aka the multicast latent tail)
    MULTIMEM_AR = 1  # in-switch (ld_reduce) reduces, then the replicated tail
    FUSED_LANE_AR = 2  # join tier: lane one-shot / cat+one-shot / grouped NCCL
    SEPARATE_REDUCE = 3  # portable: reduce each partial on its own
    MNNVL_BT_DEFERRED = 4  # BT first half; original second-half reduction
    MNNVL_HT_DEFERRED = 5  # HT first half; original second-half reduction


# Measured profit edge of the fused tail; the kernel's own capacity is larger.
TAIL_FUSION_MAX_TOKENS = 32

MULTIMEM_AR_MIN_TOKENS = 256
# Upper edge of the measured window; larger batches take the join's grouped path.
MULTIMEM_AR_MAX_TOKENS = 8192

# Continuous policy, not a whitelist of CUDA-graph buckets.
MNNVL_BT_MAX_TOKENS = 1024


def select_k3_moe_tail_tier(
    *,
    num_tokens: int,
    graph_phase: bool,
    tail_fusion_max_tokens: int,
    fused_moe_ar: bool,
    multimem_ok: bool,
    is_decode: bool = False,
    join_moe_reduce: bool = False,
) -> K3MoETailTier:
    """Pick the tail tier; every input must be rank-uniform.

    Args:
        num_tokens: Tokens in this forward (identical on every rank).
        graph_phase: Whether the forward runs under the CUDA-graph phase.
        tail_fusion_max_tokens: Largest token count the fused tail is both
            able and worth running at, 0 when absent.
        fused_moe_ar: Whether the fused-AR execution plan is armed (implies a
            backend-owned lane, so TRT-LLM only).
        join_moe_reduce: Whether the routed and shared partials can be reduced
            together without a lane, via a concatenated one-shot or a grouped
            all-reduce. Portable, so this is what lets non-TRT-LLM backends
            reach the join tier.
        multimem_ok: Collectively-agreed multimem availability.
        is_decode: Whether this forward is a decode (spec-verify included);
            rank-uniform and stable between graph capture and replay.

    Returns:
        The best applicable ``K3MoETailTier``.
    """
    # Tested first, so a fused-tail capacity that ever reached into the
    # multimem window would still resolve here rather than overlap.
    if graph_phase and 1 <= num_tokens <= tail_fusion_max_tokens:
        return K3MoETailTier.TAIL_FUSION
    if not fused_moe_ar:
        # No lane, but the join only needs a concatenated or grouped
        # all-reduce. Taking it halves the tail's collectives -- SEPARATE_REDUCE
        # reduces the routed and shared partials over the same group one after
        # the other -- and each collective is a rendezvous whose cost does not
        # amortize with batch, so the saving is largest at low concurrency.
        #
        # SEPARATE_REDUCE stays the fallback for layouts that cannot join,
        # which includes a sharded up projection: its tail folds the projection
        # between two sequential all-reduces instead of calling
        # kimi3_join_reduce_moe, so it would save no collective while still
        # giving up the routed_in_fork overlap.
        #
        # MULTIMEM_AR is deliberately still not reachable here. It already
        # required fused_moe_ar before this join existed, so promoting
        # lane-less backends into the multimem window would be a separate
        # behavioural change rather than part of this one.
        if join_moe_reduce:
            return K3MoETailTier.FUSED_LANE_AR
        return K3MoETailTier.SEPARATE_REDUCE
    if (
        multimem_ok
        # Decode buckets skip multimem: same bytes, but it leaves the GPU idle there.
        and not is_decode
        and MULTIMEM_AR_MIN_TOKENS <= num_tokens <= MULTIMEM_AR_MAX_TOKENS
    ):
        return K3MoETailTier.MULTIMEM_AR
    return K3MoETailTier.FUSED_LANE_AR


def select_bt_ht_first_half(
    *,
    original: K3MoETailTier,
    num_tokens: int,
    enabled: bool,
    bt_ok: bool,
    ht_ok: bool,
) -> K3MoETailTier:
    """Override only finalize/AR1/RMSNorm after the original selector.

    Args:
        original: Tier chosen by the original main policy.
        num_tokens: Rank-uniform actual/padded token count.
        enabled: Collectively agreed first-half opt-in and eligible plan.
        bt_ok: Prepared BT workspace supports this token count.
        ht_ok: Prepared HT workspace supports this token count.

    Returns:
        Original small/fallback tier, or BT/HT with the original second half.
    """
    if original is K3MoETailTier.TAIL_FUSION or not enabled:
        return original
    if TAIL_FUSION_MAX_TOKENS < num_tokens <= MNNVL_BT_MAX_TOKENS and bt_ok:
        return K3MoETailTier.MNNVL_BT_DEFERRED
    if MNNVL_BT_MAX_TOKENS < num_tokens <= MULTIMEM_AR_MAX_TOKENS and ht_ok:
        return K3MoETailTier.MNNVL_HT_DEFERRED
    return original
