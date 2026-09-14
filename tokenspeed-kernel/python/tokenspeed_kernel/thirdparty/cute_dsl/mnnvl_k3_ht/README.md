# Native-width K3 HT specialization

This directory adapts FlashInfer 0.6.18's HT protocol/device implementation.
The upstream design is described in
[FlashInfer PR #4358](https://github.com/flashinfer-ai/flashinfer/pull/4358).
Original Apache-2.0 notices are retained. The BT kernel is not vendored or
modified here; the optional FlashInfer adapter calls it directly.

The K3-specific device change predicates the partial reduction warp for latent
width 3584: a TP8 shard contains 56 BF16x8 vectors rather than a whole number of
warps. Runtime explicitly supplies its 10-stage HT preset; this is independent
of the upstream default token thresholds and the protocol's fallback tuning.
Do not update the optional dependency or reuse upstream benchmark numbers
without validating this application's shapes, routing and numerical contract.
