# DeepSeek-V4.1-Flash on RTX 6000D: optimization branch

Branch `opt/6000d-dsv41`, two commits on top of upstream `ee5b468f1`:

1. the SM120 code as measured at handoff (the baseline of every number here);
2. the optimizations. Each one is an environment switch that is off by default; with all switches off the forward
   path equals commit 1.

## Start here

| File | What it is |
| --- | --- |
| `docs/PROGRESS_SUMMARY.md` | Conclusions, speedups ranked by gain, gaps, where the code is (Chinese) |
| `docs/OPTIMIZATION_RESULTS.md` | Full gold-standard results, change list with kernel-level validation, audit notes (Chinese) |
| `docs/SOL_ANALYSIS.md`, `docs/OPTIMIZATION_GOAL.md` | Per-kernel speed-of-light analysis and the goal the work followed |
| `switch-sets/` | The switch combination of each tier, as sourced by the engine launch |
| `launch/` | The engine launch scripts behind every number, and the settings a benchmark of this branch must keep |
| `placement/` | How to solve a static expert placement (with or without replicas) for a workload |
| `docs/EP16_SPLIT_RESULTS.md` | The second MegaMoE backend on two nodes: kernel tuning on two route distributions, packed dispatch, compile buckets, real-text and step-time results with a static placement (Chinese) |
| `flashinfer/` | The FlashInfer patch the two-node readings of the second backend need, and what it changes |
| `tuning/` | The kernel tuner (balanced and power-law routes) and the kernel-level timing scripts |
| `realtext/` | The real-text decode measurement: prompt bank, client, how a run goes |
| `results/ep16-split-20261126/` | Raw files of those results |
| `docs/EP8_VS_EP16.md` | One node (EP8) against two (EP16) at the same load per GPU: both backends, real text with static placements, balanced routes (Chinese) |
| `docs/LOAD_PROFILE_FILLER_ROWS.md` | The load profile behind static placements counted the filler rows of padded CUDA graph replays: the flaw, the fix, how to check a profile, the effect on placements and on throughput (Chinese) |
| `results/ep8-vs-ep16-20261127/` | Raw files of those two documents |
| `results/timelines/` | Per-kernel time and launch counts per step from Nsight Systems captures, before and after |

## Results (gold standards of the handoff, two nodes of 8x RTX 6000D)

| Gold standard | Baseline | Bit-identical | Same precision | Changes precision |
| --- | ---: | ---: | ---: | ---: |
| Prefill ADP8 (req/s) | 5.688 | 7.341 (+29%) | 8.991 (+58%) | 9.755 (+72%) |
| Prefill QCP8 (req/s) | 4.596 | — | 7.168 (+56%) | — |
| Decode D16, handoff settings (tok/s per GPU, mean TPOT < 10 ms) | 1108.6 | 1203.0 (+8.5%) | 1425.2 (+28.6%) | 1488.1 (+34.2%) |
| Decode D16, larger KV pool | — | same | 1451.9 (+31.0%) | 1540.2 (+38.9%) |

The gold standards force the acceptance length and store no outputs, so they show speed only. Numerical claims
(bit-identical, same precision) come from kernel-level tests against bitwise or FP64 references.

## Switch sets

| Use | Bit-identical | Same precision | Changes precision |
| --- | --- | --- | --- |
| Prefill ADP8 | `p7-bit.sh` | `p7-same.sh` | `p7-prec.sh`, `p8-prec1.sh` (adds single-pass BF16 PV) |
| Prefill QCP8 | — | `q7-same.sh` | — |
| Decode D16 | `d7-bit.sh` | `d7-same.sh` | `d7-prec.sh` |

The two `-overlap` files additionally drop `--disable-overlap-schedule`, which is outside the handoff settings.
The expert placement file named in the prefill sets is solved from the routing statistics of the gold-standard
workload and is not in this branch; another workload needs its own (`placement/README.md`: the load profile of
`python/tokenspeed/runtime/moe/expert_file_placement.py` and the solver `placement/solve_placement.py`).

## A second MegaMoE backend for decode

`--moe-backend mega_moe_split` runs the MoE layers on FlashInfer's CuTe DSL split kernels instead of its
persistent kernel. It is opt-in, changes nothing for `mega_moe`, and is faster or slower depending on the load,
the routing of the traffic and the expert placement. On two nodes (EP16), both backends with a static placement
solved for the workload (real text 2026-11-27, synthetic tokens 2026-11-26):

- Real text (8192-token prompts, real DSpark acceptance; permutation placement solved from a load profile that
  counts real rows only): the split backend gives 17.6% / 20.9% / 16.4% / 8.5% / 6.6% more output at 1 / 2 / 4 /
  8 / 12 requests per GPU, the same at 16 (+0.6%) and 6.7% less at 24. Replicas bring nothing on top of that
  permutation.
- Synthetic tokens of the step-time measurement, whose routes are far more concentrated (six engines under
  identical conditions, placements solved from held-out inputs): without a placement the split backend's decode
  step is 15% / 9% shorter at 1 / 2 requests per GPU, about equal at 4 and 16% to 34% longer from 8; with a
  permutation it is 17% / 15% / 10% shorter at 1 / 2 / 4 and 4% to 31% longer from 8; with 4 replica slots per
  rank it is 18% / 16% / 13% / 3% shorter up to 8 and 6% / 17% longer at 16 / 32.

So the backend is a choice per deployment: the split kernels for low concurrency per GPU, the persistent kernel
for high. The reason is the routing: on balanced routes the split kernels are faster up to about 144 rows per
rank, on power-law routes only up to 16, and no kernel configuration changes that. The branch carries what the
split backend needs at EP16: a kernel table (tuned on balanced and on power-law routes), and, as a patch for
FlashInfer, the packed dispatch between NUMA nodes and hosts (6.9% to 19.5% more output on real text from 4
requests per GPU) and compile buckets at the row counts of DSpark decode steps. `launch/README.md` has the
launch and what to know before switching, `docs/EP16_SPLIT_RESULTS.md` the full results.

## One node (EP8) against two (EP16)

At the same number of requests per GPU, an EP16 group over two nodes gives more output per GPU than an EP8 group
on one node (2026-11-27, real text, each layout with a static placement solved from its own profile, the same
code, FlashInfer, switch set and prompts). The persistent kernel gives the same at 1 request per GPU (+1% to
+2%), 7% to 9% more at 2 and 14.3% to 21.5% more from 4 to 24. The split kernels, with a kernel table and the packed
dispatch on both sides, give 5.4% to 6.6% more at 1, 13% to 22% more from 2 to 16 and 6.5% to 9.2% more at 24.
Two nodes as one EP16 group give 2.0 to 2.5 times the output of one node as EP8. On balanced routes (synthetic
tokens) the EP16 decode step is 12% to 24% shorter on the persistent kernel and 18% to 30% on the split kernels
up to 16 requests per GPU, and 2% to 6% at 32. The likely reason, not measured separately: at low load a MoE
launch is bound by reading the weights of every expert that receives rows, and a GPU holds 48 experts at EP8 and
24 at EP16. `docs/EP8_VS_EP16.md` has the conditions, the tables and the two settings to check before comparing
split runs across EP sizes (the kernel table is selected by the exact number of expert slots; the packed
dispatch is on by default at EP16 only).

## Load profiles count real rows only

Until 2026-11-27 the load profile (`TOKENSPEED_MOE_LOAD_PROFILE`) counted the filler rows that pad CUDA graph
replays under attention DP as traffic: 30% of the decode rows of the two-node real-text profile, 17% on one
node, 2% to 4% in the step-time measurements. Filler rows route to the same few experts, so a placement solved
from such a profile spreads those and hardly balances the real traffic (busiest rank over the mean on real
traffic at EP16: 1.60 without a placement, 1.45 with the old permutation, 1.00 with the new one). The profile
now counts a rank's real rows only, `placement/check_live_rows.py` checks a profile against the engine logs, and
the real-text results above use placements solved again. In the same session, with only the placement file
changed, the new placement gives the two-node split engine the same output at 1 to 4 requests per GPU and 3.2% /
4.3% / 5.4% / 8.0% more at 8 / 12 / 16 / 24 with 4 replica slots per rank, and 3.9% to 9.4% more at every level
with a permutation only. `docs/LOAD_PROFILE_FILLER_ROWS.md` has the details and which earlier numbers are
affected. Prefill profiles are not affected (no graph replay), the synthetic-token step-time placements hardly.

## Launch requirements

The numbers need the launch they were measured with; `launch/README.md` has the scripts and the details.

- **PDL on.** Do not pass `--disable-pdl`: with it a decode step is 11 to 14 ms slower (29.3 ms becomes 40.3 ms at
  one request per rank) and a prefill step 12.6% slower.
- **A switch set sourced before the engine starts.** All switches are off by default, so without one the engine
  runs the baseline.

## Not covered

Task-level output quality and real acceptance rate (most relevant for the "changes precision" tier), and
end-to-end prefill-decode serving.
