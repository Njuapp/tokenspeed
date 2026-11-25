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

A later commit adds `--moe-backend mega_moe_split`: the MoE layers on FlashInfer's CuTe DSL split kernels instead
of its persistent kernel. It is opt-in, changes nothing for `mega_moe`, and is faster or slower depending on the
load, the routing of the traffic and the expert placement (first readings on one node, EP8: 16% more output on
real text at 2 requests per GPU and 5% more at 8; on synthetic tokens without an expert placement, whose routes
are far more concentrated, 8% to 9% shorter decode steps at 1 to 4 requests per GPU and longer steps from 8).
For two nodes (EP16) the branch carries a tuned kernel table for the split kernels: against FlashInfer's
rule-based selection their decode step is 3% to 11% shorter (synthetic tokens, 1 to 32 requests per GPU). Against
`mega_moe` the split backend is then 13% and 6% shorter at 1 and 2 requests per GPU and 6% to 51% longer from 4.
`launch/README.md` has the launch, the FlashInfer source it needs and what to know before switching.

## Launch requirements

The numbers need the launch they were measured with; `launch/README.md` has the scripts and the details.

- **PDL on.** Do not pass `--disable-pdl`: with it a decode step is 11 to 14 ms slower (29.3 ms becomes 40.3 ms at
  one request per rank) and a prefill step 12.6% slower.
- **A switch set sourced before the engine starts.** All switches are off by default, so without one the engine
  runs the baseline.

## Not covered

Task-level output quality and real acceptance rate (most relevant for the "changes precision" tier), and
end-to-end prefill-decode serving.
