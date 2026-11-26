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
solved for the workload (2026-11-26):

- Real text (8192-token prompts, real DSpark acceptance): the split backend gives 15.6% / 16.6% / 14.5% / 5.6%
  more output at 1 / 2 / 4 / 8 requests per GPU and 4.2% / 9.8% less at 16 / 24.
- Synthetic tokens of the step-time measurement, whose routes are far more concentrated: with a permutation the
  split backend's decode step is 17% / 15% / 11% shorter at 1 / 2 / 4 requests per GPU and 3% to 30% longer from 8;
  with 4 replica slots per rank it is 18% / 16% / 13% / 3% shorter up to 8 and 5% / 16% longer at 16 / 32.

So the backend is a choice per deployment: the split kernels for low concurrency per GPU, the persistent kernel
for high. The reason is the routing: on balanced routes the split kernels are faster up to about 144 rows per
rank, on power-law routes only up to 16, and no kernel configuration changes that. The branch carries what the
split backend needs at EP16: a kernel table (tuned on balanced and on power-law routes), and, as a patch for
FlashInfer, the packed dispatch between NUMA nodes and hosts (6.9% to 19.5% more output on real text from 4
requests per GPU) and compile buckets at the row counts of DSpark decode steps. `launch/README.md` has the
launch and what to know before switching, `docs/EP16_SPLIT_RESULTS.md` the full results.

## Launch requirements

The numbers need the launch they were measured with; `launch/README.md` has the scripts and the details.

- **PDL on.** Do not pass `--disable-pdl`: with it a decode step is 11 to 14 ms slower (29.3 ms becomes 40.3 ms at
  one request per rank) and a prefill step 12.6% slower.
- **A switch set sourced before the engine starts.** All switches are off by default, so without one the engine
  runs the baseline.

## Not covered

Task-level output quality and real acceptance rate (most relevant for the "changes precision" tier), and
end-to-end prefill-decode serving.
