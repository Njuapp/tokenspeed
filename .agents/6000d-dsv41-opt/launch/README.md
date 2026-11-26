# Engine launch settings

The engine launches behind every number of this branch. The scripts are the ones that were run, with the site
values (addresses, interface, paths) turned into variables and the hooks of our benchmark clients removed; with
our values filled in, each prints the same `ts serve` command line as the recorded runs.

Read this before benchmarking the branch under another harness: a different launch can cost more than the
optimizations gain.

## Two settings that decide the result

1. **Keep PDL on.** Programmatic dependent launch is on by default; do not pass `--disable-pdl`. The same code,
   same node, same day, with that one argument added (measured as described under "Step-time measurements"):

   | Decode step (ms), one node, attention DP8, MoE EP8, `d7-same.sh` | 1 request per rank | 8 | 32 |
   | --- | ---: | ---: | ---: |
   | PDL on | 29.32 | 40.39 | 66.83 |
   | `--disable-pdl` | 40.26 | 51.71 | 81.26 |

   A prefill step of 8 requests (attention DP8, 4096 new tokens after 126976 cached ones per request) takes
   630.8 ms with PDL on and 710.6 ms with `--disable-pdl` (+12.6%; mean of 10 steps, one run each).

2. **Source a switch set.** Every optimization is an environment switch that is off by default. An engine started
   without one of `../switch-sets/*.sh` runs the baseline (the first commit). The scripts take the file as
   `OPT_EXTRA_ENV`.

## Files

| File | Launch |
| --- | --- |
| `env.sh` | Common environment: code from source on `PYTHONPATH`, GPU order, caches |
| `adp-engine.sh` | Prefill gold standard ADP8 (one node, attention DP8, MoE EP8) |
| `qcp-engine.sh` | Prefill gold standard QCP8 (one node, one request over 8 GPUs, MoE EP8) |
| `decode-engine.sh` | Decode gold standard D16 (two nodes, attention DP16, MoE EP16), or one node |
| `decode-step-engine.sh` | Decode step time at a fixed number of requests per rank (1 to 32) |
| `realtext-engine.sh` | Decode engine of the real-text measurement (`../realtext/`): HTTP gateway, real acceptance |

The three decode scripts take `OPT_MOE_BACKEND` (`mega_moe`, the default, or `mega_moe_split`); see
"Two MegaMoE backends for decode".

Site variables: `OPT_TREE`, `OPT_VENV`, `OPT_FLASHINFER_SRC`, `OPT_MODEL`, and for the prefill scripts
`OPT_HOST_IP` and `OPT_NET_IF`; each script's header lists them. For example:

```bash
SETS=../switch-sets
OPT_EXTRA_ENV=$SETS/p7-same.sh bash adp-engine.sh hot "$OUT"
OPT_EXTRA_ENV=$SETS/q7-same.sh bash qcp-engine.sh hot "$OUT"
# once per node; NODE is 0 or 1
OPT_EXTRA_ENV=$SETS/d7-same.sh MEGA_MOE_TRANSPORT=all_gin \
    bash decode-engine.sh dep16 "$NODE" "$SELF_IP" "$MASTER_IP" "$NET_IF" dspark "$OUT"
OPT_EXTRA_ENV=$SETS/d7-same.sh \
    bash decode-step-engine.sh dep16 "$NODE" "$SELF_IP" "$MASTER_IP" "$NET_IF" 32 "$OUT"
```

The "larger KV pool" row of the results adds `OPT_EXTRA_ARGS="--max-total-tokens 393216"`. The two `-overlap`
switch sets export `OPT_DROP_ARGS=--disable-overlap-schedule`, which `adp-engine.sh` removes from its command.
The clients and workloads of the gold standards are not in this branch.

## Settings common to every launch

| Setting | Value | Note |
| --- | --- | --- |
| PDL | on (the default) | See above |
| Switch set | one of `../switch-sets/*.sh` | Unset = baseline |
| MoE | `--moe-backend mega_moe --mega-moe-transport all_gin` | The fused MegaMoE path the optimizations build on; decode can run `mega_moe_split` instead, see below |
| MegaMoE row capacity | `--deepseek-v4-mega-moe-max-num-tokens 2048` | At least the rows one rank computes per step (4352 for the prefill step below) |
| Engram table | `--engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm` | |
| GPU memory | `--gpu-memory-utilization 0.98` | |
| mHC and MoE top-k kernels | four `TOKENSPEED_KERNEL_OVERRIDE_*=triton` variables | Set in every run; see below |
| Fused top-k/top-p sampling | `TS_DISABLE_FUSED_TOPK_TOPP=1` | Required on our nodes; see below |

**Kernel overrides.** Leaving the four variables unset and adding `--force-deterministic-rsag`, measured together
and on top of `--disable-pdl`, read 40.86 / 51.05 / 81.44 ms against the 40.26 / 51.71 / 81.26 ms above: no
difference in decode. Not measured for prefill.

**Fused top-k/top-p.** Without `TS_DISABLE_FUSED_TOPK_TOPP=1` an engine that uses the default sampling backend
does not finish starting on our nodes: the startup check of the sampling backend spins in
`fused_topk_topp_workspace_size` of the prebuilt library. The cause is not investigated. Where that kernel works,
the variable can stay unset; its effect on step time is not measured.

## Two MegaMoE backends for decode

The MoE layers of the target model can run on either of two FlashInfer kernels for the same checkpoint:

| `--moe-backend` | Kernel | Shared expert |
| --- | --- | --- |
| `mega_moe` (default) | FlashInfer's persistent kernel: dispatch, both GEMMs, return and combine in one launch | inside the kernel with `TOKENSPEED_MEGA_MOE_FUSED_SHARED=1` |
| `mega_moe_split` | FlashInfer's CuTe DSL split kernels: dispatch + FC1 and FC2 + return at the same time on two partitions of the GPU, then the top-k reduce | on the separate path |

`mega_moe_split` is a backend of its own. Nothing chooses between the two at run time and there is no row
threshold: every forward of an engine started with it runs the split kernels. Which backend is faster depends on
the rows per forward (requests per GPU times the verify tokens), on how the routes spread over the experts and on
the expert placement, so measure both on your own traffic and keep the faster one.

**Launch.** `OPT_MOE_BACKEND=mega_moe_split` with `decode-engine.sh` or `decode-step-engine.sh`; everything else
stays the same, including the switch set. The scripts then pass
`--moe-backend mega_moe_split --draft-moe-backend mega_moe`: the split kernels need an even top-k, and the DSpark
draft model routes to three experts, so it stays on the persistent kernel (`--mega-moe-transport` is still
required for it). An engine whose draft model is given the split kernels stops at start-up with a message that
names the option.

**FlashInfer.** The split kernels are not in upstream `656eaee45`. `OPT_FLASHINFER_SRC` must be a checkout that
ships them on FlashInfer's NCCL transport; ours is `https://github.com/Euynaheh/flashinfer`, branch
`sm120-w4a8-megamoe`, with its submodules. The two-node readings below used commit `1ecb6730b549` with the patch
of `../flashinfer/` (packed dispatch at EP16, optional compile buckets); the one-node readings used commit
`0dc6f0785abd`. Both backends run from that checkout: the persistent kernel reads the same step times on it as
on `656eaee45`.

**What to expect.** It depends on the traffic and on the expert placement; `../docs/EP16_SPLIT_RESULTS.md` has
the full results and the method (Chinese), `../results/ep16-split-20261126/` the raw files.

*Two nodes (EP16), real text, static placement.* The workload of `../realtext/` (8192-token prompts, 1024 output
tokens, greedy, the real DSpark acceptance, closed waves), `d7-same.sh`, both backends with the same placement
file (a permutation solved from a held-out prompt bank; real text needs no replicas: the busiest rank receives
1.99 times the mean without a placement and 1.01 times with it). Output tokens per second of the 16 GPUs,
2026-11-26:

| Requests per GPU | 1 | 2 | 4 | 8 | 16 | 24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mega_moe` | 1644 | 2864 | 5330 | 9456 | 16093 | 20963 |
| `mega_moe_split` | 1901 | 3338 | 6103 | 9990 | 15422 | 18899 |
| Difference | +15.6% | +16.6% | +14.5% | +5.6% | -4.2% | -9.8% |

TPOT goes from 9.73 / 11.17 / 12.01 / 13.54 / 15.91 / 18.32 ms to 8.42 / 9.59 / 10.49 / 12.81 / 16.60 / 20.32 ms.
The split run used the tuned table and the packed dispatch with the default compile buckets; without the table
it reads 1766 / 3068 / 5508 / 9155 / 13902 / 16272 tokens per second, without the packed dispatch 1899 / 3336 /
5709 / 8978 / 13080 / 15818. So on real text the split backend is ahead up to 8 requests per GPU and behind
from 16. A rerun with compile buckets at the six decode row counts and every bucket packing (see "Compile
buckets" below) reads 1924 / 3437 / 6122 / 10032 / 14998 / 18916: 1.2% and 3.0% more at 1 and 2 requests per GPU,
2.7% less at 16, no change elsewhere. At 12 requests per GPU, measured in that run and in one more of the
persistent kernel, the two backends tie (about 13195 against 13099; only two of the split run's five waves were
spread evenly, so that number is not a median of three).

*Two nodes (EP16), synthetic tokens, static placement.* The step-time measurement described below feeds random
tokens (131072-token context, simulated acceptance length 6). Their routes are far more concentrated than those
of real text: without a placement the busiest rank receives 2.88 times the mean, a permutation brings that to
1.74 only, 4 replica slots per rank to 1.09. Decode step (ms), 2026-11-26; the split backend with compile buckets
at the six decode row counts and every bucket packing:

| Requests per rank | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mega_moe`, no placement | 30.49 | 31.73 | 35.18 | 40.83 | 53.80 | 81.97 |
| `mega_moe`, permutation | 30.80 | 32.04 | 35.01 | 38.60 | 48.14 | 68.77 |
| `mega_moe`, 4 replicas per rank | 32.24 | 33.46 | 35.87 | 38.70 | 45.15 | 60.09 |
| `mega_moe_split`, permutation | 25.50 | 27.14 | 31.23 | 39.92 | 55.02 | 89.64 |
| `mega_moe_split`, 4 replicas per rank | 26.54 | 28.13 | 31.16 | 37.42 | 47.58 | 69.73 |

With a placement the split backend leads at low concurrency and the persistent kernel at high. With the
permutation the split backend's step is 17.2% / 15.3% / 10.8% shorter at 1 / 2 / 4 requests per rank and 3.4% /
14.3% / 30.3% longer from 8. With 4 replica slots per rank, which balance this workload much better, it is 17.7% /
15.9% / 13.1% / 3.3% shorter up to 8 and 5.4% / 16.0% longer at 16 and 32. Replicas cost both backends 1 to 1.4 ms
at 1 and 2 requests per rank. Two more split readings with the permutation: with the default compile buckets and
packing threshold 25.98 / 28.27 / 31.32 / 40.09 / 53.90 / 89.67 ms, so the extra buckets with every bucket packing
are worth 1.8% and 4.0% at 1 and 2 requests per rank, nothing at 4, 8 and 32, and cost 2.1% at 16 (one run each);
and on FlashInfer `0dc6f0785abd`, without the packed dispatch and without a placement, 26.33 / 29.64 / 37.11 /
52.30 / 72.80 / 123.66 ms.

*One node (EP8), earlier readings* (FlashInfer `0dc6f0785abd`, no expert placement, `d7-same.sh`). Real text,
output tokens per second of the node:

| Requests per GPU | 2 | 8 | 16 |
| --- | ---: | ---: | ---: |
| Persistent kernel | 1379 | 4197 | 7027 |
| Split kernels | 1594 | 4421 | 6929 |
| Difference | +15.7% | +5.3% | -1.4% |

The column for 2 requests per GPU is `mega_moe_split` as in this branch (TPOT 10.0 against 11.6 ms). The other
two columns come from an earlier build that ran the split kernels for the decode forwards and the persistent
kernel for prefill. Synthetic tokens of the step-time measurement without a placement, the least favourable
case for the split kernels (one expert takes about 12% of a layer's routes and the busiest GPU receives about
twice the mean), decode step (ms), 2026-11-25:

| Requests per rank | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mega_moe` | 29.32 | 31.65 | 35.95 | 40.27 | 49.11 | 66.82 |
| `mega_moe_split` | 26.76 | 28.91 | 33.20 | 42.26 | 56.41 | 83.53 |
| Difference | -8.7% | -8.7% | -7.6% | +4.9% | +14.9% | +25.0% |

**Why the answer depends on the routes.** One MoE launch at EP16 (microseconds; the slowest of the 16 ranks,
median of 100 launches after an L2 flush; the persistent kernel without the shared expert), on balanced routes
(every expert receives the same number of rows) and on power-law routes (Zipf exponent 1.0, no placement):

| Rows per rank | 16 | 32 | 64 | 96 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistent kernel, balanced | 704.5 | 735.2 | 827.4 | 888.8 | 915.5 | 1183.7 |
| Split kernels, balanced | 554.1 | 558.0 | 658.9 | 754.5 | 875.0 | 1435.0 |
| Persistent kernel, power-law | 682.0 | 749.5 | 876.5 | 1038.3 | 1175.6 | 1720.3 |
| Split kernels, power-law | 570.3 | 835.5 | 1067.8 | 1143.4 | 1344.9 | about 2980 |

On balanced routes the split kernels win up to 144 rows per rank; on power-law routes they lose from 32 rows,
and no kernel configuration we tried changes that (at 128 rows the fastest of 23 configurations reads 1315
microseconds on power-law routes). Real traffic with a placement sits between the two.

**Two nodes (EP16): a tuned kernel table, packed dispatch, compile buckets.** FlashInfer takes the split
kernels' configuration for a row bucket (tiles, pipeline stages, the SM split between the two concurrent
kernels, cache hints) from tables measured per model and setup, and selects by rule where it has none. It has
tables for the RTX 6000D (156 SMs) at EP4 and EP8 and none for EP16 over two hosts. This branch carries a table
for that setup (`mega_moe_sm120_split_tuned.json` in
`tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/`) and adds it to FlashInfer's lookup before
the first layer is built. Nothing changes in the launch; the engine log names the table ("Split MegaMoE: tuned
kernel tables added to FlashInfer's: ..."). On two nodes the backend is covered by this branch's multi-rank
kernel test (run on 16 ranks, with its reference check) and by the engine runs above.

*How the table was tuned.* With the tuning tools of the kernels' authors (their standalone runner; not part of
the FlashInfer checkout) at EP16 over the two nodes. First, 2026-11-25, their coordinate descent on balanced
routes for 16, 32, 64, 128 and 256 rows per rank. Then, 2026-11-26, with `../tuning/tune_two_routes.py` on two
route distributions, balanced and power-law, because a table tuned on balanced routes alone need not be the
fastest on real traffic: the rule-based default is the first baseline; a candidate that is more than 0.5%
slower than the current baseline on either distribution is excluded; otherwise it replaces the baseline if the
mean of its two times is lower. At 128 and at 96 rows per rank that rule ends on the knobs the table already
had at 128 rows (15% faster than the default on balanced routes, 20% and 11% on power-law routes, where the
gain depends on the route seed). The table therefore has the same knobs as before plus an anchor at 96 rows.
The other row counts of a DSpark decode step (6, 12, 24, 48, 192) are not tuned under this rule yet.

*Packed dispatch.* The kernels reach experts on another NUMA node or host over RDMA. The authors' standalone
runner sends each of those owners only the rows routed to it (from 32 rows per rank at EP16); FlashInfer's
adapter lacked the sending side, so an engine never packed. The patch of `../flashinfer/` ports it. One MoE
launch at EP16 through this backend (microseconds, balanced routes):

| Row bucket | 7 | 16 | 32 | 64 | 128 | 168 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based selection, no packing | 609.7 | 658.9 | 750.4 | 903.6 | 1261.2 | 1456.5 | 2049.8 |
| Tuned table, no packing | 521.8 | 552.1 | 611.2 | 768.5 | 1105.9 | 1298.9 | 1880.3 |
| Rule-based selection, packing | 610.8 | 658.3 | 695.6 | 792.8 | 1027.0 | 1157.9 | 1598.9 |
| Tuned table, packing | 517.6 | 554.1 | 558.0 | 658.9 | 875.0 | 997.7 | 1435.0 |

The table was tuned without packing; the authors' tuner with packing selects the same knobs at 16, 64, 128 and
256 rows (at 32 rows ours are 0.7% faster), so the table stays as it is.

*Compile buckets at the decode row counts.* A decode step at 1 / 2 / 4 / 8 / 16 / 32 requests per rank has
6 / 12 / 24 / 48 / 96 / 192 rows per rank and runs the default buckets 7 / 16 / 32 / 64 / 128 / 256. With
`MEGAMOE_EXTRA_COMPILE_BUCKETS=6,12,24,48,96,192` (patch of `../flashinfer/`) each of them runs kernels compiled
for exactly its rows, and with `MEGAMOE_XNUMA_PACK_MIN_TOKENS=6` every bucket packs (a 24-row bucket is below
the default packing threshold and is 7% slower than the 32-row bucket without it). One launch, balanced routes,
tuned table (microseconds):

| Rows per rank | 6 | 12 | 24 | 48 | 96 | 192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default buckets | 505.3 | 552.4 | 536.0 | 609.7 | 780.3 | 1243.6 |
| Own buckets, every bucket packs | 498.6 | 519.6 | 533.4 | 607.7 | 754.5 | 1232.5 |
| Difference | -1.3% | -5.9% | -0.5% | -0.3% | -3.3% | -0.9% |

In an engine their effect is smaller and not uniform: the 6- and 12-row buckets help (1 and 2 requests per GPU:
1.2% and 3.0% more output on real text, 1.8% and 4.0% shorter steps on synthetic tokens), the 24-, 48- and
192-row buckets change nothing, and the 96-row bucket costs at 16 requests per GPU on both workloads (2.7% less
output, 2.1% longer steps), although it is 3.3% faster on balanced routes.
`../switch-sets/d7-same-split-ep16.sh` therefore sets the packing threshold and the buckets without 96
(`6,12,24,48,192`; 96 rows then run the default 128-row bucket); that list itself has not run in an engine. The
buckets also cost start-up time (871 instead of 691 s to ready on real text) and memory: an engine with these 18
buckets (step-time launch, KV pool of 524288 tokens) peaked at 79.3 GB of the 85.65 GB of a GPU, and at 82.4 GB
with 4 replica slots per rank.

What to know about the table:

- It applies where FlashInfer's key matches: this model with 384 expert slots, an EP16 group whose P2P domains
  hold 4 GPUs (measured as two hosts of two 4-GPU NUMA nodes) and GPUs with 156 SMs. Any other setup keeps what
  FlashInfer selects. The key does not cover the network: ours is four RDMA NICs per node, one per pair of GPUs
  on the same PCIe switch. On nodes that differ, compare against a run with the table turned off (last item).
- A placement with replicas has more expert slots (384 plus 16 per replica slot of a rank) and matches no table
  of the plain model. The file carries a second table for 448 slots (4 replica slots per rank) with the same
  knobs; it ran in an engine on the step-time workload and is not tuned at that slot count. Any other number of
  replicas runs FlashInfer's rule-based selection until a table for its slot count is added.
- The 96-row anchor is used only when FlashInfer compiles a 96-row bucket; with the default buckets 96 rows run
  the 128-row bucket and its anchor, whose knobs are the same.
- The kernels reach the other NUMA nodes and the other host over RDMA NICs, which NCCL picks per GPU.
  FlashInfer's `MEGAMOE_GIN_HCA_MAP` pins them instead (one NIC name per GPU of a node, in GPU order). The tuning
  ran with each GPU pinned to the NIC on its PCIe switch and with `NCCL_NVLS_ENABLE=0`. Kernel timings without
  the two settings were within 0.3% of those with them, inside the run-to-run spread, so the launch scripts do
  not set them.
- `TOKENSPEED_DISABLE_MEGA_MOE_SPLIT_TUNED_TABLES=1` on every rank runs without it. To use a table of your own,
  see "Tuned kernel tables" in `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.md`.

**Things to know before switching.**

- Numerics: both kernels quantize the expert intermediate to FP8. Against a result without that rounding both are
  about 2.7% off in relative L2 on our test inputs; they differ from each other by 1.2% to 1.6% (the rounding
  falls at another point), so the outputs are in the same precision tier and not bit-identical.
- Reproducibility: the persistent kernel returned the same tokens on every repeat of a workload (1512 requests).
  On the split kernels 2 of 168 requests differed between repeats. The cause is not investigated.
- Start-up and memory: the split kernels are compiled per row bucket and every bucket up to the MegaMoE row
  capacity is launched once before CUDA graph capture. With a capacity of 2048 rows, on one node (EP8) the engine
  took 13 minutes to its first request instead of 5 and held 2.8 GB more per GPU. On two nodes (EP16) it was
  ready after 8 to 11 minutes instead of 6 (14 minutes when the kernels of a configuration compile for the first
  time) and held 6.5 GB more per GPU with the tuned table and 7.0 GB with the rule-based selection (free memory
  after graph capture with a KV pool of 524288 tokens: 20.2 GB on the persistent kernel, 13.7 and 13.3 GB on the
  split kernels; 13.0 GB with the tuned table and the default pool). The six extra compile buckets of
  the runs above add to both: that engine peaked at 79.3 GB of the 85.65 GB of a GPU with a permutation
  placement and at 82.4 GB with 4 replica slots per rank (step-time launch, KV pool of 524288 tokens), so check
  the headroom before adding replicas or a larger pool.
- Prefill-sized forwards run the split kernels too, and are slower on them. That is why this backend is meant
  for decode engines.
- Expert placement files (`TOKENSPEED_MOE_EXPERT_PLACEMENT`, see `../placement/`) apply to both backends, and
  both need one to be compared fairly. On the split backend they are covered by the kernel test and by the
  two-node engine runs above (a permutation on real text; a permutation and 4 replica slots per rank on
  synthetic tokens). FlashInfer serves at most 64 expert slots per rank there when the EP group spans NUMA
  nodes, and it selects its tuned kernel configuration by the total number of expert slots (see "What to know
  about the table").

## Environment of the measurements

- Two nodes, 8x RTX 6000D each, driver 580.95.05, CUDA toolkit 13.4 for JIT builds.
- Python 3.12, torch 2.14.0+cu130, the dependencies of the base commit `ee5b468f1`.
- FlashInfer: source checkout of upstream `656eaee45`, placed before the installed package on `PYTHONPATH`
  (the readings under "Two MegaMoE backends for decode" used the checkout named there).
- `tokenspeed` and `tokenspeed_kernel` run from the checkout through `PYTHONPATH`. The 24 compiled libraries of
  the kernel package (`tokenspeed_kernel/thirdparty/cuda/objs`, not tracked in git) were copied into the checkout.
  23 are byte-identical to the package `tokenspeed_kernel` 0.1.4.post20261106 of the base commit. The exception is
  `trtllm_comm/trtllm_comm.so`, a rebuild that came with the measured code; its source is not in this branch, and
  whether these launches exercise it was not checked. This branch changes no compiled source.

## Step-time measurements

`decode-step-engine.sh` is the engine for a step-time measurement with a fixed number of requests per rank; its
header lists what differs from the gold-standard launch. Our readings with it, for comparison with yours:
two nodes, 16 ranks, `d7-same.sh`, context 131072 tokens, fixed acceptance length 6, rank 0, mean of 100 step
intervals after dropping the 20 smallest and the 20 largest.

| Requests per rank | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode step (ms), PDL on | 30.3 | 31.6 | 35.1 | 40.7 | 53.7 | 81.8 |

A reading of this branch that is about 10 ms above these at every load points to PDL being off.

The prefill step above (8 requests per step) used the same five environment variables, `p7-same.sh` without its
`TOKENSPEED_MOE_EXPERT_PLACEMENT` line (the placement file is solved for another workload), and these engine
arguments:

```text
--dtype bfloat16 --moe-backend mega_moe --mega-moe-transport all_gin --deepseek-v4-mega-moe-max-num-tokens 4352
--world-size 8 --nprocs-per-node 8 --nnodes 1 --node-rank 0
--tensor-parallel-size 1 --data-parallel-size 8 --dense-tp-size 1 --moe-tp-size 1 --expert-parallel-size 8
--query-context-parallel-size 1 --kv-parallel-size 1 --decode-context-parallel-size 1
--deepseek-v41-indexer-query-parallel-size 1 --deepseek-v41-attention-query-parallel-size 1
--disaggregation-mode null --max-model-len 135168 --max-total-tokens 262144
--chunked-prefill-size 4224 --max-prefill-tokens 4224 --max-num-seqs 160 --gpu-memory-utilization 0.98
--engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm
--enforce-eager --disable-prefill-graph --kvstore-write-policy write_through --language-model-only
--ep-num-redundant-experts 0 --init-expert-location trivial
```
