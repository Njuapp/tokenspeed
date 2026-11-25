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

Both decode scripts take `OPT_MOE_BACKEND` (`mega_moe`, the default, or `mega_moe_split`); see
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
`sm120-w4a8-megamoe`, commit `0dc6f0785abd`, with its submodules. Both backends run from that checkout: the
persistent kernel reads the same step times on it as on `656eaee45`.

**What to expect.** It depends on the traffic. Readings so far, one node, attention DP8, MoE EP8, `d7-same.sh`:

*Real text* (8192-token prompts, 1024 output tokens, greedy, the real DSpark acceptance, closed loop), output
tokens per second of the node:

| Requests per GPU | 2 | 8 | 16 |
| --- | ---: | ---: | ---: |
| Persistent kernel | 1379 | 4197 | 7027 |
| Split kernels | 1594 | 4421 | 6929 |
| Difference | +15.7% | +5.3% | -1.4% |

The column for 2 requests per GPU is `mega_moe_split` as in this branch (TPOT 10.0 against 11.6 ms). The other
two columns come from an earlier build that ran the split kernels for the decode forwards and the persistent
kernel for prefill; the backend itself is being measured at those loads.

*Synthetic tokens without an expert placement*, the least favourable case for the split kernels. The step-time
measurement described below feeds random tokens (131072-token context, fixed acceptance length 6). Their routes
are far more concentrated than those of real text: one expert takes about 12% of a layer's routes, a layer's
routes reach about 43 of its 384 experts at one request per rank, and with the default placement the busiest GPU
receives about twice the mean number of rows. Decode step (ms), 2026-11-25:

| Requests per rank | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mega_moe` | 29.32 | 31.65 | 35.95 | 40.27 | 49.11 | 66.82 |
| `mega_moe_split` | 26.76 | 28.91 | 33.20 | 42.26 | 56.41 | 83.53 |
| Difference | -8.7% | -8.7% | -7.6% | +4.9% | +14.9% | +25.0% |

The split kernels gain while each active expert receives few rows, and a GPU that receives twice the mean costs
them more than it costs the persistent kernel. A balanced expert placement (`../placement/`) therefore helps
them more: in a kernel-level replay of these routes, the MoE time of the split kernels against the persistent
kernel went from -18% / -17% / -11% / +6% / +22% / +43% at 1 / 2 / 4 / 8 / 16 / 32 requests per rank with the
default placement to -19% / -19% / -16% / -2% / +14% / +36% with the experts rebalanced over the GPUs (no
replicas). Engine results with a static placement, for both backends, will be added here.

**Two nodes (EP16): a tuned kernel table.** FlashInfer takes the split kernels' configuration for a row bucket
(tiles, pipeline stages, the SM split between the two concurrent kernels, cache hints) from tables measured per
model and setup, and selects by rule where it has none. Commit `0dc6f0785abd` has tables for the RTX 6000D (156
SMs) at EP4 and EP8 and none for EP16 over two hosts, so a `dep16` engine ran the rule-based selection. This
branch carries a table for that setup (`mega_moe_sm120_split_tuned.json` in
`tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/`) and adds it to FlashInfer's lookup before
the first layer is built. Nothing changes in the launch; the engine log names the table ("Split MegaMoE: tuned
kernel tables added to FlashInfer's: ..."). FlashInfer's notes at this commit say that FlashInfer itself does not
run these kernels at EP16; on two nodes the backend is covered by this branch's multi-rank kernel test (run on
16 ranks, with its reference check) and by the engine runs below.

It was measured with the tuning tools of the kernels' authors (`tune_table.py` with the kernels' standalone
runner; they are not part of the FlashInfer checkout) at EP16 over the two nodes: coordinate descent per anchor,
median of 100 launches after an L2 flush, balanced routes (every expert receives the same number of rows), a
second route seed to confirm, for 16, 32, 64, 128 and 256 rows per rank. Every anchor was then timed through this
backend with balanced routes and with decode routes recorded from the engine, with a reference check of the
output. One anchor differs from the tuner's choice: at 32 rows the knobs of the 64-row anchor are about 10% faster
on the recorded decode routes and no slower on balanced routes. Buckets above 256 rows keep the rule-based
selection.

One MoE launch at EP16 through this backend (microseconds): the kernel launch without the input staging and the
output copy, slowest of the 16 ranks, median of 100 launches, balanced routes. A decode step at 1 / 2 / 4 / 8 /
16 / 32 requests per rank has 6 / 12 / 24 / 48 / 96 / 192 rows per rank and runs the buckets 7 / 16 / 32 / 64 /
128 / 256.

| Row bucket | 7 | 16 | 32 | 64 | 128 | 168 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based selection | 609.7 | 660.5 | 750.4 | 904.5 | 1258.9 | 1456.7 | 2050.5 |
| Tuned table | 520.8 | 552.4 | 611.7 | 768.8 | 1106.7 | 1298.8 | 1883.7 |
| Difference | -14.6% | -16.4% | -18.5% | -15.0% | -12.1% | -10.8% | -8.1% |

On the recorded decode routes of the step-time workload (synthetic tokens, concentrated routes), the sum over
the same 800 MoE launches (20 decode steps of 40 layers) at 1 / 2 / 4 / 8 / 16 / 32 requests per rank changes by
-10.0% / -13.5% / -12.4% / -12.2% / -15.6% / -2.3%.

Decode step of the engine (ms), two nodes, 16 ranks, `d7-same.sh`, the step-time measurement described below
(synthetic tokens, no expert placement), 2026-11-26:

| Requests per rank | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mega_moe` | 30.36 | 31.62 | 35.05 | 40.61 | 53.67 | 81.68 |
| `mega_moe_split`, rule-based selection | 27.45 | 31.23 | 39.85 | 57.22 | 82.23 | 127.31 |
| `mega_moe_split`, tuned table | 26.33 | 29.64 | 37.11 | 52.30 | 72.80 | 123.66 |
| Tuned table against rule-based | -4.1% | -5.1% | -6.9% | -8.6% | -11.5% | -2.9% |
| Tuned table against `mega_moe` | -13.3% | -6.3% | +5.9% | +28.8% | +35.6% | +51.4% |

The three runs had a KV pool of 524288 tokens (`OPT_MAX_TOTAL_TOKENS`; the script's default is 786432), and the
two `mega_moe_split` runs pinned the NICs as described below. A repeat of the tuned-table run as
`decode-step-engine.sh` launches it (default pool, NICs left to NCCL) read 26.41 / 29.76 / 37.10 / 52.32 / 72.76
/ 123.61 ms.

So on two nodes the table shortens every step of the split backend. On this workload the split backend is ahead
of the persistent kernel at 1 and 2 requests per rank and behind from 4, with the table (13.3% and 6.3% ahead)
and without it (9.6% and 1.2%). Real text spreads its routes more, which on one node moved that point up; it is
not measured on two nodes.

What to know about the table:

- It applies where FlashInfer's key matches: this model with 384 expert slots, an EP16 group whose P2P domains
  hold 4 GPUs (measured as two hosts of two 4-GPU NUMA nodes) and GPUs with 156 SMs. Any other setup keeps what
  FlashInfer selects; a placement with replicas has more expert slots and therefore keeps the rule-based
  selection. The key does not cover the network: ours is four RDMA NICs per node, one per pair of GPUs on the
  same PCIe switch. On nodes that differ, compare against a run with the table turned off (last item).
- It was tuned for the dispatch FlashInfer `0dc6f0785abd` runs in an engine. The kernels' standalone runner packs
  the dispatch to the other NUMA nodes and the other host, by default from 32 rows per rank at EP16; with the
  rule-based configuration at 64 rows that reads 788.6 against 901.7 microseconds per launch. FlashInfer's
  adapter at this commit does not use the packed dispatch (`MEGAMOE_XNUMA_PACK` has no effect in an engine), so
  the tuner ran with packing off. A FlashInfer that packs needs a table tuned with packing (the tuner's default);
  once FlashInfer ships a table of its own for this setup, that one is used and ours is ignored.
- The kernels reach the other NUMA nodes and the other host over RDMA NICs, which NCCL picks per GPU.
  FlashInfer's `MEGAMOE_GIN_HCA_MAP` pins them instead (one NIC name per GPU of a node, in GPU order). The tuning
  and the kernel timings above ran with each GPU pinned to the NIC on its PCIe switch and with
  `NCCL_NVLS_ENABLE=0`. Kernel timings without the two settings (buckets 16, 64, 128 and 256 with balanced routes;
  2, 8 and 16 requests per rank on the recorded routes) were within 0.3% of those with them, inside the
  run-to-run spread, so the launch scripts do not set them.
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
  split kernels; 13.0 GB with the tuned table and the default pool).
- Prefill-sized forwards run the split kernels too, and are slower on them. That is why this backend is meant
  for decode engines.
- Expert placement files (`TOKENSPEED_MOE_EXPERT_PLACEMENT`, see `../placement/`) apply to both backends. On the
  split backend this is covered by the kernel test only, not yet by an engine run; FlashInfer serves at most 64
  expert slots per rank there when the EP group spans NUMA nodes, and it selects its tuned kernel configuration
  by the total number of expert slots, so a placement with replicas runs its rule-based selection.

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
