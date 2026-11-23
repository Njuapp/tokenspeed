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
| MoE | `--moe-backend mega_moe --mega-moe-transport all_gin` | The fused MegaMoE path the optimizations build on |
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

## Environment of the measurements

- Two nodes, 8x RTX 6000D each, driver 580.95.05, CUDA toolkit 13.4 for JIT builds.
- Python 3.12, torch 2.14.0+cu130, the dependencies of the base commit `ee5b468f1`.
- FlashInfer: source checkout of upstream `656eaee45`, placed before the installed package on `PYTHONPATH`.
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
