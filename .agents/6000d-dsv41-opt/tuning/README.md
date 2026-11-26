# Tuning and timing the split MegaMoE kernels at EP16

What produced the kernel table of this branch for two nodes and the kernel-level numbers of
`../launch/README.md` and `../docs/EP16_SPLIT_RESULTS.md`. Results: `../results/ep16-split-20261126/`.

| File | What it is |
| --- | --- |
| `tune_two_routes.py` | Tunes the kernel configuration of one row count ("anchor") on two route distributions, balanced and power-law. Its header states the rule |
| `tune.sh` | The invocation we used at EP16 over two nodes |
| `show_entries.py`, `show_trials.py` | Print a tuner output file; print every trial of one anchor on both distributions |
| `kernel-bench/bench16.py` | One MoE launch of `mega_moe_split` through this branch's wrapper over an EP group, with a reference check of the output (`--check`); balanced, uniform or recorded routes; `BENCH_TABLE=asis` (the tables as an engine sees them), `none` (rule-based) or a tuner output file |
| `kernel-bench/bench16_cake.py` | The same measurement for the persistent kernel (`mega_moe`), balanced, uniform, power-law or recorded routes |

## Tuning

`tune_two_routes.py` does not contain the kernels' runner. It drives the tuning tools of the split kernels'
authors (their `tune_table.py`, their standalone `mega_runner` and their two-host launcher), which are not part
of the FlashInfer checkout; ask the authors for the tuning package. With it:

```bash
export MM_PKG=/path/to/tuning-package
./tune.sh 96 sequential dsv41-ep16 0.015        # one anchor: about 45 minutes at EP16 (70 trials of 34 s)
./tune.sh 96 power_law dsv41-ep16-pl 0.015      # the best power-law time alone
python3 show_entries.py work/dsv41-ep16.json
```

The rule (`--rule sequential`): the rule-based default is the first baseline; a candidate that is more than
0.5% slower than the current baseline on balanced or on power-law routes is excluded; otherwise the two times
are averaged and the candidate replaces the baseline if its average is lower. A power-law time is the median
of three trials, because single power-law trials are not repeatable to 0.5% at the larger row counts (the
same configuration read 2877 / 3033 / 3062 microseconds at 256 rows per rank).

Tune at the row counts your decode steps have. Under DSpark with block size 5 a request has 6 rows per step,
so 1 / 2 / 4 / 8 / 16 / 32 requests per GPU are 6 / 12 / 24 / 48 / 96 / 192 rows per rank; the runner compiles
for exactly the rows it is given, which matches an engine only if FlashInfer has a compile bucket of that size
(`../flashinfer/README.md`).

A tuner output goes into `mega_moe_sm120_split_tuned.json` as one entry per anchor (`tokens`, `knobs`, and the
measured times for the record); see "Tuned kernel tables" in
`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.md`.

## Kernel timing

Both scripts run under `torch.distributed.run`, one process per GPU, on the engine's `PYTHONPATH`
(`../launch/env.sh`), once per node:

```bash
source ../launch/env.sh
cd kernel-bench
BENCH_TABLE=asis MEGAMOE_EXTRA_COMPILE_BUCKETS=6,12,24,48,96,192 MEGAMOE_XNUMA_PACK_MIN_TOKENS=6 \
python -m torch.distributed.run --nnodes=2 --node_rank=$NODE --nproc_per_node=8 \
    --master_addr=$MASTER_IP --master_port=$PORT \
    bench16.py --tokens 6 12 24 48 96 192 --route balanced --check
python -m torch.distributed.run --nnodes=2 --node_rank=$NODE --nproc_per_node=8 \
    --master_addr=$MASTER_IP --master_port=$PORT \
    bench16_cake.py --tokens 6 12 24 48 96 192 --route power_law
```

Every round writes 512 MB to flush the L2 cache, aligns the ranks and then times one launch replayed from a
CUDA graph; a round's time is the slowest rank's, the reported number the median of 100 rounds after 20 warm-up
rounds. `bench16.py` prints one `BENCHRESULT` line per row count, `bench16_cake.py` one `CAKERESULT` line.
Our runs set `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_NVLS_ENABLE=0` and pinned each GPU to the RDMA
NIC on its PCIe switch with `MEGAMOE_GIN_HCA_MAP`.
