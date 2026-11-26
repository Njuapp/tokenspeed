# Raw results: `mega_moe_split` on two nodes (EP16), 2026-11-26

The files behind `../../docs/EP16_SPLIT_RESULTS.md` and the two-node readings of `../../launch/README.md`.
Two nodes of 8x RTX 6000D, attention DP16, MoE EP16, `d7-same.sh`, FlashInfer as in `../../flashinfer/`.
Paths and addresses of our site are replaced by placeholders (`<work>`, `<node0>`, ...).

## `tuning/` (the kernels' standalone runner, microseconds per launch)

| File | Content |
| --- | --- |
| `sequential-96.json`, `sequential-128.json` | Output of `tune_two_routes.py --rule sequential` for 96 and 128 rows per rank: the final knobs, both times, the default, the table's earlier anchor, every step that replaced the baseline, the second-seed confirmation, the three power-law trials behind each median |
| `sequential-*.log` | The tuner's log: every trial and every baseline change |
| `earlier-rule-*.json`, `.log` | 128 and 256 rows under two earlier readings of the rule (see the tuner's header); single power-law trials |
| `power-law-of-excluded-candidates-128.log` | Power-law time of the 23 configurations within 12% of the best balanced time at 128 rows |
| `power-law-repeatability.log` | The same configuration measured again: the default and the table's anchor at 256 and 128 rows |
| `power-law-spot-64-32-16.log` | Power-law time of the default, the table's anchor and the generic preset at 64, 32 and 16 rows (single trials) |
| `trials.jsonl` | Every trial of the day, one JSON object per line: key, rows, route, seed, knobs, p50 in microseconds |

## `kernel/` (through this branch's wrapper, `tuning/kernel-bench/`)

One line per row count. `launch p50` is the kernel launch, `run p50` adds the wrapper's staging and output copy,
`check=ok` is the reference check of the output.

| Files | Content |
| --- | --- |
| `p1-none`, `p1-asis`, `p1-packtab` | Split kernels, default buckets, packed dispatch: rule-based, this branch's table, the authors' tuner output with packing |
| `p1-nopack-none`, `p1-nopack-asis` | The same without packed dispatch |
| `c1-bal`, `d1-cake` | Persistent kernel, balanced routes: at the bucket sizes; at the row counts of decode steps |
| `d1-none`, `d1-asis` | Split kernels at the row counts of decode steps (6 rows per request), default buckets |
| `d2-cake-pl*`, `d3-cake-pl6*` | Persistent kernel on power-law routes: seed 9341 three times, seed 6197; at 6 x 2^N rows for both seeds |
| `k6-old`, `k6-new` | Split kernels with the default buckets and with buckets at 6 x 2^N rows (packing threshold unchanged) |
| `k6-p24`, `k6-p12`, `k6-p6` | The new buckets with the packing threshold at 24, 12 and 6 rows |
| `pt16-b6` | The multi-rank kernel test on 16 ranks with the new buckets |

## `realtext/` (output tokens per second of the 16 GPUs; `realtext/README.md` of this folder's parent explains the client)

Run directories with `cc<concurrency>/bench_summary.txt` (one JSON line per wave, `RESULT`, the per-wave spread
over the ranks and `EVEN_RESULT`, the number we quote) and the device memory at ready and at the end.
`../../realtext/show.py DIR` prints one.

| Run | Backend and settings |
| --- | --- |
| `rt16-profile-1126b` | Persistent kernel with the load profile on, held-out prompt bank: not a benchmark |
| `rt16-cake-r0-1126a` | Persistent kernel, permutation placement |
| `rt16-cake-r4-1126a` | Persistent kernel, placement with 4 replica slots per rank |
| `rt16-split-r0-1126a` | Split kernels: tuned table, packed dispatch, default buckets, permutation placement |
| `rt16-splitrule-r0-1126a` | The same with the rule-based selection instead of the table |
| `rt16-splitnopack-r0-1126a` | The same as the first with the packed dispatch off |
| `rt16-split-r0-b6-1126a` | Split kernels with `d7-same-split-ep16.sh` (buckets at 6 x 2^N rows, every bucket packs), permutation placement; adds 12 requests per GPU (CC 192), where only two waves were spread evenly |
| `rt16-cake-r0-c192-1126a` | Persistent kernel at 12 requests per GPU, permutation placement |


## `steptime/` (decode step in milliseconds, the step-time measurement of `launch/README.md`)

`steptime.txt`: one line per run with the trimmed mean step per number of requests per rank, then each run's
settings and `ts serve` command line, then the peak device memory of the split runs. In the settings, tree
`all13` is this branch at commit `78d60c9` and `all15` is the same code with the kernel table of this commit;
the switch files are `d7-same.sh` plus the placement file and, for the `b6p` runs, the two variables of
`switch-sets/d7-same-split-ep16.sh`.

| Run | Backend and settings |
| --- | --- |
| `c16-cake-prof-1126a` | Persistent kernel, no placement, load profile on |
| `c16-cake-r0-1126a`, `c16-cake-r4-1126a` | Persistent kernel with the permutation placement and with 4 replica slots per rank |
| `c16-split-r0-b6p-1126a`, `c16-split-r4-b6p-1126a` | Split kernels, buckets at 6 x 2^N rows, every bucket packs, the two placements |
| `c16-split-r0-1126a` | Split kernels, default buckets and packing threshold, the table as committed before this day, permutation placement |
| `dep16-split-tab-1126a` | Split kernels on FlashInfer `0dc6f0785abd`: no packed dispatch, no placement |

## `placement/`

The placement files of the two workloads (`realtext-*`, `steptime-*`; `r0` = permutation, `r4` = 4 replica slots
per rank) and the solver's reports (per layer: busiest rank over the mean before and after, share of routes
that cross GPUs). They belong to these workloads; solve your own for other traffic (`../../placement/`).
