# Raw results: one node (EP8) against two (EP16), and the load-profile fix, 2026-11-27

The files behind `../../docs/EP8_VS_EP16.md`, `../../docs/LOAD_PROFILE_FILLER_ROWS.md` and section 4.1 of
`../../docs/EP16_SPLIT_RESULTS.md`. Nodes of 8x RTX 6000D: one node is attention DP8 and MoE EP8, two nodes are
DP16 and EP16. Every engine ran `d7-same.sh` (the split backend with the two settings of
`d7-same-split-ep16.sh` on top, at both EP sizes) and FlashInfer as in `../../flashinfer/`. Paths and addresses
of our site are replaced by placeholders (`<work>`, `<node0>`, ...). In the run headers tree `all17` is this
branch, `all18` is the same with the load profile counting real rows only (the profile runs), and `all19` is
`all17` with the EP8 kernel table registered for 408 expert slots as well; the branch has all three changes.

## `realtext/` (the measurement of `../../realtext/`)

Run directories with `bench_summary.txt` per level (one JSON line per wave, the per-wave spread over the ranks
and `EVEN_RESULT`) and the device memory at ready and at the end. Two-node runs hold `cc<CC>/`. One-node runs
hold `cc<CC>-h0/` and `cc<CC>-h1/`: the level measured on the two halves of the prompts that the two-node
engine serves at twice the concurrency, to be averaged. `../../realtext/collect.py 16 DIR ..` or `collect.py 8
DIR ..` prints the table lines. A run whose name ends in `x` repeats single levels of the run of the same name,
with the same settings and a new engine start, where the first run had fewer than three evenly spread waves; the
documents use the repeat for those levels.

In the names, `r0` is a permutation and `r4` / `r8` the replica cap given to the solver (the EP8 file asked for
8 has 3 replica slots per rank, hence `r3` in later names); `L` marks a placement solved from a profile that
counts real rows only; `b5` the five extra compile buckets of `d7-same-split-ep16.sh` (the EP8 split runs have
them too).

| Run | Layout, backend, placement |
| --- | --- |
| `rt16-profile-1127c` | EP16, persistent kernel with the load profile on (real rows only), held-out prompt bank: not a benchmark |
| `rt8-profile-1127c` | EP8, the same |
| `rt8-profile-1127d` | EP8, the load profile with the earlier count (filler rows included), for the comparison of the two counts |
| `rt16-cake-r4L-1127c` | EP16, persistent kernel, 4 replica slots per rank |
| `rt16-split-r4L-b5-1127c`, `..cx` | EP16, split kernels, the same placement |
| `rt16-cake-r0L-1127e` | EP16, persistent kernel, permutation |
| `rt16-split-r0L-b5-1127e`, `..ex` | EP16, split kernels, permutation |
| `rt16-split-r4-b5-1127e` | EP16, split kernels, the placement of 2026-11-26 with 4 replica slots per rank (solved from a profile with filler rows) |
| `rt16-split-r0-b5-1127e` | EP16, split kernels, the permutation of 2026-11-26 (solved from a profile with filler rows) |
| `rt8-cake-r8L-1127d`, `..dx` | EP8, persistent kernel, 3 replica slots per rank |
| `rt8-split-r3Ltab-1127e` | EP8, split kernels, the same placement (408 expert slots), kernel table |
| `rt8-splitpack-r3Ltab-1127e`, `..ex` | The same with `MEGAMOE_XNUMA_PACK=1` |
| `rt8-split-r8L-1127d`, `..dx` | The same as `rt8-split-r3Ltab-1127e` before the table was registered for 408 slots: rule-based kernel selection |
| `rt8-cake-r0L-1127e`, `..ex` | EP8, persistent kernel, permutation |
| `rt8-split-r0L-1127e` | EP8, split kernels, permutation (384 expert slots, kernel table) |
| `rt8-splitpack-r0L-1127e` | The same with `MEGAMOE_XNUMA_PACK=1` |
| `rt8-split-r8-1127c`, `..cx` | EP8, split kernels, a placement solved from the profile with filler rows (3 replica slots), rule-based kernel selection: compare with `rt8-split-r8L-1127d` |

The EP16 persistent-kernel run with the placement of 2026-11-26 is `../ep16-split-20261126/realtext/rt16-cake-r4-1126a`.

## `placement/`

| Files | Content |
| --- | --- |
| `realtext-ep16-r0`, `realtext-ep16-r4` | EP16 placements solved from the profile that counts real rows only: permutation, 4 replica slots per rank (448 expert slots). `.json` is the file the engine reads, `.report.txt` the solver's report |
| `realtext-ep8-r0`, `realtext-ep8-r3` | The same at EP8: permutation, and the file asked for 8 replica slots, which has 3 (408 expert slots) |
| `realtext-ep8-old-r0`, `realtext-ep8-old-r3` | EP8 placements solved from the profile with the earlier count |

The EP16 placements of 2026-11-26 are `../ep16-split-20261126/placement/realtext-r0` and `realtext-r4`.

## `steptime/` (decode step in milliseconds, the step-time measurement of `../../launch/README.md`)

| File | Content |
| --- | --- |
| `steptime-balanced.txt` | Balanced routes (`TOKENSPEED_MOE_ROUTING_SIMULATION=balanced`), no placement: EP8 and EP16, both backends, the EP8 split run with the default compile buckets, and the run that recorded a profile to confirm the routes. Per run the trimmed mean per number of requests per rank, each formal round alone, the settings, the switch file's environment, the `ts serve` command line, input hashes and the kernel tables named in the log |
| `pack-control-ep8.txt` | The EP8 split run again with the packed dispatch forced on |

## `checks/`

| File | Content |
| --- | --- |
| `live-rows-ep16.txt`, `live-rows-ep8.txt` | `placement/check_live_rows.py` on the two profiles recorded with the fix: counted decode cycles over those of the requests in the engine logs |
| `live-rows-ep8-old-count.txt` | The same on the EP8 profile recorded with the earlier count |
| `eval-ep16.txt`, `eval-ep8.txt` | `placement/eval_placement.py`: every placement, old and new, evaluated on the profile that counts real rows only (busiest rank over the mean per layer, share of routes that cross ranks) |
| `balance-ep8.txt` | `placement/check_balance.py` on the profile of the balanced-routing run |
| `split-kernel-selection.txt` | Per split run: the kernel configuration of the 192-row bucket from the engine's compile log (tells a table from the rule-based selection) and the packed-dispatch state of the launches |
