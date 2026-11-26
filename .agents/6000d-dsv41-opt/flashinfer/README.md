# FlashInfer changes behind the two-node `mega_moe_split` readings

`sm120-split-pack-ep16-and-buckets.diff` is a patch for the FlashInfer checkout that ships the SM120 split
MegaMoE kernels: `https://github.com/Euynaheh/flashinfer`, branch `sm120-w4a8-megamoe`, commit
`1ecb6730b54958a1852806ab3bdb620f1104431f`, with its submodules. Every two-node (EP16) reading of
`--moe-backend mega_moe_split` dated 2026-11-26 in `../launch/README.md` and `../docs/EP16_SPLIT_RESULTS.md` was
measured on that commit with this patch; both MegaMoE backends ran from the same checkout.

```bash
git clone --recurse-submodules -b sm120-w4a8-megamoe https://github.com/Euynaheh/flashinfer
cd flashinfer && git checkout 1ecb6730b54958a1852806ab3bdb620f1104431f && git submodule update --init --recursive
git apply /path/to/sm120-split-pack-ep16-and-buckets.diff
export OPT_FLASHINFER_SRC=$PWD        # for ../launch/env.sh
```

It changes seven files and nothing outside the SM120 split backend. With none of the variables below set, the
only behaviour that changes is the first item.

1. **Packed dispatch to the other NUMA nodes and the other host.** The kernels' standalone runner sends each
   owner on another NUMA node or host only the rows routed to it, plus a row map, instead of the whole input
   span, and its heuristic turns that on at EP16 from 32 rows per rank. The FlashInfer adapter at this commit
   has the receiving side of it but not the sender (its `VENDOR.md` says so). The patch ports the sender from
   the standalone runner (`transport/xnuma.py`) and plans it per compile bucket in the shim. At EP16 with
   balanced routes one launch of the 64-row bucket takes 658.9 instead of 768.5 microseconds with the tuned
   table (792.8 instead of 903.6 with the rule-based selection); the reference check of the kernel test and of
   our kernel timing passes with the same error as without it.
   `MEGAMOE_XNUMA_PACK=0|1` switches it off or on for every bucket.
2. **`MEGAMOE_XNUMA_PACK_MIN_TOKENS=N`** (default: unset, the heuristic's 32 rows at EP16) moves the row
   threshold. With `N=6` every decode bucket packs; measured at EP16 with balanced routes that is faster at 6, 12
   and 24 rows per rank as well (498.6 / 519.6 / 533.4 against 503.3 / 538.1 / 574.9 microseconds).
3. **`MEGAMOE_EXTRA_COMPILE_BUCKETS=6,12,24,48,96,192`** (default: unset) adds compile buckets. The kernels are
   compiled per row bucket (7, 16, 32, 64, 128, 168, 256, ... by default) and a launch runs the smallest
   bucket that holds its rows. A decode step under DSpark with block size 5 has 6 rows per request, so 1 / 2 /
   4 / 8 / 16 / 32 requests per GPU are 6 / 12 / 24 / 48 / 96 / 192 rows; with these buckets each of them runs
   kernels compiled, and tunable, for exactly its rows. Each bucket costs one more set of compiled kernels and
   launch graphs (start-up time and device memory). Note that a 24-row bucket is below the default packing
   threshold of 32 rows: without item 2 it is slower than the default 32-row bucket. In an engine, items 2 and 3
   together shortened the decode step by 1.8% and 4.0% at 1 and 2 requests per rank, changed nothing at 4, 8 and
   32 and lengthened it by 2.1% at 16 (synthetic tokens, permutation placement, one run each); on real text they
   gave 1.2% and 3.0% more output at 1 and 2 requests per GPU and 2.7% less at 16. The 96-row bucket is faster
   on balanced routes only: leave it out for traffic around 16 requests per GPU.

All three must have the same value on every EP rank. `../switch-sets/d7-same-split-ep16.sh` sets items 2 and 3
the way the engine runs used them.

Checked with the patch: FlashInfer's host tests of this backend (61 pass, with and without the variables),
TokenSpeed's multi-rank kernel test on 16 ranks over two nodes (27 pass, 5 skipped; the log shows which buckets
pack), and a reference check of the output for every bucket we timed.
