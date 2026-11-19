# Selected attention over a dense workspace

`dsv4_prefill` consumes BF16 queries and a dense BF16 shared-KV workspace.
The caller supplies selected row indices and a valid width for each query.
The operator masks invalid rows, applies the optional sink once, and can
return natural-log LSE for a partial attention result.

On SM120, `triton_dsv4_prefill_sm120` shares each selected K/V tile across
16 heads. It uses BF16 QK products and FP32 accumulation. The PV products
use TF32x3 to retain FP32 softmax probabilities. The two 256-channel PV
tiles fit the device's shared-memory limit. This implementation promises
numerical agreement within BF16 tolerance, not bitwise or batch invariance.

Row counts, selected widths, metadata strides, and cache capacities stay
runtime values. Two warp-count buckets cover occupancy changes. Address
products that grow with a token or workspace row use int64. All temporary
state belongs to the current call; the operator does not retain cache data.

## Opt-in SM120 variants: head grouping and PV arithmetic

Three environment switches, read once at import, all off by default and
honoured on SM120 only. With all at their defaults nothing extra is
registered.

- `TOKENSPEED_SM120_PREFILL_ATTN_H32=1` shares each selected K/V tile across
  32 heads instead of 16. It regroups work and never changes a result bit;
  head counts that are not a multiple of 32 keep 16 heads per tile.
- `TOKENSPEED_SM120_PREFILL_ATTN_PV=tf32x3|bf16x2|bf16` selects how the FP32
  softmax probabilities meet the BF16 K/V tile. `tf32x3` is the default and
  today's arithmetic. `bf16x2` splits each probability into a BF16 high part
  and a BF16 residual and accumulates two BF16 products in FP32, as the V4.1
  paged decode kernel does; it keeps the probability precision, and a stored
  output value differs from `tf32x3` in about 0.2% of positions, by one BF16
  step. `bf16` rounds the probabilities to BF16 and uses one product; about a
  third of the stored values change and the relative L2 error against an FP64
  reference goes from 1.7e-3 to 2.0e-3. Any other value is an import error.

QK, the softmax and the denominator are the same in every combination. Both
switches select `triton_dsv4_prefill_variant_sm120`, which ranking never
picks: a caller opts in by entering `dsv4.prefill_attention_override()`
around its attention launch. The V4.1 prefill backend does so; the DSpark
window attention and decode do not and keep `triton_dsv4_prefill_sm120`. The
variant hands LSE requests and non-BF16 inputs to the default kernel.

Measured at 2048 rows, 64 heads and 640 selected rows on an RTX 6000D (cold
L2, default kernel 4.2 ms): 32 heads alone 4.3 ms; `bf16x2` 2.4 ms at 16
heads and 2.5 ms at 32; `bf16` 3.2 ms at 16 heads and 2.1 ms at 32. The
grouping alone does not pay: three TF32 passes dominate the default kernel.

### Pipelined `bf16x2` / `bf16` kernel

`TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1` requires
`TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16x2` or `bf16` (anything else is an import
error) and replaces that mode's launch by `_compact_attention_bf16x2_pipelined`
(its `RESIDUAL` flag is the PV mode: with `bf16` the residual product is left
out, a third less tensor-core arithmetic per tile; that output is not
bit-identical to the unpipelined `bf16` launch: at most 4e-4 of the stored
values differ, a relative L2 gap below 1e-4, with the same 2.0e-3 error
against an FP64 reference)
for head counts that are a multiple of 32 and a dense workspace of 512-wide
rows with at most 2^22 rows. Other requests keep the launch above. The
arithmetic is that of `bf16x2`, operation for operation, so the stored output
is bit-identical to `bf16x2` at every row count, selected width, validity
pattern and with or without a sink; only the placement of the work changes:

- 32 heads share each selected K/V tile on eight warps. Fetching the tiles
  from L2 is bound at about 5.5 TB/s on this device, 0.9 ms of the 16-head
  kernel at 2048 rows; sharing halves it.
- The PV products sit in their own region, so QK takes a [2, 4] warp layout
  and PV a [1, 8] one. In one region both take [1, 8] and the eight warps
  compute the four QK column tiles twice.
- Five pipeline stages give the K/V tile two shared-memory buffers filled by
  asynchronous copies one iteration ahead; the tile never passes through
  registers. Q and the two buffers take 96 KiB of the device's 99 KiB, which
  is why the residual probabilities are formed after the high product (both
  BF16 operands then share one slot) and why 64 heads per tile cannot fit.
- Workspace element offsets are int32 with a compile-time row stride.

Row counts and selected widths stay runtime values: one specialization per
sink presence and index dtype. Measured through the runtime's call on an RTX
6000D, cold L2, 64 heads: 2048 rows x 640 selected rows 2.45 ms -> 1.34 ms
(default kernel 4.22 ms); 2048 x 128 0.55 -> 0.40 ms; 40 rows x 640 0.15 ->
0.10 ms; 1 row 0.14 -> 0.09 ms. It is faster at every row count, so there is
no small-grid bucket. Back-to-back launches of this kernel exceed the
device's per-SM arithmetic quota (three BF16 passes, as `bf16x2`); time it
with idle gaps.

If Triton cannot build the pipelined kernel (it leaves 768 bytes of shared
memory, so another Triton release or tile constant can raise `OutOfResources`
at the first launch), the process logs one warning and serves every later
request with the `bf16x2` launch: same bits, without the speedup.
