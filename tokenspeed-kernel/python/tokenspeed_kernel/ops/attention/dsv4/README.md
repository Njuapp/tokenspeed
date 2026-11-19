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
