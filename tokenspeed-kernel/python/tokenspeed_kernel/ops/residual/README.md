# Residual kernels

## SM120 mHC coefficients

`mhc_mixes` returns the pre, post and combination coefficients. It does not
update residual streams or keep request state.

For BF16 residuals with four streams of width 5120 and 1–16 token rows,
`triton_mhc_mixes_sm120` multiplies the FP32 projection weights with FP32
residual values. Each program reduces one projection row and one part of the
hidden dimension. The implementation uses 16, 8 or 4 parts for token counts
1, 2–8 or 9–16. These three buckets bound compilation. Token counts remain
runtime arguments. Larger batches at the same hidden width use tiled TF32x3
products with FP32 accumulation. This retains FP32 projection weights and
avoids the portable kernel's scalar FP32 matrix products. Other hidden
widths use the portable implementation.

The existing FP32 reducer applies RMS normalization, the pre/post sigmoid
maps and the requested Sinkhorn iterations. Reducing fewer partials changes
the FP32 summation order. Tests compare coefficients with an FP64 reference
at `rtol=1e-5, atol=2e-6`, including CUDA Graph replay with changed inputs.
Weights retain FP32 precision.

The coefficient reducer uses int32 partial-buffer offsets. Its caller checks
the buffer's element count before allocation. Token and split offsets in the
SM120 projection kernels use int64.

## SM120 fused mHC boundary (`TOKENSPEED_SM120_MHC_FUSED`)

After a sublayer the model needs the new residual streams, the layer input
of the next sublayer and that sublayer's coefficients. The separate kernels
read the `[T, 4, 5120]` residual three times: `_mhc_post_hc4_triton_kernel`
(which also writes it), the mixing projection and
`_mhc_pre_layer_norm_hc4_kernel`. The switch is off by default and honoured
on SM120 only.

- `TOKENSPEED_SM120_MHC_FUSED=1`: `mhc_post_input_norm_hc4` produces the new
  residual and the normalized layer input from one read and one write of the
  residual. One program handles one token: per output stream the FP32 post
  product and four FMAs in input-stream order, one rounding to BF16, then the
  layer input as the FP32 sum of the four pre-weighted BF16 streams, one
  rounding to BF16 and the RMSNorm. The operations, their order and the
  one-CTA-per-token variance reduction (same tensor shape and warp count) are
  those of the two kernels, and the results are bit-identical. The mixing
  projection is not part of it: the caller computes the coefficients where it
  does today. `try_mhc_post_input_norm_sm120` is the entry point; the model
  uses it for prefill and mixed forwards and keeps the two kernels in decode,
  where the forked coefficients start right after the post.
- `TOKENSPEED_SM120_MHC_FUSED=2`: as `1`, and `triton_mhc_mixes_sm120` uses at
  least 64 projection splits. Up to 128 rows nothing changes (the unfused
  count is 80 or 64). Above, the split count rises from 32/16/8/4/2/1 to 64,
  so each split accumulates five 64-column tiles instead of up to 320, and the
  reducer sums 64 partials. The products are the same TF32x3 products; only
  the grouping of the FP32 partial sums changes, to the one small batches
  already use. The next coefficients differ in their low-order bits.

A single kernel that also computes the projection (one pass instead of two)
was built and verified bit-identical, but in Triton it is slower than what it
replaces at 2048 rows: 372 us for post and projection with the unfused split
count, and 447 us with 64 splits including the un-normalized layer input,
against 147 us + 176 us for the two separate kernels. A tile-shaped post moves
memory at about half the rate of the per-token one, and the projection's
accumulator chain does not get shorter. It is not part of the tree.
