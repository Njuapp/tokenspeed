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
