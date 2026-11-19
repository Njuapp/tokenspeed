# Tree all7, D16, tier "bit-identical": the (32,128) GEMM tile, grouped index scoring and the four small decode
# items of round 2 (mixes after the router, pre-swizzled RMSNorm scales, Engram lookup under MegaMoE, block-table
# upload on its own stream).
export TOKENSPEED_SM120_MXFP8_TILE_32X128=1
export TOKENSPEED_SM120_DSV41_INDEX_GROUPED=1
export TOKENSPEED_SM120_FFN_MIXES_AFTER_ROUTER=1
export TOKENSPEED_SM120_RMSNORM_SWIZZLED_SCALES=1
export TOKENSPEED_ENGRAM_LOOKUP_UNDER_MEGAMOE=1
export TOKENSPEED_BLOCK_TABLE_UPLOAD_STREAM=1
