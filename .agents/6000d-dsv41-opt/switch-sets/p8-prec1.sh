# Tree all8, ADP8 prefill, tier "changes precision", second step: p7-prec with the attention PV in one BF16 product
# (probabilities rounded to BF16; relative L2 error against FP64 2.0e-3 instead of 1.7e-3). The pipelined kernel serves it in all8.
export TOKENSPEED_MOE_EXPERT_PLACEMENT=/deployment/opt-20261117/work/p-moe/placement-adp-ep8-r8.json
export TOKENSPEED_SM120_MHC_FUSED=1
export TOKENSPEED_PREFILL_ATTN_SKIP_ZERO_FILL=1
export TOKENSPEED_PREFILL_GATHER_SELECTED_HISTORY=1
export TOKENSPEED_SM120_DSV41_INDEX_GROUPED=1
export TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND=1
export TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16
export TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1
export TOKENSPEED_MEGA_MOE_FUSED_SHARED=1
export TOKENSPEED_SM120_WO_A_MXFP8=1
