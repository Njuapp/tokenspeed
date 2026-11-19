# Tree all7, ADP8 prefill, tier "same precision": p7-bit plus the two-term BF16 PV in prefill attention with the
# pipelined kernel (bit-identical to bf16x2) and the fused shared expert.
export TOKENSPEED_MOE_EXPERT_PLACEMENT=/deployment/opt-20261117/work/p-moe/placement-adp-ep8-r8.json
export TOKENSPEED_SM120_MHC_FUSED=1
export TOKENSPEED_PREFILL_ATTN_SKIP_ZERO_FILL=1
export TOKENSPEED_PREFILL_GATHER_SELECTED_HISTORY=1
export TOKENSPEED_SM120_DSV41_INDEX_GROUPED=1
export TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND=1
export TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16x2
export TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1
export TOKENSPEED_MEGA_MOE_FUSED_SHARED=1
