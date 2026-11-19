# Tree all7/all8, ADP8 prefill, tier "same precision": p7-same with overlap scheduling ON (outside the handoff settings).
export TOKENSPEED_MOE_EXPERT_PLACEMENT=/deployment/opt-20261117/work/p-moe/placement-adp-ep8-r8.json
export TOKENSPEED_SM120_MHC_FUSED=1
export TOKENSPEED_PREFILL_ATTN_SKIP_ZERO_FILL=1
export TOKENSPEED_PREFILL_GATHER_SELECTED_HISTORY=1
export TOKENSPEED_SM120_DSV41_INDEX_GROUPED=1
export TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND=1
export TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16x2
export TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1
export TOKENSPEED_MEGA_MOE_FUSED_SHARED=1
export OPT_DROP_ARGS=--disable-overlap-schedule
