# Tree all8, ADP8 prefill: p8-prec1 with overlap scheduling ON (the handoff fixes --disable-overlap-schedule; the
# customer benchmark requires it on). Outside the handoff settings: report separately.
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
export OPT_DROP_ARGS=--disable-overlap-schedule
