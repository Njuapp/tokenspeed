# Tree all7, ADP8 prefill, tier "bit-identical": placement with 8 hot-expert replicas, fused mHC, and the round-2
# bit-identical items (no zero fill of the attention output, gather only selected history, grouped index scoring,
# row-bounded decoder MegaMoE). Engram host prefetch stays off (no measurable engine gain).
export TOKENSPEED_MOE_EXPERT_PLACEMENT=/deployment/opt-20261117/work/p-moe/placement-adp-ep8-r8.json
export TOKENSPEED_SM120_MHC_FUSED=1
export TOKENSPEED_PREFILL_ATTN_SKIP_ZERO_FILL=1
export TOKENSPEED_PREFILL_GATHER_SELECTED_HISTORY=1
export TOKENSPEED_SM120_DSV41_INDEX_GROUPED=1
export TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND=1
