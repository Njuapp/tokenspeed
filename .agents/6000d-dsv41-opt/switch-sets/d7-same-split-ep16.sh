# Decode on two nodes (EP16) with --moe-backend mega_moe_split, tier "same precision": d7-same plus the two
# FlashInfer settings of ../flashinfer/README.md (the FlashInfer checkout must carry that patch; without it the
# two variables are ignored). Launch with OPT_MOE_BACKEND=mega_moe_split.
#   compile buckets at the row counts of DSpark decode steps (6 rows per request, 1 to 32 requests per GPU)
#   packed dispatch for every bucket instead of from 32 rows per rank
# Add your workload's static expert placement (../placement/README.md):
#   export TOKENSPEED_MOE_EXPERT_PLACEMENT=/path/to/placement.json
source "$(dirname "${BASH_SOURCE[0]}")/d7-same.sh"
export MEGAMOE_EXTRA_COMPILE_BUCKETS=6,12,24,48,96,192
export MEGAMOE_XNUMA_PACK_MIN_TOKENS=6
