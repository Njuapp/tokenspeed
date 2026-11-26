# Decode on two nodes (EP16) with --moe-backend mega_moe_split, tier "same precision": d7-same plus the two
# FlashInfer settings of ../flashinfer/README.md (the FlashInfer checkout must carry that patch; without it the
# two variables are ignored). Launch with OPT_MOE_BACKEND=mega_moe_split.
#   compile buckets at the row counts of DSpark decode steps (6 rows per request; 1, 2, 4, 8 and 32 requests per GPU)
#   packed dispatch for every bucket instead of from 32 rows per rank
# No 96-row bucket (16 requests per GPU): measured in engines with all six buckets, the 6- and 12-row buckets help,
# 24, 48 and 192 change nothing, and the 96-row bucket is 2% to 3% slower end to end at 16 requests per GPU although
# it is faster on balanced routes. Without it 96 rows run the default 128-row bucket, as in the runs with the
# default buckets. The engine runs of 2026-11-26 had 96 in the list; this list itself has not run in an engine.
# Add your workload's static expert placement (../placement/README.md):
#   export TOKENSPEED_MOE_EXPERT_PLACEMENT=/path/to/placement.json
source "$(dirname "${BASH_SOURCE[0]}")/d7-same.sh"
export MEGAMOE_EXTRA_COMPILE_BUCKETS=6,12,24,48,192
export MEGAMOE_XNUMA_PACK_MIN_TOKENS=6
