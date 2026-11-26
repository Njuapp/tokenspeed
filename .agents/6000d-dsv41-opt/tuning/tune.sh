#!/bin/bash
# Two-distribution tuning of the SM120 W4A8 split MegaMoE for DeepSeek-V4.1-Flash at EP16 (two hosts of 8 GPUs),
# as run for the table of this branch. Needs the tuning package of the split kernels' authors (their standalone
# runner, their tune_table.py and their two-host launcher run2_tune.sh); nothing in it is changed.
# usage: tune.sh ANCHORS RULE WORKNAME PL_MARGIN [VAR=VAL ...]
#   ANCHORS    rows per rank, comma-separated, e.g. 96 or 6,12,24,48,96,192
#   RULE       sequential (the rule of the published table) or power_law (best power_law time alone)
#   WORKNAME   name of the work directory under TUNE_WORK (trial cache, logs) and of the output JSON
#   PL_MARGIN  a first power_law trial more than this slower than the baseline is excluded without repeats
#              (we used 0.015 up to 128 rows and 0.04 above, where single trials are noisier)
#   VAR=VAL    passed to every trial through the launcher (e.g. MEGAMOE_XNUMA_PACK=0)
# Environment:
#   MM_PKG      root of the tuning package (its tools/env.sh is sourced: MM_SRC, MM_ROOT, MM_NUM_SMS, MM_PEER)
#   TUNE_WORK   optional: where work directories go (default: ./work next to this script)
#   TUNE_SHIPPED_TABLE  optional: the table whose anchors are tried as candidates (default: this branch's)
: "${MM_PKG:?root of the tuning package of the split kernels}"
H=$(cd "$(dirname "$0")" && pwd)
. "$MM_PKG/tools/env.sh"
A=$1; RULE=$2; W=${TUNE_WORK:-$H/work}/$3; MARGIN=$4; shift 4; mkdir -p "$W"
SHIPPED=${TUNE_SHIPPED_TABLE:-$H/../../../tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mega_moe_sm120_split_tuned.json}
ENVS=(); [ $# -gt 0 ] && ENVS=(--launcher_env "$@")
# The extra starting points of the package's own tools/tune.sh for 156 SMs.
STARTS=(
 '{"backend": "p2p_split", "k1_n": 16, "k2_n": 16, "k2_warps": 8, "k1_ctas_per_sm": 2, "k2_launch_factor": 2, "weight_prefetch_ahead": 0, "fc1_utility_postprocess": "on", "weight_eviction_policy": "evict_first", "k2_tail_reclaim": "on", "dispatch_chunk_tokens": 32, "k2_ready_queue_bundle": 2}'
 '{"backend": "p2p_split", "k1_n": 16, "k2_n": 32, "k2_warps": 8, "k1_ctas_per_sm": 2, "k2_launch_factor": 2, "k1_stages": 3, "weight_prefetch_ahead": 0, "fc1_utility_postprocess": "on", "weight_eviction_policy": "evict_first", "k2_tail_reclaim": "on", "dispatch_chunk_tokens": 32, "k2_ready_queue_bundle": 2}'
 '{"backend": "p2p_split", "k1_n": 16, "k2_n": 16, "k2_warps": 8, "k1_ctas_per_sm": 1, "k2_launch_factor": 2, "k1_stages": 3, "weight_prefetch_ahead": 4, "fc1_utility_postprocess": "on", "weight_eviction_policy": "evict_first", "k2_tail_reclaim": "on", "dispatch_chunk_tokens": 32, "k2_ready_queue_bundle": 2}'
 '{"backend": "p2p_split", "k1_n": 32, "k2_n": 16, "k2_warps": 8, "k1_ctas_per_sm": 2, "k2_launch_factor": 2, "k1_stages": 2, "weight_prefetch_ahead": 0, "fc1_utility_postprocess": "on", "weight_eviction_policy": "default", "k2_tail_reclaim": "on", "dispatch_chunk_tokens": 32, "k2_ready_queue_bundle": 2, "sms": [88, 0, 0, 68]}'
 '{"backend": "p2p_split", "k1_stages": 3, "k2_tail_reclaim": "on", "sms": [112, 0, 0, 44], "weight_eviction_policy": "evict_first", "fc1_utility_postprocess": "on"}'
 '{"backend": "p2p_split", "k1_stages": 3, "k2_tail_reclaim": "on", "sms": [112, 0, 0, 44], "weight_eviction_policy": "evict_first", "k2_n": 32}'
 '{"backend": "p2p_split", "fc1_utility_postprocess": "on", "k1_stages": 3, "k2_n": 32, "k2_tail_reclaim": "on", "weight_eviction_policy": "evict_first", "weight_prefetch_ahead": 4, "sms": [112, 0, 0, 44]}'
)
echo "[tune] $(date '+%F %T') anchors=$A rule=$RULE margin=$MARGIN env=${ENVS[*]} work=$W" | tee -a "$W.out"
python3 "$H/tune_two_routes.py" --base "$MM_SRC/tools/sm120_w4a8_tuning/tune_table.py" --model dsv41 --ep 16 \
  --gpus 0,1,2,3,4,5,6,7 --num_sms "$MM_NUM_SMS" --anchors "$A" \
  --launcher "$MM_ROOT/tools/run2_tune.sh" --src "$MM_SRC" --trial_timeout "${MM_TRIAL_TIMEOUT:-900}" \
  --extra_sms '[96, 0, 0, 60]' '[120, 0, 0, 36]' --extra_starts "${STARTS[@]}" \
  --shipped "$SHIPPED" --rule "$RULE" --pl_repeats 3 --pl_margin "$MARGIN" "${ENVS[@]}" \
  --work_dir "$W" --output "$W.json" 2>&1 | tee -a "$W.out"
rc=${PIPESTATUS[0]}
echo "[tune] $(date '+%F %T') finished rc=$rc -> $W.json" | tee -a "$W.out"
exit $rc
