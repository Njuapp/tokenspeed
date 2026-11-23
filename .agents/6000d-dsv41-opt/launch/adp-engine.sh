#!/usr/bin/env bash
# Prefill gold standard ADP8, as measured: one node, attention DP8, MoE EP8.
# usage: adp-engine.sh cold|hot OUTPUT_DIR
# Environment: the variables of env.sh, plus
#   OPT_MODEL       model directory
#   OPT_HOST_IP     this node's address
#   OPT_NET_IF      network interface NCCL and Gloo bootstrap over
#   OPT_EXTRA_ENV   optional: a file of ../switch-sets, sourced before the engine starts (unset = baseline)
#   OPT_EXTRA_ARGS  optional: extra `ts serve` arguments
#   MEGA_MOE_TRANSPORT  optional (default all_gin)
# PDL stays on: do not add --disable-pdl (see README.md).
set -euo pipefail
cache_mode=${1:?cold or hot}
out=${2:?output directory}
: "${OPT_MODEL:?model directory}" "${OPT_HOST_IP:?address of this node}" "${OPT_NET_IF:?network interface}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_SHIFTED_POST_PRE_NORM=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton
export TOKENSPEED_KERNEL_OVERRIDE_MOE_TOPK=triton
export TS_DISABLE_FUSED_TOPK_TOPP=1
export TOKENSPEED_HOST_IP=$OPT_HOST_IP HOST_IP=$OPT_HOST_IP
export NCCL_SOCKET_IFNAME=$OPT_NET_IF GLOO_SOCKET_IFNAME=$OPT_NET_IF NCCL_DEBUG=INFO
unset NCCL_IB_HCA NCCL_IB_DISABLE MC_INTRANODE_NVLINK
if [[ -n "${OPT_EXTRA_ENV:-}" ]]; then source "$OPT_EXTRA_ENV"; fi
command=(ts serve --headless --model "$OPT_MODEL" --served-model-name deepseek-v41-flash
    --world-size 8 --nprocs-per-node 8 --nnodes 1 --node-rank 0
    --tensor-parallel-size 1 --data-parallel-size 8 --dense-tp-size 1
    --moe-tp-size 1 --expert-parallel-size 8
    --query-context-parallel-size 1 --kv-parallel-size 1 --decode-context-parallel-size 1
    --deepseek-v41-indexer-query-parallel-size 1
    --deepseek-v41-attention-query-parallel-size 1
    --sampling-backend greedy --dtype bfloat16
    --moe-backend mega_moe --mega-moe-transport "${MEGA_MOE_TRANSPORT:-all_gin}"
    --deepseek-v4-mega-moe-max-num-tokens 2048
    --engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm
    --max-model-len 131072 --gpu-memory-utilization 0.98
    --max-total-tokens 262144 --max-num-seqs 192 --chunked-prefill-size 2048
    --enforce-eager --disable-prefill-graph --disable-kvstore --trust-remote-code
    --disable-overlap-schedule --decode-log-interval 1
    --disaggregation-mode null --host "$OPT_HOST_IP" --port 19645
    --data-parallel-address "$OPT_HOST_IP" --data-parallel-rpc-port 29610
    --zmq-engine-index 0 --dist-init-addr "$OPT_HOST_IP:29700"
    --speculative-algorithm DSPARK --speculative-num-steps 5
    --speculative-num-draft-tokens 6 --hf-overrides '{"dspark_block_size":5}'
    --synthetic-acceptance-length 5.35)
case "$cache_mode" in
    cold) command+=(--disable-prefix-caching) ;;
    hot) ;;
    *) exit 2 ;;
esac
if [[ -n "${OPT_EXTRA_ARGS:-}" ]]; then read -r -a _extra <<< "$OPT_EXTRA_ARGS"; command+=("${_extra[@]}"); fi
# OPT_DROP_ARGS (set by the -overlap switch sets): space-separated flags without a value to remove from the
# command, for experiments on a setting the gold standard fixes (--disable-overlap-schedule). Unset = no change.
if [[ -n "${OPT_DROP_ARGS:-}" ]]; then
    _kept=()
    for _arg in "${command[@]}"; do
        _drop=0
        for _d in $OPT_DROP_ARGS; do [[ "$_arg" == "$_d" ]] && _drop=1; done
        [[ $_drop == 1 ]] || _kept+=("$_arg")
    done
    command=("${_kept[@]}")
fi
printf '%q ' "${command[@]}" > "$out/node-0-command.txt"
printf '\n' >> "$out/node-0-command.txt"
exec "${command[@]}"
