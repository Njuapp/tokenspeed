#!/usr/bin/env bash
# Decode engine behind the real-text measurements (../realtext/): one `ts serve` per node with the HTTP gateway
# on node 0, real DSpark acceptance, greedy sampling, no prefix cache. Run once per node.
# usage: realtext-engine.sh LAYOUT NODE SELF_IP MASTER_IP NETWORK_IF OUTPUT_DIR
#   LAYOUT      dep16 (attention DP16, MoE EP16 over two nodes) or dep8 (one node: attention DP8, MoE EP8)
#   NODE        index of this node, 0 or 1; node 0 serves HTTP on OPT_PORT
#   SELF_IP     this node's address;  MASTER_IP  node 0's address
#   NETWORK_IF  routed interface to the peer node, for the NCCL and Gloo bootstrap
# What differs from decode-engine.sh, and why:
#   no --headless                 requests arrive over HTTP through the gateway, which places them on the ranks
#   --sampling-backend greedy     the workload decodes greedily
#   real acceptance               neither a synthetic nor a simulated acceptance length
#   --disable-prefix-caching      every request prefills its own 8192-token prompt
#   CUDA graph capture sizes      1 to 24 requests per rank
# Environment: the variables of env.sh, plus
#   OPT_MODEL             model directory
#   OPT_EXTRA_ENV         optional: a file of ../switch-sets, sourced before the engine starts (unset = baseline)
#   OPT_EXTRA_ARGS        optional: extra `ts serve` arguments
#   OPT_MAX_TOTAL_TOKENS  optional (default 786432);  MEGA_MOE_TRANSPORT  optional (default all_gin)
#   OPT_MOE_BACKEND       optional: mega_moe (default) or mega_moe_split, as in decode-engine.sh
#   OPT_PORT              optional: HTTP port of node 0 (default 30000)
# The gateway answers /health before the engine is loaded: wait for /readiness to return 200.
# PDL stays on: do not add --disable-pdl (see README.md).
set -euo pipefail
layout=${1:?dep16 or dep8}
node=${2:?physical node index}
self_ip=${3:?local address}
master_ip=${4:?frontend address}
network_if=${5:?routed interface to the peer node}
out=${6:?output directory}
: "${OPT_MODEL:?model directory}"
[[ "$node" == 0 || "$node" == 1 ]]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_SHIFTED_POST_PRE_NORM=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton
export TOKENSPEED_KERNEL_OVERRIDE_MOE_TOPK=triton TS_DISABLE_FUSED_TOPK_TOPP=1
export TOKENSPEED_HOST_IP="$self_ip" HOST_IP="$self_ip"
export NCCL_SOCKET_IFNAME="$network_if" GLOO_SOCKET_IFNAME="$network_if" NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
unset NCCL_IB_HCA NCCL_IB_DISABLE
if [[ -n "${OPT_EXTRA_ENV:-}" ]]; then source "$OPT_EXTRA_ENV"; fi
case "${OPT_MOE_BACKEND:-mega_moe}" in
    mega_moe) moe_backend=(--moe-backend mega_moe) ;;
    mega_moe_split) moe_backend=(--moe-backend mega_moe_split --draft-moe-backend mega_moe) ;;
    *) echo "OPT_MOE_BACKEND must be mega_moe or mega_moe_split" >&2; exit 2 ;;
esac
case "$layout" in
    dep16)
        topology=(--world-size 16 --nprocs-per-node 8 --nnodes 2 --node-rank "$node"
            --tensor-parallel-size 1 --data-parallel-size 16 --dense-tp-size 1 --moe-tp-size 1
            --expert-parallel-size 16 --dist-init-addr "$master_ip:31487")
        ;;
    dep8)
        topology=(--world-size 8 --nprocs-per-node 8 --nnodes 1 --node-rank 0
            --tensor-parallel-size 1 --data-parallel-size 8 --dense-tp-size 1 --moe-tp-size 1
            --expert-parallel-size 8 --dist-init-addr "$self_ip:31487")
        ;;
    *) exit 2 ;;
esac
mkdir -p "$out"
command=(ts serve --model "$OPT_MODEL" --served-model-name deepseek-v4.1-flash --trust-remote-code
    --host 0.0.0.0 --port "${OPT_PORT:-30000}"
    --sampling-backend greedy --dtype bfloat16
    "${moe_backend[@]}" --mega-moe-transport "${MEGA_MOE_TRANSPORT:-all_gin}"
    --deepseek-v4-mega-moe-max-num-tokens 2048
    --engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm
    --max-model-len 131072 --gpu-memory-utilization 0.98
    --max-total-tokens "${OPT_MAX_TOTAL_TOKENS:-786432}" --max-num-seqs 1024
    --chunked-prefill-size 2048 --max-cudagraph-capture-size 24
    --cudagraph-capture-sizes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    --disable-prefill-graph --disable-kvstore --disable-prefix-caching
    "${topology[@]}"
    --speculative-algorithm DSPARK --speculative-num-steps 5 --speculative-num-draft-tokens 6
    --hf-overrides '{"dspark_block_size":5}')
if [[ -n "${OPT_EXTRA_ARGS:-}" ]]; then read -r -a _extra <<< "$OPT_EXTRA_ARGS"; command+=("${_extra[@]}"); fi
printf '%q ' "${command[@]}" > "$out/node-$node-command.txt"
printf '\n' >> "$out/node-$node-command.txt"
exec "${command[@]}"
