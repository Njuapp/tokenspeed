#!/usr/bin/env bash
# Decode gold standard D16, as measured. Run once per node.
# usage: decode-engine.sh LAYOUT NODE SELF_IP MASTER_IP NETWORK_IF SPECULATION OUTPUT_DIR
#   LAYOUT       dep16 (attention DP16, MoE EP16 over two nodes) or dep8 (one node: attention DP8, MoE EP8)
#   NODE         index of this node, 0 or 1; node 0 hosts the frontend
#   SELF_IP      this node's address;  MASTER_IP  node 0's address
#   NETWORK_IF   routed interface to the peer node, for the NCCL and Gloo bootstrap
#   SPECULATION  none or dspark (the gold standard uses dspark)
# Environment: the variables of env.sh, plus
#   OPT_MODEL           model directory
#   MEGA_MOE_TRANSPORT  all_gin (the gold standard), all_lsa, lsa_where_reachable or auto
#   OPT_MOE_BACKEND     optional: mega_moe (default; FlashInfer's persistent kernel) or mega_moe_split (FlashInfer's
#                       split kernels for the target model; the DSpark draft model, whose top-k is odd, stays on
#                       mega_moe). mega_moe_split needs a FlashInfer that ships the split kernels (see README.md).
#   OPT_EXTRA_ENV       optional: a file of ../switch-sets, sourced before the engine starts (unset = baseline)
#   OPT_EXTRA_ARGS      optional: extra `ts serve` arguments ("--max-total-tokens 393216" = the larger KV pool)
# PDL stays on: do not add --disable-pdl (see README.md).
set -euo pipefail
layout=${1:?dep16 or dep8}
node=${2:?physical node index}
self_ip=${3:?local address}
master_ip=${4:?frontend address}
network_if=${5:?routed interface to the peer node}
speculation=${6:?none or dspark}
out=${7:?case output directory}
transport=${MEGA_MOE_TRANSPORT:?all_gin, all_lsa, lsa_where_reachable or auto}
case "${OPT_MOE_BACKEND:-mega_moe}" in
    mega_moe) moe_backend=(--moe-backend mega_moe) ;;
    mega_moe_split) moe_backend=(--moe-backend mega_moe_split --draft-moe-backend mega_moe) ;;
    *) echo "OPT_MOE_BACKEND must be mega_moe or mega_moe_split" >&2; exit 2 ;;
esac
: "${OPT_MODEL:?model directory}"
[[ "$node" == 0 || "$node" == 1 ]]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"
# The kernel overrides used in every run of this branch.
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_SHIFTED_POST_PRE_NORM=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton
export TOKENSPEED_KERNEL_OVERRIDE_MOE_TOPK=triton TS_DISABLE_FUSED_TOPK_TOPP=1
export TOKENSPEED_HOST_IP="$self_ip" HOST_IP="$self_ip"
export NCCL_SOCKET_IFNAME="$network_if" GLOO_SOCKET_IFNAME="$network_if"
export NCCL_DEBUG=INFO
unset NCCL_IB_HCA NCCL_IB_DISABLE
case "$layout" in
    dep16)
        topology=(--world-size 16 --nprocs-per-node 8 --nnodes 2 --node-rank "$node"
            --tensor-parallel-size 1 --data-parallel-size 16 --dense-tp-size 1
            --moe-tp-size 1 --expert-parallel-size 16
            --dist-init-addr "$master_ip:29600")
        handshake=29510
        ;;
    dep8)
        topology=(--world-size 8 --nprocs-per-node 8 --nnodes 1 --node-rank 0
            --tensor-parallel-size 1 --data-parallel-size 8 --dense-tp-size 1
            --moe-tp-size 1 --expert-parallel-size 8
            --dist-init-addr "$self_ip:29600")
        handshake=$((29510 + node * 10))
        ;;
    *) exit 2 ;;
esac
# max-num-seqs is global: divide by attention DP size for per-rank capacity.
# Both layouts use 24 slots and 262144 KV tokens per DP replica.
# The MegaMoE row capacity covers one prefill chunk per rank.
case "$layout" in
    dep16) max_num_seqs=384 ;;
    dep8) max_num_seqs=192 ;;
esac
if [[ -n "${OPT_EXTRA_ENV:-}" ]]; then source "$OPT_EXTRA_ENV"; fi
command=(ts serve --headless --model "$OPT_MODEL" --served-model-name deepseek-v41-flash
    --sampling-backend greedy --dtype bfloat16
    "${moe_backend[@]}" --mega-moe-transport "$transport"
    --deepseek-v4-mega-moe-max-num-tokens 2048
    --engram-host-table --engram-host-table-layout shared
    --engram-host-table-dir /dev/shm
    --max-model-len 131072 --gpu-memory-utilization 0.98
    --max-total-tokens 262144 --max-num-seqs "$max_num_seqs"
    --chunked-prefill-size 2048 --max-cudagraph-capture-size 24
    --cudagraph-capture-sizes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    --disable-prefill-graph --disable-kvstore --trust-remote-code
    --host "$self_ip" --port 19445
    --data-parallel-address "$master_ip" --data-parallel-rpc-port "$handshake"
    --zmq-engine-index 0 "${topology[@]}")
case "$speculation" in
    none) ;;
    dspark)
        command+=(--speculative-algorithm DSPARK --speculative-num-steps 5
            --speculative-num-draft-tokens 6 --hf-overrides '{"dspark_block_size":5}'
            --synthetic-acceptance-length 5.35)
        ;;
    *) exit 2 ;;
esac
if [[ -n "${OPT_EXTRA_ARGS:-}" ]]; then read -r -a _extra <<< "$OPT_EXTRA_ARGS"; command+=("${_extra[@]}"); fi
printf '%q ' "${command[@]}" > "$out/node-$node-command.txt"
printf '\n' >> "$out/node-$node-command.txt"
exec "${command[@]}"
