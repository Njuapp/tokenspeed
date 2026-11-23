#!/usr/bin/env bash
# Decode engine for step-time measurements at a fixed number of requests per rank, as measured. Run once per node.
# usage: decode-step-engine.sh LAYOUT NODE SELF_IP MASTER_IP NETWORK_IF MAX_BATCH OUTPUT_DIR
#   LAYOUT      dep16 (attention DP16, MoE EP16 over two nodes) or dep8 (one node: attention DP8, MoE EP8)
#   NODE        index of this node, 0 or 1; node 0 hosts the frontend
#   SELF_IP     this node's address;  MASTER_IP  node 0's address
#   NETWORK_IF  routed interface to the peer node, for the NCCL and Gloo bootstrap
#   MAX_BATCH   largest number of requests per rank that will be measured (32 in our runs)
# What differs from decode-engine.sh, and why:
#   --max-model-len 135168        a context of 131072 tokens plus the output
#   no --sampling-backend greedy  the requests sample (temperature and top-p), so the default sampling backend runs
#   fixed acceptance length 6     through TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN instead of --synthetic-acceptance-length
#   KV pool and slots             sized for MAX_BATCH long requests per rank
#   CUDA graph capture sizes      the powers of two up to MAX_BATCH
# Environment: the variables of env.sh, plus
#   OPT_MODEL             model directory
#   OPT_EXTRA_ENV         optional: a file of ../switch-sets, sourced before the engine starts (unset = baseline)
#   OPT_EXTRA_ARGS        optional: extra `ts serve` arguments
#   OPT_MAX_TOTAL_TOKENS  optional (default 786432);  MEGA_MOE_TRANSPORT  optional (default all_gin)
# Put the step observer of your benchmark on PYTHONPATH as you would for any other engine.
# PDL stays on: do not add --disable-pdl (see README.md).
set -euo pipefail
layout=${1:?dep16 or dep8}
node=${2:?physical node index}
self_ip=${3:?local address}
master_ip=${4:?frontend address}
network_if=${5:?routed interface to the peer node}
maxb=${6:?largest number of requests per rank}
out=${7:?output directory}
: "${OPT_MODEL:?model directory}"
[[ "$node" == 0 || "$node" == 1 ]]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_SHIFTED_POST_PRE_NORM=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE=triton
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton
export TOKENSPEED_KERNEL_OVERRIDE_MOE_TOPK=triton TS_DISABLE_FUSED_TOPK_TOPP=1
export TOKENSPEED_HOST_IP="$self_ip" HOST_IP="$self_ip"
export NCCL_SOCKET_IFNAME="$network_if" GLOO_SOCKET_IFNAME="$network_if" NCCL_DEBUG=INFO
unset NCCL_IB_HCA NCCL_IB_DISABLE
if [[ -n "${OPT_EXTRA_ENV:-}" ]]; then source "$OPT_EXTRA_ENV"; fi
export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=6
case "$layout" in
    dep16)
        ranks=16
        topology=(--world-size 16 --nprocs-per-node 8 --nnodes 2 --node-rank "$node"
            --tensor-parallel-size 1 --data-parallel-size 16 --dense-tp-size 1 --moe-tp-size 1
            --expert-parallel-size 16 --dist-init-addr "$master_ip:32054")
        ;;
    dep8)
        ranks=8
        topology=(--world-size 8 --nprocs-per-node 8 --nnodes 1 --node-rank 0
            --tensor-parallel-size 1 --data-parallel-size 8 --dense-tp-size 1 --moe-tp-size 1
            --expert-parallel-size 8 --dist-init-addr "$self_ip:32054")
        ;;
    *) exit 2 ;;
esac
sizes=()
for size in 1 2 4 8 16 32 64; do
    if (( size <= maxb )); then sizes+=("$size"); fi
done
command=(ts serve --headless --model "$OPT_MODEL" --served-model-name deepseek-v41-flash --dtype bfloat16
    --moe-backend mega_moe --mega-moe-transport "${MEGA_MOE_TRANSPORT:-all_gin}"
    --deepseek-v4-mega-moe-max-num-tokens 2048
    --engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm
    --max-model-len 135168 --gpu-memory-utilization 0.98
    --max-total-tokens "${OPT_MAX_TOTAL_TOKENS:-786432}" --max-num-seqs $((ranks * maxb))
    --chunked-prefill-size 2048 --max-cudagraph-capture-size "$maxb" --cudagraph-capture-sizes "${sizes[@]}"
    --disable-prefill-graph --disable-kvstore --trust-remote-code
    --host "$self_ip" --port 19445
    --data-parallel-address "$master_ip" --data-parallel-rpc-port 32051 --zmq-engine-index 0
    "${topology[@]}"
    --speculative-algorithm DSPARK --speculative-num-steps 5 --speculative-num-draft-tokens 6
    --hf-overrides '{"dspark_block_size":5}' --decode-log-interval 100000)
if [[ -n "${OPT_EXTRA_ARGS:-}" ]]; then read -r -a _extra <<< "$OPT_EXTRA_ARGS"; command+=("${_extra[@]}"); fi
printf '%q ' "${command[@]}" > "$out/node-$node-command.txt"
printf '\n' >> "$out/node-$node-command.txt"
exec "${command[@]}"
