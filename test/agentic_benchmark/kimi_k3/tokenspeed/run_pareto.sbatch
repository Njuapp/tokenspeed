#!/usr/bin/bash

# Submit once per topology, for example:
#   sbatch --export=ALL,TS_BENCH_CONFIG=attn_tp8_moe_tp8,\
#       KIMI_K3_CONTAINER_IMAGE=/scratch/fsw/.../tokenspeed.sqsh,\
#       KIMI_K3_KERNEL_BUILD_DIR=/scratch/fsw/.../tokenspeed-kernel-build,\
#       KIMI_K3_RUNTIME_ENV_TAR=/scratch/fsw/.../tokenspeed-runtime.tar,\
#       KIMI_K3_RUNTIME_DIR=/tmp/tokenspeed-runtime,\
#       KIMI_K3_BUILD_RUNTIME=1,\
#       KIMI_K3_BUILT_RUNTIME_TAR=/scratch/fsw/.../tokenspeed-current.tar,\
#       KIMI_K3_MODEL_DIR=/scratch/fsw/.../Kimi-K3-NVFP4,\
#       KIMI_K3_DRAFT_DIR=/scratch/fsw/.../Kimi-K3-DSpark,\
#       KIMI_K3_DATASET=/scratch/fsw/.../agentic_dataset.json,\
#       KIMI_K3_OUTPUT_ROOT=/scratch/fsw/.../agentic-pareto run_pareto.sbatch

#SBATCH --job-name=k3-ts-agentic-pareto
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TS_BENCH_CONFIG:?Set TS_BENCH_CONFIG to attn_tp8_moe_tp8, attn_tp8_moe_ep8, or attn_dp8_moe_ep8}"
: "${KIMI_K3_CONTAINER_IMAGE:?Set KIMI_K3_CONTAINER_IMAGE to the TokenSpeed SQSH path}"
: "${KIMI_K3_MODEL_DIR:?Set KIMI_K3_MODEL_DIR to the Kimi-K3-NVFP4 checkpoint}"
: "${KIMI_K3_DRAFT_DIR:?Set KIMI_K3_DRAFT_DIR to the Kimi-K3-DSpark checkpoint}"
: "${KIMI_K3_DATASET:?Set KIMI_K3_DATASET to agentic_dataset.json}"
: "${KIMI_K3_OUTPUT_ROOT:?Set KIMI_K3_OUTPUT_ROOT to a shared result directory}"

preflight_error() {
    echo "preflight error: $1" >&2
    exit 1
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SBATCH_SCRIPT=$(
        scontrol show job -o "$SLURM_JOB_ID" |
            sed -n 's/.* Command=\([^ ]*\).*/\1/p'
    )
else
    SBATCH_SCRIPT=${BASH_SOURCE[0]}
fi
[[ -n "$SBATCH_SCRIPT" ]] || preflight_error "could not resolve the submitted script"
SBATCH_SCRIPT=$(readlink -f "$SBATCH_SCRIPT")
[[ -r "$SBATCH_SCRIPT" ]] || preflight_error "submitted script is unavailable"
BENCH_DIR=$(cd "$(dirname "$SBATCH_SCRIPT")" && pwd)
REPO_ROOT=$(cd "$BENCH_DIR/../../../.." && pwd)
VALIDATOR="$BENCH_DIR/../validate_evalscope.py"
COLLECTOR="$BENCH_DIR/collect_outputs.py"
TS_BENCH_MODE=${TS_BENCH_MODE:-full}
KIMI_K3_CONTAINER_MOUNTS=${KIMI_K3_CONTAINER_MOUNTS:-/scratch/fsw:/scratch/fsw}
KIMI_K3_EVALSCOPE_BIN=${KIMI_K3_EVALSCOPE_BIN:-evalscope}
KIMI_K3_CLIENT_ENV_TAR=${KIMI_K3_CLIENT_ENV_TAR:-}
KIMI_K3_KERNEL_BUILD_DIR=${KIMI_K3_KERNEL_BUILD_DIR:-}
KIMI_K3_RUNTIME_ENV_TAR=${KIMI_K3_RUNTIME_ENV_TAR:-}
KIMI_K3_RUNTIME_DIR=${KIMI_K3_RUNTIME_DIR:-/tmp/tokenspeed-runtime}
KIMI_K3_BUILD_RUNTIME=${KIMI_K3_BUILD_RUNTIME:-0}
KIMI_K3_BUILT_RUNTIME_TAR=${KIMI_K3_BUILT_RUNTIME_TAR:-}
KIMI_K3_DATASET_SHA256=${KIMI_K3_DATASET_SHA256:-2405552a4f320bbaaf084a0f25f57dc1c9af44e69612d0c9d098a137db3463f2}
KIMI_K3_DP_LOAD_BALANCE=${KIMI_K3_DP_LOAD_BALANCE:-shortest_queue}

[[ "$SLURM_NNODES" -eq 2 ]] || preflight_error "expected two Slurm nodes"
[[ -s "$KIMI_K3_CONTAINER_IMAGE" ]] || preflight_error "container image is unavailable"
[[ -d "$KIMI_K3_MODEL_DIR" ]] || preflight_error "model directory is unavailable"
[[ -d "$KIMI_K3_DRAFT_DIR" ]] || preflight_error "draft directory is unavailable"
[[ -s "$KIMI_K3_DATASET" ]] || preflight_error "dataset is unavailable"
[[ -f "$VALIDATOR" ]] || preflight_error "request validator is unavailable"
[[ -f "$COLLECTOR" ]] || preflight_error "output collector is unavailable"
if [[ -n "$KIMI_K3_CLIENT_ENV_TAR" ]]; then
    [[ -s "$KIMI_K3_CLIENT_ENV_TAR" ]] || preflight_error "client environment is unavailable"
fi
if [[ -n "$KIMI_K3_KERNEL_BUILD_DIR" ]]; then
    [[ -d "$KIMI_K3_KERNEL_BUILD_DIR/python/tokenspeed_kernel" ]] || \
        preflight_error "TokenSpeed kernel build is unavailable"
    kernel_module_count=$(
        find "$KIMI_K3_KERNEL_BUILD_DIR/python/tokenspeed_kernel" \
            -type f -name '*.so' | wc -l
    )
    [[ "$kernel_module_count" -ge 19 ]] || \
        preflight_error "TokenSpeed kernel build has fewer than 19 native modules"
fi
if [[ -n "$KIMI_K3_RUNTIME_ENV_TAR" ]]; then
    [[ -s "$KIMI_K3_RUNTIME_ENV_TAR" ]] || \
        preflight_error "TokenSpeed server runtime archive is unavailable"
    [[ "$KIMI_K3_RUNTIME_DIR" == /tmp/tokenspeed-*-runtime ]] || \
        preflight_error "KIMI_K3_RUNTIME_DIR must be a specific /tmp/tokenspeed-*-runtime path"
fi
[[ "$KIMI_K3_BUILD_RUNTIME" =~ ^[01]$ ]] || \
    preflight_error "KIMI_K3_BUILD_RUNTIME must be 0 or 1"
if [[ "$KIMI_K3_BUILD_RUNTIME" == 1 ]]; then
    [[ -n "$KIMI_K3_RUNTIME_ENV_TAR" ]] || \
        preflight_error "KIMI_K3_BUILD_RUNTIME=1 needs a base KIMI_K3_RUNTIME_ENV_TAR"
    [[ -n "$KIMI_K3_BUILT_RUNTIME_TAR" ]] || \
        preflight_error "KIMI_K3_BUILD_RUNTIME=1 needs KIMI_K3_BUILT_RUNTIME_TAR"
    [[ "$KIMI_K3_BUILT_RUNTIME_TAR" != "$KIMI_K3_RUNTIME_ENV_TAR" ]] || \
        preflight_error "built runtime must not overwrite the base runtime"
fi
dataset_sha256=$(sha256sum "$KIMI_K3_DATASET" | awk '{print $1}')
[[ "$dataset_sha256" == "$KIMI_K3_DATASET_SHA256" ]] || \
    preflight_error "dataset checksum does not match KIMI_K3_DATASET_SHA256"

case "$TS_BENCH_MODE" in
    smoke)
        CLIENT_NUMBERS="4"
        CLIENT_PARALLELS="1"
        ;;
    full)
        CLIENT_NUMBERS="4 8 8 16 32"
        CLIENT_PARALLELS="1 2 4 8 16"
        ;;
    *)
        echo "TS_BENCH_MODE must be smoke or full" >&2
        exit 2
        ;;
esac

case "$TS_BENCH_CONFIG" in
    attn_tp8_moe_tp8)
        PARALLEL_ARGS=(--attn-tp-size 8 --moe-tp-size 8)
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-8192}
        ;;
    attn_tp8_moe_ep8)
        PARALLEL_ARGS=(--attn-tp-size 8 --ep-size 8)
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-8192}
        ;;
    attn_dp8_moe_ep8)
        PARALLEL_ARGS=(
            --data-parallel-size 8
            --ep-size 8
            --moe-tp-size 1
            --load-balance-method "$KIMI_K3_DP_LOAD_BALANCE"
        )
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-2048}
        ;;
    *)
        echo "unknown TS_BENCH_CONFIG=$TS_BENCH_CONFIG" >&2
        exit 2
        ;;
esac

RESULT_CONFIG=$TS_BENCH_CONFIG

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
[[ -n "$MASTER_ADDR" ]] || preflight_error "could not resolve the Slurm master node"
DIST_PORT=${KIMI_K3_DIST_PORT:-7500}
HTTP_PORT=${KIMI_K3_HTTP_PORT:-8000}
RUN_DIR="$KIMI_K3_OUTPUT_ROOT/tokenspeed/$RESULT_CONFIG/$SLURM_JOB_ID"
SERVER_LOG="$RUN_DIR/server.log"
mkdir -p "$RUN_DIR/preflight" "$RUN_DIR/results"

export MODEL_DIR=$KIMI_K3_MODEL_DIR
export DRAFT_DIR=$KIMI_K3_DRAFT_DIR
export DATASET=$KIMI_K3_DATASET
export EVALSCOPE_BIN=$KIMI_K3_EVALSCOPE_BIN
export CLIENT_ENV_TAR=$KIMI_K3_CLIENT_ENV_TAR
export KERNEL_BUILD_DIR=$KIMI_K3_KERNEL_BUILD_DIR
export SERVER_RUNTIME_TAR=$KIMI_K3_RUNTIME_ENV_TAR
export SERVER_RUNTIME_DIR=$KIMI_K3_RUNTIME_DIR
export BUILD_SERVER_RUNTIME=$KIMI_K3_BUILD_RUNTIME
export BUILT_SERVER_RUNTIME_TAR=$KIMI_K3_BUILT_RUNTIME_TAR
export MASTER_ADDR DIST_PORT HTTP_PORT PREFILL_TOKENS RESULT_CONFIG
export TS_BENCH_CONFIG TS_BENCH_MODE KIMI_K3_DP_LOAD_BALANCE
export CLIENT_NUMBERS CLIENT_PARALLELS
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_DMABUF_ENABLE=0
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y
export UCX_MEMTYPE_CACHE=n
export UCX_MEMTYPE_REG_WHOLE=n
export ENROOT_CACHE_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-cache"
export ENROOT_DATA_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-data"
export ENROOT_RUNTIME_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-runtime"

CONTAINER_ARGS=(
    --container-image="$KIMI_K3_CONTAINER_IMAGE"
    --container-mounts="$KIMI_K3_CONTAINER_MOUNTS"
    --container-workdir="$REPO_ROOT"
    --container-env=MODEL_DIR,DRAFT_DIR,DATASET,EVALSCOPE_BIN,CLIENT_ENV_TAR,KERNEL_BUILD_DIR,SERVER_RUNTIME_TAR,SERVER_RUNTIME_DIR,BUILD_SERVER_RUNTIME,BUILT_SERVER_RUNTIME_TAR,MASTER_ADDR,DIST_PORT,HTTP_PORT,PREFILL_TOKENS,RESULT_CONFIG,TS_BENCH_CONFIG,TS_BENCH_MODE,KIMI_K3_DP_LOAD_BALANCE,CLIENT_NUMBERS,CLIENT_PARALLELS,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NVLS_ENABLE,NCCL_DMABUF_ENABLE,UCX_TLS,UCX_CUDA_IPC_ENABLE_MNNVL,UCX_MEMTYPE_CACHE,UCX_MEMTYPE_REG_WHOLE
)

{
    scontrol show config | grep '^ClusterName'
    scontrol show job -d "$SLURM_JOB_ID"
    git -C "$REPO_ROOT" rev-parse HEAD
    git -C "$REPO_ROOT" status --short
    printf 'source_diff_sha256='
    git -C "$REPO_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}'
    sha256sum "$KIMI_K3_CONTAINER_IMAGE" "$KIMI_K3_DATASET" \
        "$SBATCH_SCRIPT" "$VALIDATOR" "$COLLECTOR"
    printf 'config=%s\nmode=%s\nmaster=%s:%s\nprefill_tokens=%s\nmodel=%s\ndraft=%s\nkernel_build=%s\nruntime_tar=%s\nruntime_dir=%s\nbuild_runtime=%s\nbuilt_runtime_tar=%s\ndataset_sha256=%s\nnumber=%s\nparallel=%s\n' \
        "$TS_BENCH_CONFIG" "$TS_BENCH_MODE" "$MASTER_ADDR" "$DIST_PORT" \
        "$PREFILL_TOKENS" "$KIMI_K3_MODEL_DIR" "$KIMI_K3_DRAFT_DIR" \
        "$KIMI_K3_KERNEL_BUILD_DIR" "$KIMI_K3_RUNTIME_ENV_TAR" \
        "$KIMI_K3_RUNTIME_DIR" "$KIMI_K3_BUILD_RUNTIME" \
        "$KIMI_K3_BUILT_RUNTIME_TAR" "$KIMI_K3_DATASET_SHA256" \
        "$CLIENT_NUMBERS" "$CLIENT_PARALLELS"
    printf 'parallel_args:'
    printf ' %q' "${PARALLEL_ARGS[@]}"
    printf '\n'
} > "$RUN_DIR/preflight/manifest.txt"

if [[ "$KIMI_K3_BUILD_RUNTIME" == 1 ]]; then
    mkdir -p "$(dirname "$KIMI_K3_BUILT_RUNTIME_TAR")"
    rm -f "$KIMI_K3_BUILT_RUNTIME_TAR"
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        "${CONTAINER_ARGS[@]}" \
        bash -lc '
            set -euo pipefail
            build_root="/tmp/k3-ts-runtime-build-'$SLURM_JOB_ID'"
            rm -rf "$build_root" "$SERVER_RUNTIME_DIR"
            mkdir -p "$build_root" "$SERVER_RUNTIME_DIR"
            tar -C "$SERVER_RUNTIME_DIR" --strip-components=1 \
                -xf "$SERVER_RUNTIME_TAR"
            build_python="$SERVER_RUNTIME_DIR/bin/python"
            [[ -x "$build_python" ]]
            site_packages="$SERVER_RUNTIME_DIR/lib/python3.12/site-packages"
            [[ -d "$site_packages" ]]
            stale_cutlass_metadata=(
                "$site_packages/nvidia_cutlass_dsl-4.6.0.dist-info"
                "$site_packages/nvidia_cutlass_dsl_libs_base-4.6.0.dist-info"
                "$site_packages/nvidia_cutlass_dsl_libs_core-4.6.0.dist-info"
                "$site_packages/nvidia_cutlass_dsl_libs_cu12-4.6.0.dist-info"
                "$site_packages/nvidia_cutlass_dsl_libs_cu13-4.6.0.dist-info"
            )
            for stale_metadata in "${stale_cutlass_metadata[@]}"; do
                [[ "$stale_metadata" == "$site_packages"/*-4.6.0.dist-info ]]
                rm -rf "$stale_metadata"
            done

            cp -a "'$REPO_ROOT'/tokenspeed-kernel/python" \
                "$build_root/kernel-python"
            mapfile -t runtime_specs < <(
                sed -n \
                    '\''s/^[[:space:]]*"\(tokenspeed-\(mooncake\|scheduler\|smg[^" ]*\)[^"]*\)",/\1/p'\'' \
                    "'$REPO_ROOT'/python/pyproject.toml"
            )
            ((${#runtime_specs[@]} > 0))

            export TOKENSPEED_KERNEL_BACKEND=cuda
            export TOKENSPEED_CUDA_ARCH=103
            export FLASHINFER_CUDA_ARCH_LIST=10.3a
            export MAX_JOBS="${KIMI_K3_KERNEL_BUILD_JOBS:-32}"
            "$build_python" -m pip install --upgrade "${runtime_specs[@]}"
            "$build_python" -m pip install --upgrade \
                -r "$build_root/kernel-python/requirements/cuda-thirdparty.txt"
            "$build_python" -c \
                '\''import importlib.metadata as m; v = m.version("nvidia-cutlass-dsl"); print("nvidia-cutlass-dsl", v); assert v == "4.7.1", v'\''
            "$build_python" -m pip install --force-reinstall --no-deps \
                --no-build-isolation "$build_root/kernel-python"

            PYTHONPATH="'$REPO_ROOT'/python" "$build_python" -c \
                '\''import tokenspeed_kernel; from tokenspeed_kernel.platform import pdl_enabled; print(tokenspeed_kernel.__file__); print("pdl_enabled", pdl_enabled())'\''
            native_count=$(find "$SERVER_RUNTIME_DIR/lib" -path \
                '\''*/tokenspeed_kernel/*'\'' -name '\''*.so'\'' -type f | wc -l)
            ((native_count >= 19))
            "$build_python" -m pip freeze \
                > "$BUILT_SERVER_RUNTIME_TAR.pip-freeze.txt"

            runtime_name=$(basename "$SERVER_RUNTIME_DIR")
            tmp_tar="$BUILT_SERVER_RUNTIME_TAR.tmp-'$SLURM_JOB_ID'"
            tar -C /tmp -cf "$tmp_tar" "$runtime_name"
            mv "$tmp_tar" "$BUILT_SERVER_RUNTIME_TAR"
            printf "runtime=%s native_modules=%s\n" \
                "$BUILT_SERVER_RUNTIME_TAR" "$native_count"
        ' > "$RUN_DIR/preflight/runtime-build.log" 2>&1
    [[ -s "$KIMI_K3_BUILT_RUNTIME_TAR" ]] || \
        preflight_error "current-source TokenSpeed runtime build did not produce an archive"
    SERVER_RUNTIME_TAR=$KIMI_K3_BUILT_RUNTIME_TAR
    KERNEL_BUILD_DIR=
    export SERVER_RUNTIME_TAR KERNEL_BUILD_DIR
    sha256sum "$SERVER_RUNTIME_TAR" \
        > "$RUN_DIR/preflight/built-runtime.sha256"
fi

SERVER_STEP_PID=
stop_server() {
    if [[ -n "$SERVER_STEP_PID" ]]; then
        kill -TERM "$SERVER_STEP_PID" 2>/dev/null || true
        wait "$SERVER_STEP_PID" 2>/dev/null || true
    fi
}
trap stop_server EXIT

srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --label \
    "${CONTAINER_ARGS[@]}" \
    bash -lc '
        set -euo pipefail
        cd "'$REPO_ROOT'"
        server_python=python3
        if [[ -n "$SERVER_RUNTIME_TAR" ]]; then
            rm -rf "$SERVER_RUNTIME_DIR"
            tar -C /tmp -xf "$SERVER_RUNTIME_TAR"
            [[ -x "$SERVER_RUNTIME_DIR/bin/python" ]]
            server_python="$SERVER_RUNTIME_DIR/bin/python"
            export PATH="$SERVER_RUNTIME_DIR/bin:$PATH"
            export PYTHONNOUSERSITE=1
        fi
        export PYTHONPATH="'$REPO_ROOT'/python${KERNEL_BUILD_DIR:+:$KERNEL_BUILD_DIR/python}${PYTHONPATH:+:$PYTHONPATH}"
        export TOKENIZERS_PARALLELISM=false
        export TOKENSPEED_CUDA_ARCH=103
        export TORCH_EXTENSIONS_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/torch-extensions"
        export TRITON_CACHE_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/triton-cache"
        export FLASHINFER_WORKSPACE_BASE="/tmp/k3-ts-'$SLURM_JOB_ID'/flashinfer-workspace"
        export FLASHINFER_CUBIN_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/flashinfer-cubins"
        mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
            "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_CUBIN_DIR"
        case "$TS_BENCH_CONFIG" in
            attn_tp8_moe_tp8)
                parallel_args=(--attn-tp-size 8 --moe-tp-size 8)
                ;;
            attn_tp8_moe_ep8)
                parallel_args=(--attn-tp-size 8 --ep-size 8)
                ;;
            attn_dp8_moe_ep8)
                parallel_args=(
                    --data-parallel-size 8
                    --ep-size 8
                    --moe-tp-size 1
                    --load-balance-method "$KIMI_K3_DP_LOAD_BALANCE"
                )
                ;;
        esac
        exec "$server_python" -m tokenspeed.cli serve \
            --model "$MODEL_DIR" \
            --served-model-name nvidia/Kimi-K3-NVFP4 \
            "${parallel_args[@]}" \
            --max-model-len 80000 \
            --max-num-seqs 16 \
            --max-prefill-tokens "$PREFILL_TOKENS" \
            --chunked-prefill-size "$PREFILL_TOKENS" \
            --gpu-memory-utilization 0.92 \
            --trust-remote-code \
            --attention-backend tokenspeed_mla \
            --kda-backend cutedsl_kda \
            --moe-backend flashinfer_trtllm \
            --kv-cache-dtype fp8 \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_DIR" \
            --speculative-num-draft-tokens 8 \
            --drafter-attention-backend tokenspeed_mla \
            --mm-encoder-tp-mode data \
            --enable-cache-report \
            --disable-weight-loader-prefetch-checkpoints \
            --dist-init-addr "$MASTER_ADDR:$DIST_PORT" \
            --host 0.0.0.0 \
            --port "$HTTP_PORT" \
            --engine-startup-timeout 7200
    ' > "$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

start=$SECONDS
until curl -sf -o /dev/null "http://$MASTER_ADDR:$HTTP_PORT/readiness"; do
    if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    if grep -Eqi 'CUDA out of memory|torch\.OutOfMemoryError|CUDA error:|device-side assert|NCCL.*(error|fail)|Traceback \(most recent call last\)' "$SERVER_LOG"; then
        echo "fatal TokenSpeed server marker before readiness" >&2
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS - start > 7200 )); then
        echo "timeout waiting for TokenSpeed readiness" >&2
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done

srun --overlap --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
    "${CONTAINER_ARGS[@]}" \
    bash -lc '
        set -euo pipefail
        client_python=python3
        evalscope_bin="$EVALSCOPE_BIN"
        if [[ -n "$CLIENT_ENV_TAR" ]]; then
            client_dir="/tmp/k3-agentic-client-'$SLURM_JOB_ID'"
            rm -rf "$client_dir"
            mkdir -p "$client_dir"
            tar -C "$client_dir" --strip-components=1 -xf "$CLIENT_ENV_TAR"
            client_python="$client_dir/bin/python"
            evalscope_bin="$client_dir/bin/evalscope"
        fi
        command -v "$evalscope_bin" >/dev/null
        "$evalscope_bin" perf \
            --model nvidia/Kimi-K3-NVFP4 \
            --url "http://$MASTER_ADDR:$HTTP_PORT/v1/chat/completions" \
            --api openai \
            --dataset swe_smith \
            --dataset-path "$DATASET" \
            --max-tokens 500 \
            --multi-turn \
            --number 2 \
            --parallel 2 \
            --extra-args '\''{"ignore_eos": true}'\'' \
            --dataset-offset 68 \
            --outputs-dir "'$RUN_DIR'/results/warmup"

        read -r -a client_number <<< "$CLIENT_NUMBERS"
        read -r -a client_parallel <<< "$CLIENT_PARALLELS"
        "$evalscope_bin" perf \
            --model nvidia/Kimi-K3-NVFP4 \
            --url "http://$MASTER_ADDR:$HTTP_PORT/v1/chat/completions" \
            --api openai \
            --dataset swe_smith \
            --dataset-path "$DATASET" \
            --max-tokens 500 \
            --multi-turn \
            --number "${client_number[@]}" \
            --parallel "${client_parallel[@]}" \
            --extra-args '\''{"ignore_eos": true}'\'' \
            --name "$RESULT_CONFIG" \
            --outputs-dir "'$RUN_DIR'/results/$TS_BENCH_MODE" \
            --no-timestamp

        "$client_python" "'$VALIDATOR'" \
            "'$RUN_DIR'/results/warmup" \
            "'$RUN_DIR'/results/$TS_BENCH_MODE" \
            --output "'$RUN_DIR'/results/request-integrity.json"
        if [[ "$TS_BENCH_MODE" == full ]]; then
            "$client_python" "'$COLLECTOR'" \
                "'$RUN_DIR'/results/full" \
                -o "'$RUN_DIR'/results/summary.csv"
        fi
    '

if grep -Eqi 'CUDA out of memory|torch\.OutOfMemoryError|CUDA error:|device-side assert|NCCL.*(error|fail)|Traceback \(most recent call last\)' "$SERVER_LOG"; then
    echo "fatal TokenSpeed server marker found" >&2
    exit 3
fi

stop_server
SERVER_STEP_PID=
echo "$TS_BENCH_MODE complete: $RUN_DIR"
