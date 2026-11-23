# Source before starting an engine of this branch. The site values come from the caller:
#   OPT_TREE            root of a checkout of this branch; python/ and tokenspeed-kernel/python/ are used from source
#   OPT_VENV            Python environment of the serving stack (ours: Python 3.12, torch 2.14.0+cu130)
#   OPT_FLASHINFER_SRC  FlashInfer source checkout, placed before any installed FlashInfer (ours: upstream 656eaee45)
#   OPT_GPUS            optional: GPU indices of this node (default 0,1,2,3,4,5,6,7)
#   OPT_CACHE           optional: directory for the Hugging Face, Triton, torch-extension and CUDA caches
: "${OPT_TREE:?root of a checkout of this branch}"
: "${OPT_VENV:?Python environment of the serving stack}"
: "${OPT_FLASHINFER_SRC:?FlashInfer source checkout}"
if [[ ! -d $OPT_TREE/python/tokenspeed || ! -d $OPT_TREE/tokenspeed-kernel/python/tokenspeed_kernel ]]; then
    echo "OPT_TREE is not a checkout of this branch: $OPT_TREE" >&2
    return 2 2>/dev/null || exit 2
fi
source "$OPT_VENV/bin/activate"
export FLASHINFER_SRC=$OPT_FLASHINFER_SRC
export PYTHONPATH=$FLASHINFER_SRC:$OPT_TREE/python:$OPT_TREE/tokenspeed-kernel/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${OPT_GPUS:-0,1,2,3,4,5,6,7}
if [[ -n "${OPT_CACHE:-}" ]]; then
    export HF_HOME=$OPT_CACHE/hf-cache XDG_CACHE_HOME=$OPT_CACHE/cache
    export TORCH_EXTENSIONS_DIR=$OPT_CACHE/torch-extensions
    export TRITON_CACHE_DIR=$OPT_CACHE/triton-cache
    export CUDA_CACHE_PATH=$OPT_CACHE/cuda-cache
fi
