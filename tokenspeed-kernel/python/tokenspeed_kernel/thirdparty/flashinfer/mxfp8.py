# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Keep MXFP8 persistent narrow tiles within one N tile.

FlashInfer 0.7.1rc2 relaxed this restriction from 0.7.0, admitting MXFP8
candidates that fault during autotuning. Keep small-M swapped and split-K
candidates, with the same check when a cached tactic serves another shape.
"""

from __future__ import annotations

import functools
import inspect
import os
import types

from flashinfer.gemm import gemm_base

# Experiment switch (off by default; unset, this module is the base module).
# The SM120 b12x tactic is otherwise the one FlashInfer's autotuner cached for
# the row bucket: 65..128 rows share the 128 bucket, profiled at 128 rows with
# L2-hot weights. With cold weights and 65..96 rows the (32, 128) tile (three
# M tiles) is faster on the wide projections and bitwise identical; from 97
# rows it needs a fourth M tile and is slower, so those keep the cached tactic.
_SM120_TILE_32X128 = os.environ.get("TOKENSPEED_SM120_MXFP8_TILE_32X128", "0") == "1"
_SM120_TILE_32X128_ROWS = (65, 96)
# (N, K) of the DeepSeek V4.1 projections measured faster with the tile:
# wq_b, wo_b, shared gate_up, shared down, indexer wq_b, Engram wkv, DSpark
# main_proj. wq_a_wkv (1792, 5120) and context_wkv (1536, 5120) are not.
_SM120_TILE_32X128_SHAPES = frozenset(
    {
        (32768, 1280),
        (5120, 8192),
        (4608, 5120),
        (5120, 2304),
        (4096, 1280),
        (25600, 6144),
        (5120, 15360),
    }
)
_SM120_TILE_32X128_TACTIC = ((32, 128), (1, 1), False, False, "sm120", None)
# Launch counts for tests; never read on the serving path.
sm120_tile_32x128_stats = {"overridden": 0, "kept": 0, "precompiled": 0}
_sm120_tile_32x128_compiled: set = set()


def has_sm120_mxfp8() -> bool:
    """Whether the installed FlashInfer can compile its SM120 MXFP8 backend."""
    try:
        from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_mxfp8_dsl_supported
    except ImportError:
        return False
    from flashinfer.jit.cpp_ext import get_cuda_version

    return get_cuda_version().major >= 13 and _b12x_mxfp8_dsl_supported()


def _uses_narrow_multi_tile(tactic, inputs) -> bool:
    tile, _, swap_ab, _, split_k_slices = tactic
    kernel_n = inputs[0].shape[0] if swap_ab else inputs[1].shape[1]
    return split_k_slices == 1 and tile[1] < 64 and kernel_n > tile[1]


@functools.cache
def _mxfp8_runner_type(*args, **kwargs):
    upstream = gemm_base._cute_dsl_gemm_mxfp8_runner(*args, **kwargs)

    class TokenspeedMxfp8GemmRunner(type(upstream)):
        def get_cache_key_extras(self, inputs):
            return (*super().get_cache_key_extras(inputs), "single-narrow-tile-v1")

        def get_valid_tactics(self, inputs, profile):
            return [
                tactic
                for tactic in super().get_valid_tactics(inputs, profile)
                if not _uses_narrow_multi_tile(tactic, inputs)
            ]

        def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
            if tactic is not None and tactic != -1:
                if _uses_narrow_multi_tile(tactic, inputs):
                    # Bucketed cache hits must also obey the actual shape's
                    # restriction. Upstream selects a valid default for this M.
                    tactic = -1
            return super().forward(
                inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
            )

    return TokenspeedMxfp8GemmRunner


def _mxfp8_runner(*args, **kwargs):
    return _mxfp8_runner_type(*args, **kwargs)()


def _precompile_sm120_tile_32x128(out_dtype, enable_pdl) -> None:
    """Compile the (32, 128) b12x variant once per device and output dtype.

    The tile is a compile-time parameter; rows, N and K are runtime values.
    Called from the first b12x MXFP8 launch of the process, i.e. from the
    startup warmup forward, so no row count reaches the compiler while
    serving. Compilation is host-side only and launches nothing.
    """
    import torch

    key = (torch.cuda.current_device(), out_dtype, enable_pdl)
    if key in _sm120_tile_32x128_compiled:
        return
    _sm120_tile_32x128_compiled.add(key)
    import cutlass
    from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype
    from flashinfer.gemm import gemm_mm_mxfp8_cute_dsl as fi
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
    )

    tile, cluster, swap_ab, use_prefetch, kernel_type, use_tma_store = (
        _SM120_TILE_32X128_TACTIC
    )
    # Same key and arguments as B12xMxfp8GemmRunner.forward, so the launch
    # below finds this entry instead of compiling.
    cache_key = (
        32,
        tile,
        cluster,
        swap_ab,
        use_prefetch,
        kernel_type,
        use_tma_store,
        enable_pdl,
        out_dtype,
    )
    fi._compile_block_scaled_gemm(
        fi._B12X_MM_MXFP8_KERNEL_CACHE,
        cache_key,
        lambda: Sm120B12xBlockScaledDenseGemmKernel(
            32,
            tile,
            cluster,
            mma_k=32,
            tile_k=128,
            use_prefetch=use_prefetch,
            enable_pdl=enable_pdl,
            swap_ab=swap_ab,
        ),
        ab_cutlass_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_cutlass_dtype=torch_to_cutlass_dtype(out_dtype),
        ab_assumed_align=32,
        cluster_shape_mn=cluster,
        swap_ab=False,
        sf_m=1,
        sf_n=1,
        sf_k=1,
        batch_size=1,
    )
    sm120_tile_32x128_stats["precompiled"] += 1


def _b12x_runner_tile_32x128(factory):
    """Wrap FlashInfer's b12x runner factory with the row-bucket tile override.

    The factory's runner class is subclassed under its own name and only
    ``forward`` is overridden; instances carry no attributes. FlashInfer keys
    its in-memory tuned winners by the runner's class name and ``hash(runner)``
    (a hash of the instance attributes) and its file entries by the class
    name, so both keys equal the unpatched runner's: tuned tactics are found,
    profiled and saved exactly as without the switch, with or without an
    autotune cache file. Only the tactic handed to ``forward`` changes, and
    never while the autotuner is timing candidates.
    """
    from flashinfer.autotuner import is_in_profile_measurement

    low, high = _SM120_TILE_32X128_ROWS

    @functools.cache
    def tiled_type(sm_major, sm_minor, enable_pdl, out_dtype):
        upstream = type(factory(sm_major, sm_minor, enable_pdl, out_dtype))

        def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
            _precompile_sm120_tile_32x128(out_dtype, enable_pdl)
            a, b = inputs[0], inputs[1]
            # a is [M, K]; b is the [K, N] view of the [N, K] weight.
            if (
                low <= a.shape[0] <= high
                and (b.shape[1], a.shape[1]) in _SM120_TILE_32X128_SHAPES
                and not is_in_profile_measurement()
            ):
                tactic = _SM120_TILE_32X128_TACTIC
                sm120_tile_32x128_stats["overridden"] += 1
            else:
                sm120_tile_32x128_stats["kept"] += 1
            return upstream.forward(
                self, inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
            )

        # Same __name__ as upstream: it is part of every autotune cache key.
        return type(
            upstream.__name__,
            (upstream,),
            {
                "forward": forward,
                "__module__": upstream.__module__,
                "__qualname__": upstream.__qualname__,
            },
        )

    def make(sm_major, sm_minor, enable_pdl, out_dtype):
        # A new instance per call, as upstream; no instance state, so its hash
        # is the unpatched runner's.
        return tiled_type(sm_major, sm_minor, enable_pdl, out_dtype)()

    return make


@functools.cache
def _private_mm_mxfp8():
    raw = inspect.unwrap(gemm_base.mm_mxfp8)
    namespace = dict(raw.__globals__)
    namespace["_cute_dsl_gemm_mxfp8_runner"] = _mxfp8_runner
    if _SM120_TILE_32X128 and "_b12x_gemm_mxfp8_runner" in namespace:
        namespace["_b12x_gemm_mxfp8_runner"] = _b12x_runner_tile_32x128(
            namespace["_b12x_gemm_mxfp8_runner"]
        )
    clone = types.FunctionType(
        raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__
    )
    clone.__kwdefaults__ = raw.__kwdefaults__
    clone.__annotations__ = raw.__annotations__
    # Preserve upstream argument validation and backend selection around the
    # private body; only the cute-dsl runner factory is replaced.
    clone = gemm_base.flashinfer_api(trace=gemm_base.mm_mxfp8_trace)(clone)
    clone = gemm_base.backend_requirement(
        {
            "cutlass": gemm_base._cutlass_gemm_mxfp8_requirement,
            "trtllm": gemm_base._trtllm_gemm_mxfp8_requirement,
            "cute-dsl": gemm_base._cute_dsl_gemm_mxfp8_requirement,
            "cutedsl_low_latency": gemm_base._cutedsl_low_latency_gemm_mxfp8_requirement,
            "cudnn": gemm_base._cudnn_mm_mxfp8_requirement,
            "b12x": gemm_base._b12x_gemm_mxfp8_requirement,
        },
        common_check=gemm_base._check_mm_mxfp8_problem_size,
        heuristic_func=gemm_base._heuristic_func_mm_mxfp8,
    )(clone)
    namespace["mm_mxfp8"] = clone
    return clone


def mm_mxfp8(*args, **kwargs):
    return _private_mm_mxfp8()(*args, **kwargs)
