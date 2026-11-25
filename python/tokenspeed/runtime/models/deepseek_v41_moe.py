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

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm import dsv4_linear_fp32
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    TopKOutput,
    simulated_router_logits,
)
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType, get_moe_backend
from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.moe.ced_row_bound import (
    MEGA_MOE_DECODER_ROW_BOUND_ENV,
    mega_moe_decoder_row_bound_requested,
)
from tokenspeed.runtime.moe.expert_file_placement import (
    EXPERT_PLACEMENT_ENV,
    LOAD_MODES,
    LayerLoadCounter,
    expert_load_profile,
    expert_placement_identity,
    expert_placement_slots,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = logging.getLogger(__name__)

# Opt-in experiment switch: "1" makes MegaMoE layers run their shared expert
# inside the fused kernel launch (the kernel's in-kernel MXFP8 shared expert)
# instead of as separate dense GEMMs. Read once per layer, at warmup. Unset or
# any other value keeps the separate path and the routed-only kernel build.
MEGA_MOE_FUSED_SHARED_ENV = "TOKENSPEED_MEGA_MOE_FUSED_SHARED"


def mega_moe_fused_shared_requested() -> bool:
    return os.environ.get(MEGA_MOE_FUSED_SHARED_ENV, "") == "1"


# MoE kernels whose input staging applies plan["expert_id_map"] (logical
# expert id -> placed slot), by the name the kernel plan reports. Every other
# kernel stages the routed ids as they are and would address the slot
# numbered like the logical expert.
_EXPERT_ID_MAP_KERNELS = frozenset(
    {
        "flashinfer_mxfp4_mega_moe_apply",
        "flashinfer_mxfp4_mega_moe_split_apply",
    }
)

# EP process groups whose ranks this process has compared; see
# DeepseekV4MoE._check_ep_group_agreement. The entries keep the groups alive,
# so the identity test stays valid for the life of the process.
_agreed_ep_groups: list[object] = []


def _mxfp8_codes_and_scales(
    linear: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """E4M3 codes ``[out, in]`` and UE8M0 scale bytes ``[out, in // 32]`` of a
    V4.1 MXFP8 linear without bias, or None when the layer holds anything else."""
    params = dict(linear.named_parameters(recurse=False))
    weight = params.get("weight")
    scale = params.get("weight_scale_inv")
    if weight is None or scale is None or params.get("bias") is not None:
        return None
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
        return None
    return weight.data, scale.data


def _deepseek_v4_mega_moe_max_num_tokens() -> int:
    override = int(
        global_server_args_dict.get("deepseek_v4_mega_moe_max_num_tokens", 0) or 0
    )
    if override > 0:
        return override

    candidates = [
        global_server_args_dict.get("chunked_prefill_size", 0),
        global_server_args_dict.get("prefill_graph_max_tokens", 0),
        global_server_args_dict.get("max_cudagraph_capture_size", 0),
        global_server_args_dict.get("max_num_seqs", 0),
    ]
    return max([int(value or 0) for value in candidates] + [1])


def _random_expert_ids(
    param: torch.Tensor, generator: torch.Generator, num_experts: int
) -> None:
    """Give each token distinct experts, chosen uniformly at random."""
    scores = torch.rand(
        param.shape[0], num_experts, generator=generator, device=param.device
    )
    param.data.copy_(scores.topk(param.shape[1], dim=1).indices)


class DeepseekV4MoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_index: int,
        hash_indices_dtype: torch.dtype = torch.int32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.is_hash_moe = layer_index < config.num_hash_layers
        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=hash_indices_dtype,
                ),
                requires_grad=False,
            )
            self.tid2eid.dummy_initializer = functools.partial(
                _random_expert_ids, num_experts=config.n_routed_experts
            )
            self.e_score_correction_bias = None
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return dsv4_linear_fp32(
            hidden_states,
            self.weight,
        )


class DeepseekV4TopK(TopK):
    def __init__(
        self,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        routed_scaling_factor: float,
        hash_routing: bool,
        weights_dtype: torch.dtype,
    ) -> None:
        super().__init__(
            top_k=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.hash_routing = hash_routing
        self.weights_dtype = weights_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        routing_correction_bias: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> TopKOutput:
        correction_bias = (
            self.topk_config.correction_bias
            if routing_correction_bias is None
            else routing_correction_bias
        )
        if self.simulate_routing:
            router_logits = simulated_router_logits(router_logits)
        topk_weights, topk_ids = moe_topk(
            router_logits,
            self.topk_config.top_k,
            "sqrt_softplus",
            "hash" if self.hash_routing else "topk",
            self.topk_config.renormalize,
            self.topk_config.routed_scaling_factor,
            correction_bias=correction_bias,
            hash_indices_table=hash_indices_table,
            input_ids=input_ids,
            topk_weights_dtype=self.weights_dtype,
        )
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)


def _deepseek_v4_routed_expert_quant_config(
    config: PretrainedConfig,
    quant_config: QuantizationConfig,
) -> tuple[QuantizationConfig, bool]:
    # ``quant_method=fp8`` describes the model-wide quantization, but DeepSeek
    # V4 can still serialize routed experts as packed MXFP4. Prefer the explicit
    # expert contract so packed FP4 tensors are never allocated as block FP8.
    is_block_fp8 = isinstance(quant_config, Fp8Config) and (
        getattr(config, "expert_dtype", None) != "fp4"
    )
    if is_block_fp8:
        return quant_config, True
    return (
        Mxfp4Config(
            ignored_layers=quant_config.ignored_layers,
            is_checkpoint_mxfp4_serialized=True,
        ),
        False,
    )


def _load_mode(ctx: ForwardContext | None) -> int:
    """Row of the load counters a forward counts into (see ``LOAD_MODES``)."""
    mode = None if ctx is None else ctx.forward_mode
    if mode is None:
        return LOAD_MODES.index("other")
    if mode.is_extend():
        return LOAD_MODES.index("extend")
    if mode.is_decode():
        return LOAD_MODES.index("decode")
    if mode.is_mixed():
        return LOAD_MODES.index("mixed")
    return LOAD_MODES.index("other")


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
        aux_stream: object | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek V4 MoE scoring: {self.scoring_func}"
            )
        moe_backend = get_moe_backend()
        self.use_mega_moe = moe_backend.is_mega_moe()
        self.use_gluon_petit = moe_backend.is_gluon_petit()
        self.owns_ep_communication = self.use_mega_moe or self.use_gluon_petit
        # Petit's VMM-backed collective and auxiliary-stream shared experts do
        # not replay safely together. Keep both branches on the main stream.
        self.stream_fork = StreamFork(None if self.use_gluon_petit else aux_stream)
        if mapping.moe.ep_size > 1:
            if global_server_args_dict.get("enable_eplb", False):
                raise ValueError(
                    "DeepSeek V4 EP does not support EPLB with precomputed routing."
                )
            if global_server_args_dict.get("init_expert_location", "trivial") not in (
                None,
                "trivial",
            ):
                raise ValueError(
                    "DeepSeek V4 EP requires trivial expert placement with "
                    "precomputed routing."
                )
        if self.use_mega_moe:
            if mapping.moe.ep_size <= 1:
                raise ValueError("DeepSeek V4 MegaMoE requires expert parallelism.")
            if mapping.moe.tp_size != 1:
                raise ValueError("DeepSeek V4 MegaMoE does not support mixed TP/EP.")
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 MegaMoE does not support redundant EP experts."
                )
            if config.n_routed_experts % mapping.moe.ep_size != 0:
                raise ValueError(
                    "DeepSeek V4 MegaMoE requires n_routed_experts divisible by "
                    "EP size."
                )
        elif mapping.moe.ep_size > 1:
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 normal EP does not support redundant experts "
                    "with precomputed routing."
                )
            if mapping.attn.tp_size not in (1, mapping.moe.tp_ep_size):
                raise ValueError(
                    "DeepSeek V4 normal EP requires attention TP size 1 or "
                    "attention TP size equal to the MoE TPxEP size."
                )
            if config.hidden_size % 256:
                raise ValueError(
                    "DeepSeek V4 normal EP requires hidden_size divisible by 256."
                )
        self.hash_indices_dtype = torch.int32
        self.gate = DeepseekV4MoEGate(
            config,
            layer_index,
            hash_indices_dtype=self.hash_indices_dtype,
        )

        if config.n_shared_experts is not None:
            # MegaMoE outputs are complete, so its shared expert stays dense and local.
            if self.use_mega_moe or self.use_gluon_petit:
                shared_tp_rank = mapping.dense.tp_rank
                shared_tp_size = mapping.dense.tp_size
                shared_tp_group = mapping.dense.tp_group
            else:
                shared_tp_rank = mapping.moe.tp_ep_rank
                shared_tp_size = mapping.moe.tp_ep_size
                shared_tp_group = mapping.moe.tp_ep_group
            self.shared_experts = GatedMLP.tensor_parallel(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                activation=gated_activation(
                    config.hidden_act, getattr(config, "swiglu_limit", None)
                ),
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
                tp_group=shared_tp_group,
                reduce_results=False,
                fusions=(),
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

        if self.use_mega_moe and global_server_args_dict["moe_mxfp4_fp8_activation"]:
            raise ValueError(
                "--moe-mxfp4-fp8-activation selects the FlashInfer cutlass W4A8 "
                "MoE; it does not apply to MegaMoE"
            )
        routed_quant_config, _ = _deepseek_v4_routed_expert_quant_config(
            config, quant_config
        )
        # Expert slots of this layer over the EP group: the routed experts,
        # plus the replica slots of a placement file that replicates some
        # (TOKENSPEED_MOE_EXPERT_PLACEMENT, format 2). The router keeps
        # ``n_routed_experts`` logical experts; only the expert weights and
        # the fused kernel see the slots.
        self.num_expert_slots = expert_placement_slots(config.n_routed_experts)
        if self.num_expert_slots != config.n_routed_experts:
            if not self.use_mega_moe:
                raise ValueError(
                    f"{EXPERT_PLACEMENT_ENV} replicates experts, which needs "
                    "--moe-backend mega_moe"
                )
            if self.num_expert_slots % mapping.moe.ep_size != 0:
                raise ValueError(
                    f"{EXPERT_PLACEMENT_ENV}: {self.num_expert_slots} expert slots "
                    f"do not divide over EP size {mapping.moe.ep_size}"
                )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=self.num_expert_slots,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=routed_quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu",
            swiglu_limit=getattr(config, "swiglu_limit", None),
            with_bias=False,
            routing_mode="precomputed_topk",
            routing_config={
                "routed_scaling_factor": 1.0,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
            persistent_max_num_tokens_per_gpu=(
                _deepseek_v4_mega_moe_max_num_tokens() if self.use_mega_moe else None
            ),
        )
        if self.num_expert_slots != config.n_routed_experts:
            # Replica slots are only addressable through the id map.
            self._require_expert_id_map_kernel()
        self.topk = DeepseekV4TopK(
            top_k=config.num_experts_per_tok,
            renormalize=self._renormalize_routing_weights(),
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_routing=self.gate.tid2eid is not None,
            weights_dtype=self.experts.topk_weights_dtype,
        )
        # True once the MegaMoE kernel runs this layer's shared expert itself:
        # its output is then routed + shared and the dense path is skipped.
        self.shared_fused = False
        self._mega_moe_warmed = False
        # Digest of the expert placement file this layer's routes follow; 0
        # for contiguous ownership (see set_expert_placement).
        self._expert_placement_digest = 0
        # Route counters of TOKENSPEED_MOE_LOAD_PROFILE; None when unset, and
        # then no forward launches anything for them.
        profile = expert_load_profile()
        self._load_counter: LayerLoadCounter | None = (
            None
            if profile is None
            else profile.counter(
                layer_index=layer_index,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                ep_rank=mapping.moe.ep_rank,
                ep_size=mapping.moe.ep_size,
            )
        )

    def set_expert_placement(
        self, logical_to_physical: torch.Tensor, digest: int
    ) -> None:
        """Route this layer's experts through a placement.

        The router keeps logical expert ids. The MegaMoE input staging maps
        them through ``logical_to_physical`` in the launch that already copies
        the ids, so the kernel addresses the slots the checkpoint loader
        filled from the same placement.

        Args:
            logical_to_physical: int32 ``[n_routed_experts]``, logical expert
                id -> slot as this rank routes it; rank ``r`` owns slots
                ``[r * s / EP, (r + 1) * s / EP)`` of the layer's ``s`` slots.
            digest: Digest of the placement file, compared across the EP
                ranks at warmup.

        Raises:
            ValueError: The selected MoE kernel does not map expert ids, the
                map is not this layer's, or ``digest`` is not that of the
                file named by the environment.
        """
        self._require_expert_id_map_kernel()
        expected = (self.config.n_routed_experts,)
        if (
            tuple(logical_to_physical.shape) != expected
            or logical_to_physical.dtype != torch.int32
        ):
            raise ValueError(
                f"expert placement map must be int32 {expected}, got "
                f"{logical_to_physical.dtype} {tuple(logical_to_physical.shape)}"
            )
        # The layers were sized, and the EP ranks are compared, by the bytes
        # this process first read from the environment's file.
        identity = expert_placement_identity()
        if identity is None or identity.digest != digest:
            raise ValueError(
                f"{EXPERT_PLACEMENT_ENV}: the placement to apply has digest "
                f"{digest}, the file this process first read has "
                f"{None if identity is None else identity.digest}. Do not "
                "replace the file while an engine starts."
            )
        self.experts.plan["expert_id_map"] = logical_to_physical
        self._expert_placement_digest = digest

    def _require_expert_id_map_kernel(self) -> None:
        """Refuse a placement unless the selected MoE kernel maps expert ids.

        Decided from the kernel plan, not from the backend or the platform:
        a MegaMoE kernel without the map (DeepGEMM on SM100-SM109) would run
        with placed weights and unmapped ids, wrong experts and no error.
        """
        kernel = self.experts.plan.get("apply_kernel_name")
        if not self.use_mega_moe or kernel not in _EXPERT_ID_MAP_KERNELS:
            raise ValueError(
                f"{EXPERT_PLACEMENT_ENV}: the selected MoE kernel ({kernel}) does "
                "not map routed expert ids to placed slots. An expert placement "
                "file needs --moe-backend mega_moe or mega_moe_split resolving to "
                f"one of {sorted(_EXPERT_ID_MAP_KERNELS)}; unset the variable "
                "otherwise."
            )

    def _check_ep_group_agreement(self) -> None:
        """Fail unless the ranks of the EP group made the same MoE choices.

        Compared are the choices each rank process reads from its own
        environment: the placement file of TOKENSPEED_MOE_EXPERT_PLACEMENT
        (digest of its bytes, its format and its slot count), whether
        TOKENSPEED_MEGA_MOE_FUSED_SHARED asks for the in-kernel shared
        expert, and whether TOKENSPEED_MEGA_MOE_DECODER_ROW_BOUND narrows
        the row bound of the launches behind the CED narrowing. A rank
        started without the file would serve contiguous experts to peers
        that placed theirs; a rank started without the fused-shared variable
        would build the other kernel variant, and the kernel set-up cannot
        tell (same windows, same signals); a rank started without the
        row-bound variable would pass another bound, and so another row
        schedule, into the same cooperative launch.

        One collective per EP group and process, issued by the first MegaMoE
        layer that warms up, before any kernel is built: the choices are
        per process, so it settles them for every MoE stack on the group
        (the target's and the draft's) and for later warmup() calls.

        Default path: with every variable unset this is the one thing a
        MegaMoE start does that the baseline did not. That first warmup
        issues one NCCL all-reduce of ten int64 (all zero then) on the EP
        device group and reads it back on the host, right before the
        kernel's own collective warm-up. No forward is affected. It is not
        skipped when this rank's variables are unset: the rank that lacks
        them is the one the check exists for, and skipping it there would
        leave its peers waiting in the collective.
        """
        group = self.experts.plan.get("process_group")
        identity = expert_placement_identity()
        digest = 0 if identity is None else identity.digest
        if (
            not self.use_mega_moe
            or group is None
            or any(agreed is group for agreed in _agreed_ep_groups)
        ):
            return
        fused = int(mega_moe_fused_shared_requested())
        row_bound = int(mega_moe_decoder_row_bound_requested())
        file_format = 0 if identity is None else identity.format
        slots = 0 if identity is None else identity.num_slots
        mine = (digest, fused, file_format, slots, row_bound)
        # One max-reduce of (v, -v) per value yields the largest and the
        # smallest of the group.
        extremes = torch.tensor(
            [signed for value in mine for signed in (value, -value)],
            dtype=torch.int64,
            device=self.gate.weight.device,
        )
        dist.all_reduce(extremes, op=dist.ReduceOp.MAX, group=group)
        reduced = extremes.tolist()
        largest, smallest = reduced[0::2], [-value for value in reduced[1::2]]
        problems = []
        if (largest[0], largest[2], largest[3]) != (
            smallest[0],
            smallest[2],
            smallest[3],
        ):
            problems.append(
                "EP ranks disagree on the MoE expert placement: this rank's "
                f"digest is {digest} (format {file_format}, {slots} slots), the "
                f"group's digests range from {smallest[0]} to {largest[0]}, its "
                f"formats from {smallest[2]} to {largest[2]}, its slots from "
                f"{smallest[3]} to {largest[3]} (0 = no placement). Give every "
                f"rank the same {EXPERT_PLACEMENT_ENV} file."
            )
        if largest[1] != smallest[1]:
            problems.append(
                f"EP ranks disagree on {MEGA_MOE_FUSED_SHARED_ENV}: this rank "
                f"{'asks' if fused else 'does not ask'} for the shared expert "
                "inside the MegaMoE kernel, other ranks do the opposite. Set "
                f"{MEGA_MOE_FUSED_SHARED_ENV}=1 on every rank or on none."
            )
        if largest[4] != smallest[4]:
            problems.append(
                f"EP ranks disagree on {MEGA_MOE_DECODER_ROW_BOUND_ENV}: this "
                f"rank {'asks' if row_bound else 'does not ask'} for the "
                "narrowed row bound of the MegaMoE launches behind the CED "
                "narrowing, other ranks do the opposite. Set "
                f"{MEGA_MOE_DECODER_ROW_BOUND_ENV}=1 on every rank or on none."
            )
        if problems:
            raise RuntimeError(" ".join(problems))
        _agreed_ep_groups.append(group)
        placement = (
            "none"
            if identity is None
            else f"digest {digest}, format {file_format}, {slots} slots"
        )
        logger.info(
            f"EP group of rank {self.mapping.moe.ep_rank} agrees: expert "
            f"placement {placement}; {MEGA_MOE_FUSED_SHARED_ENV} "
            f"{'requested' if fused else 'not requested'}; "
            f"{MEGA_MOE_DECODER_ROW_BOUND_ENV} "
            f"{'requested' if row_bound else 'not requested'}"
        )

    def _record_expert_load(
        self, topk_output: TopKOutput, ctx: ForwardContext | None
    ) -> None:
        if self._load_counter is not None:
            self._load_counter.record(topk_output.topk_ids, _load_mode(ctx))

    def warmup(self) -> None:
        if self._load_counter is not None:
            # Sized for the layer's row capacity where the kernel plan has
            # one, so the counters' buffer of ones never grows under graphs.
            self._load_counter.allocate(
                self.gate.weight.device,
                self.experts.plan.get("persistent_max_num_tokens_per_gpu"),
            )
        id_map = self.experts.plan.get("expert_id_map")
        if id_map is not None and id_map.device != self.gate.weight.device:
            self.experts.plan["expert_id_map"] = id_map.to(self.gate.weight.device)
        self._check_ep_group_agreement()
        warmup = self.experts.plan.get("warmup")
        if warmup is not None:
            if not self.experts._weights_processed:
                raise RuntimeError("MoE weights must be processed before warmup")
            if not self._mega_moe_warmed:
                # The kernel variant is fixed when the layer is first built.
                self._maybe_fuse_shared_expert()
            warmup(self.experts.plan, self.experts)
            self._mega_moe_warmed = True

    def _maybe_fuse_shared_expert(self) -> None:
        """Hand the shared expert to the MegaMoE kernel when the switch asks for it.

        Runs once, after the weights are final and before the kernel is built.
        Anything the kernel cannot take leaves the layer on the separate path.
        """
        if not (self.use_mega_moe and mega_moe_fused_shared_requested()):
            return
        if self.shared_experts is None:
            return
        fuse = self.experts.plan.get("fuse_shared_expert")
        gate_up = _mxfp8_codes_and_scales(self.shared_experts.gate_up_proj)
        down = _mxfp8_codes_and_scales(self.shared_experts.down_proj)
        if fuse is None:
            reason = "the selected MoE kernel has no in-kernel shared expert"
        elif self.mapping.dense.tp_size != 1:
            reason = "the shared expert is tensor-parallel"
        elif gate_up is None or down is None:
            reason = "the shared expert is not bias-free MXFP8 with UE8M0 scales"
        elif fuse(self.experts.plan, self.experts, *gate_up, *down):
            self.shared_fused = True
            logger.info(
                f"{MEGA_MOE_FUSED_SHARED_ENV}=1: layer {self.layer_index} runs its "
                "shared expert inside the MegaMoE kernel"
            )
            return
        else:
            reason = "the kernel refused the shared expert's shapes or layout"
        logger.warning(
            f"{MEGA_MOE_FUSED_SHARED_ENV}=1 ignored for layer {self.layer_index}: "
            f"{reason}; the shared expert stays on the separate path"
        )

    def _routing_inputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if hidden_states.shape[0] == 0:
            router_logits = hidden_states.new_empty(
                (0, self.config.n_routed_experts), dtype=torch.float32
            )
        else:
            router_logits = self.gate(hidden_states)
        hash_indices_table = self.gate.tid2eid
        return (
            router_logits,
            self.gate.e_score_correction_bias,
            hash_indices_table,
            input_ids if hash_indices_table is not None else None,
        )

    def _renormalize_routing_weights(self) -> bool:
        return self.config.norm_topk_prob

    def _compute_topk_output(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> TopKOutput:
        router_logits, correction_bias, hash_indices_table, routing_input_ids = (
            self._routing_inputs(hidden_states, input_ids)
        )
        return self.topk(
            hidden_states,
            router_logits,
            routing_correction_bias=correction_bias,
            hash_indices_table=hash_indices_table,
            input_ids=routing_input_ids,
        )

    def _forward_routed_experts(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
        )

    def _forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        if comm_manager is not None:
            hidden_states = comm_manager.pre_dense_comm(hidden_states, ctx)
        if hidden_states.shape[0] == 0:
            return None
        with nvtx_range("moe_shared_experts"):
            shared = self.shared_experts(hidden_states)
        if comm_manager is not None:
            shared, _ = comm_manager.post_dense_comm(shared, None, ctx)
        return shared

    def forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
        comm_manager: CommManager,
        before_launch: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Route, then run the routed (and shared) experts in MegaMoE.

        ``before_launch`` is an optional callback of the caller, run on the
        calling stream once the kernel's inputs are staged and before the
        MegaMoE kernel is launched (never when this rank has no kernel launch
        path that supports it). It orders the caller's side-stream work in
        front of the launch; None, the default, leaves the forward unchanged.
        """
        if before_launch is None:
            return self._forward_mega_moe(
                hidden_states,
                input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx,
                comm_manager,
            )
        plan = self.experts.plan
        plan["before_kernel_launch"] = before_launch
        try:
            return self._forward_mega_moe(
                hidden_states,
                input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx,
                comm_manager,
            )
        finally:
            plan.pop("before_kernel_launch", None)

    def _forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
        comm_manager: CommManager,
    ) -> torch.Tensor:
        with nvtx_range("moe_select_experts"):
            topk_output = self._compute_topk_output(hidden_states, input_ids)
        self._record_expert_load(topk_output, ctx)
        if self.shared_fused:
            # One launch returns routed + shared for this rank's rows.
            with nvtx_range("moe_mega_experts"):
                return self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=True,
                )
        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_mega_experts"):
                routed = self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=True,
                )
            with fork.branch():
                shared = self._forward_shared_experts(
                    hidden_states,
                    ctx=ctx,
                    comm_manager=comm_manager,
                )
        return routed + shared if shared is not None else routed

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write the routed + shared sum into out when supplied.

        The following all-reduce can read this memory in place.
        """
        if hidden_states.shape[0] == 0 and not self.use_gluon_petit:
            return hidden_states if out is None else out.copy_(hidden_states)
        with nvtx_range("moe_select_experts"):
            topk_output = self._compute_topk_output(hidden_states, input_ids)
        self._record_expert_load(topk_output, None)
        deferred_finalize = (
            hidden_states.dtype == torch.bfloat16
            and self.experts.supports_deferred_cuda_finalize
        )
        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_experts"):
                routed = self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=not deferred_finalize,
                )
            with fork.branch():
                shared = self._forward_shared_experts(hidden_states)
        if deferred_finalize:
            # Reduce routed rows and add the shared expert in one finalize.
            gemm2_out, expert_weights, expanded_idx = routed
            result = moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                shared,
                top_k=expert_weights.shape[1],
                enable_pdl=pdl_enabled(),
                hidden_dim=hidden_states.shape[-1],
            )
            return result if out is None else out.copy_(result)
        if shared is None:
            return routed if out is None else out.copy_(routed)
        return torch.add(routed, shared, out=out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
        out: torch.Tensor | None = None,
        before_launch: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        if self.use_mega_moe:
            if out is not None:
                raise ValueError("DeepSeek V4 MegaMoE does not support out")
            if before_launch is not None:
                return self.forward_mega_moe(
                    hidden_states,
                    input_ids,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    ctx,
                    comm_manager,
                    before_launch=before_launch,
                )
            return self.forward_mega_moe(
                hidden_states,
                input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx,
                comm_manager,
            )
        if before_launch is not None:
            raise ValueError("before_launch needs the MegaMoE backend")
        return self.forward_normal(
            hidden_states, input_ids, num_global_tokens, max_num_tokens_per_gpu, out
        )
