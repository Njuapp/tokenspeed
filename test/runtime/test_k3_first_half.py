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

"""CPU routing and host orchestration; not GPU numerical qualification."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenspeed.runtime.models import kimi_k3_tail_policy as policy

ROOT = Path(__file__).resolve().parents[2]
COMM = ROOT / "python/tokenspeed/runtime/models/kimi_k3_comm.py"


def comm_method(name, namespace, owner):
    """Execute a real host method against explicit device test doubles."""
    tree = ast.parse(COMM.read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == owner)
    method = next(n for n in cls.body if getattr(n, "name", None) == name)
    ns = dict(vars(policy))
    ns.update(namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(COMM), "exec"), ns)
    return ns[name]


def original(m, graph, decode):
    return policy.select_k3_moe_tail_tier(
        num_tokens=m,
        graph_phase=graph,
        tail_fusion_max_tokens=32,
        fused_moe_ar=True,
        multimem_ok=True,
        is_decode=decode,
        join_moe_reduce=False,
    )


class FirstHalfPolicyTests(unittest.TestCase):
    def test_every_integer_and_missing_backends(self):
        for graph in (False, True):
            for decode in (False, True):
                for m in range(8194):
                    base = original(m, graph, decode)
                    for bt, ht in (
                        (True, True),
                        (False, False),
                        (True, False),
                        (False, True),
                    ):
                        result = policy.select_bt_ht_first_half(
                            original=base,
                            num_tokens=m,
                            bt_ok=bt,
                            ht_ok=ht,
                        )
                        expected = base
                        if bt and 33 <= m <= 1024:
                            expected = policy.K3MoETailTier.MNNVL_BT_DEFERRED
                        elif ht and 1025 <= m <= 8192:
                            expected = policy.K3MoETailTier.MNNVL_HT_DEFERRED
                        self.assertIs(result, expected, (m, graph, decode))

    def test_small_tail_priority(self):
        for m in (1, 32, 896):
            self.assertIs(
                policy.select_bt_ht_first_half(
                    original=policy.K3MoETailTier.TAIL_FUSION,
                    num_tokens=m,
                    bt_ok=True,
                    ht_ok=True,
                ),
                policy.K3MoETailTier.TAIL_FUSION,
            )
        self.assertIs(original(32, True, False), policy.K3MoETailTier.TAIL_FUSION)
        self.assertIs(original(32, False, False), policy.K3MoETailTier.FUSED_LANE_AR)

    def test_plan_keeps_ar2_choice_without_any_second_half_workspace(self):
        plan = comm_method(
            "plan",
            dict(
                torch=SimpleNamespace(Tensor=object),
                TailPlan=SimpleNamespace,
                get_is_cuda_graph_phase=lambda: False,
            ),
            "K3MoeTailComm",
        )
        instance = SimpleNamespace(
            latent_tail=None,
            execution_plan=SimpleNamespace(fused_moe_ar=True, join_moe_reduce=False),
            state=SimpleNamespace(
                multimem_ar_ok=True,
                mnnvl_bt_deferred=SimpleNamespace(
                    supports_num_tokens=lambda m: 33 <= m <= 1024
                ),
                mnnvl_ht_deferred=SimpleNamespace(
                    supports_num_tokens=lambda m: 1025 <= m <= 8192
                ),
            ),
        )
        for decode in (False, True):
            for m in (33, 64, 255, 256, 896, 1024, 1025, 1280, 2048, 4096, 8192):
                result = plan(instance, m, None, is_decode=decode)
                self.assertTrue(result.defer_finalize)
                self.assertEqual(result.second_multimem, not decode and m >= 256)
                self.assertIs(
                    result.tier,
                    (
                        policy.K3MoETailTier.MNNVL_BT_DEFERRED
                        if m <= 1024
                        else policy.K3MoETailTier.MNNVL_HT_DEFERRED
                    ),
                )

    def test_constructor_selects_capable_plans_without_an_environment_switch(self):
        records = []
        state = SimpleNamespace(latent_tail_ok=False)
        config = {"disable_pdl": False}

        def get(**kwargs):
            records.append(kwargs)
            return state

        init = comm_method(
            "__init__",
            dict(
                global_server_args_dict=config,
                K3MoeTailCommState=SimpleNamespace(get=get),
            ),
            "K3MoeTailComm",
        )
        args = dict(
            mapping=SimpleNamespace(moe=SimpleNamespace(tp_size=8, ep_size=1)),
            hidden_size=7168,
            prefix="test.layer",
            layer_index=0,
            model_scope="main",
            routed_hidden=3584,
            top_k=16,
            routed_norm=SimpleNamespace(variance_epsilon=1e-5),
            up_proj=SimpleNamespace(shard_group="tp8"),
            execution_plan=SimpleNamespace(fused_moe_ar=True, use_native=False),
            experts_supports_deferred_finalize=True,
        )
        # No os/environ test double: eligible plans must select BT/HT without
        # reading an environment variable or allocating a per-layer output.
        for _ in range(2):
            obj = SimpleNamespace()
            init(obj, **args)
            self.assertIs(obj.state, state)
            self.assertTrue(records[-1]["first_half_eligible"])
            self.assertEqual(records[-1]["first_half_capacity"], 8192)
            self.assertNotIn("fused_tail_output", vars(obj))
        for field in (
            "disable_pdl",
            "no_deferred",
            "replicated",
            "no_fused_plan",
            "no_norm",
            "tp4",
            "ep2",
        ):
            changed = dict(args)
            config["disable_pdl"] = field == "disable_pdl"
            if field == "no_deferred":
                changed["experts_supports_deferred_finalize"] = False
            if field == "replicated":
                changed["up_proj"] = SimpleNamespace(shard_group=None)
            if field == "no_fused_plan":
                changed["execution_plan"] = SimpleNamespace(
                    fused_moe_ar=False, use_native=False
                )
            if field == "no_norm":
                changed["routed_norm"] = None
            if field in ("tp4", "ep2"):
                changed["mapping"] = SimpleNamespace(
                    moe=SimpleNamespace(
                        tp_size=4 if field == "tp4" else 8,
                        ep_size=2 if field == "ep2" else 1,
                    )
                )
            init(SimpleNamespace(), **changed)
            self.assertFalse(records[-1]["first_half_eligible"])
            self.assertEqual(records[-1]["first_half_capacity"], 0)

    def test_runtime_does_not_read_a_bt_ht_environment_flag(self):
        self.assertNotIn("TOKENSPEED_K3_BT_HT", COMM.read_text())
        self.assertNotIn("enabled", policy.select_bt_ht_first_half.__annotations__)

    def test_no_second_half_fusion_dependencies_in_runtime(self):
        source = COMM.read_text()
        for forbidden in (
            "mnnvl_fused_tail",
            "FusedRsUpAgWorkspace",
            "symmetric_up_projection",
            "TOKENSPEED_K3_FUSED_RS_UP_AG",
        ):
            self.assertNotIn(forbidden, source)
        kernel = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
        self.assertFalse((kernel / "ops/communication/mnnvl_fused_tail.py").exists())
        self.assertFalse(
            (kernel / "thirdparty/cute_dsl/symmetric_up_projection").exists()
        )

    def test_bt_and_ht_keep_original_ordinary_projection_and_allreduce(self):
        for tier, m in (
            (policy.K3MoETailTier.MNNVL_BT_DEFERRED, 896),
            (policy.K3MoETailTier.MNNVL_HT_DEFERRED, 4096),
        ):
            calls = []
            rows, scales, indices, shared, residual, latent, gamma, projected, final = [
                object() for _ in range(9)
            ]

            def first(*args):
                self.assertEqual(args, (rows, scales, indices, gamma))
                calls.append("first half")
                return latent

            def projection(*args):
                self.assertEqual(args, (latent, shared, residual, m, 7168))
                calls.append("original projection")
                return projected

            def allreduce(value, group):
                self.assertIs(value, projected)
                self.assertEqual(group, "tp8")
                calls.append("original AR2")
                return final

            method = comm_method(
                "_tail_first_half_deferred",
                dict(
                    dist=SimpleNamespace(
                        group=SimpleNamespace(WORLD=SimpleNamespace(group_name="world"))
                    ),
                    all_reduce=allreduce,
                ),
                "K3MoeTailComm",
            )
            instance = SimpleNamespace(
                state=SimpleNamespace(mnnvl_bt_deferred=first, mnnvl_ht_deferred=first),
                routed_norm=SimpleNamespace(weight=gamma),
                mapping=SimpleNamespace(moe=SimpleNamespace(tp_ep_group="tp8")),
                _project_and_inject_local_block=projection,
            )
            self.assertIs(
                method(
                    instance,
                    tier,
                    (rows, scales, indices),
                    shared,
                    residual,
                    m,
                    7168,
                    False,
                ),
                final,
            )
            self.assertEqual(
                calls, ["first half", "original projection", "original AR2"]
            )

    def test_bt_and_ht_keep_multimem_owner_rounding_order_and_clone(self):
        for tier, m in (
            (policy.K3MoETailTier.MNNVL_BT_DEFERRED, 896),
            (policy.K3MoETailTier.MNNVL_HT_DEFERRED, 4096),
        ):
            calls = []
            (
                rows,
                scales,
                indices,
                shared,
                latent,
                gamma,
                residual_slice,
                weight_t,
                final,
            ) = [object() for _ in range(9)]
            owner = self

            class Target:
                def __iadd__(self, rhs):
                    owner.assertIs(rhs, residual_slice)
                    calls.append("BF16 owner residual add")
                    return self

                def addmm_(self, left, right):
                    owner.assertEqual((left, right), (latent, weight_t))
                    calls.append("cuBLAS addmm")

            target = Target()

            class Stage:
                def __getitem__(self, key):
                    owner.assertEqual(key, (slice(None), slice(1792, 2688)))
                    return target

            stage = Stage()

            def stage_copy(value, group, capacity):
                self.assertEqual((value, group, capacity), (shared, "world", 8192))
                calls.append("shared stage")
                return stage

            def first(*args):
                self.assertEqual(args, (rows, scales, indices, gamma))
                calls.append("first half")
                return latent

            def narrow(dim, start, width):
                self.assertEqual((dim, start, width), (-1, 1792, 896))
                return residual_slice

            def view(*shape):
                self.assertEqual(shape, (m, 7168))
                return SimpleNamespace(narrow=narrow)

            def clone():
                calls.append("clone")
                return final

            def allreduce(value, group):
                self.assertEqual((value, group), (stage, "world"))
                calls.append("original multimem AR2")
                return SimpleNamespace(view=lambda *shape: SimpleNamespace(clone=clone))

            instance = SimpleNamespace(
                state=SimpleNamespace(mnnvl_bt_deferred=first, mnnvl_ht_deferred=first),
                routed_norm=SimpleNamespace(weight=gamma),
                up_proj=SimpleNamespace(
                    shard_slice=(1792, 896), weight=SimpleNamespace(t=lambda: weight_t)
                ),
            )
            method = comm_method(
                "_tail_first_half_deferred",
                dict(
                    dist=SimpleNamespace(
                        group=SimpleNamespace(WORLD=SimpleNamespace(group_name="world"))
                    ),
                    multimem_stage=stage_copy,
                    multimem_all_reduce_staged=allreduce,
                ),
                "K3MoeTailComm",
            )
            self.assertIs(
                method(
                    instance,
                    tier,
                    (rows, scales, indices),
                    shared,
                    SimpleNamespace(view=view),
                    m,
                    7168,
                    True,
                ),
                final,
            )
            self.assertEqual(
                calls,
                [
                    "shared stage",
                    "first half",
                    "BF16 owner residual add",
                    "cuBLAS addmm",
                    "original multimem AR2",
                    "clone",
                ],
            )

    def test_collective_init_only_builds_bt_ht_and_rejects_capture_or_disagreement(
        self,
    ):
        builds = []
        control = {"capture": False, "disagree": False, "support": True}

        def gather(out, identity, group):
            out[:] = [identity] * 8
            if control["disagree"]:
                out[7] = (False, 0)

        def initialize(kind, kwargs):
            builds.append((kind, kwargs))
            return kind

        ns = dict(
            torch=SimpleNamespace(
                int32=object(),
                bfloat16=object(),
                cuda=SimpleNamespace(
                    is_current_stream_capturing=lambda: control["capture"]
                ),
                tensor=lambda values, **kwargs: SimpleNamespace(tolist=lambda: values),
            ),
            dist=SimpleNamespace(
                group=SimpleNamespace(WORLD="world"),
                get_world_size=lambda group: 8,
                all_gather_object=gather,
                all_reduce=lambda *args, **kwargs: None,
                ReduceOp=SimpleNamespace(MIN=object()),
            ),
            logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
            mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported=lambda **kwargs: control[
                "support"
            ],
            mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported=lambda **kwargs: control[
                "support"
            ],
            MNNVLCuteDSLFinalizeAllReduceRMSNorm=SimpleNamespace(
                initialize=lambda **kwargs: initialize("BT", kwargs)
            ),
            MNNVLCuteDSLHTFinalizeAllReduceRMSNorm=SimpleNamespace(
                initialize=lambda **kwargs: initialize("HT", kwargs)
            ),
            MNNVLCuteDSLBTFinalizeTuning=lambda **kwargs: kwargs,
            MNNVLCuteDSLHTFinalizeTuning=lambda **kwargs: kwargs,
        )
        method = comm_method("_initialize_first_half", ns, "K3MoeTailCommState")
        mapping = SimpleNamespace(
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_size=8)
        )
        instance = SimpleNamespace(
            first_half_eligible=True,
            first_half_capacity=8192,
            multimem_ar_ok=True,
            hidden_size=7168,
            latent_size=3584,
            top_k=16,
            rms_eps=1e-5,
        )
        method(instance, mapping)
        self.assertEqual([kind for kind, _ in builds], ["BT", "HT"])
        self.assertEqual(builds[0][1]["candidate_max_tokens"], 1024)
        self.assertEqual(builds[1][1]["candidate_min_tokens"], 1025)
        self.assertEqual(builds[1][1]["tuning_routes"][0]["stages"], 10)
        self.assertEqual(
            (instance.mnnvl_bt_deferred, instance.mnnvl_ht_deferred), ("BT", "HT")
        )
        builds.clear()
        control["support"] = False
        method(instance, mapping)
        self.assertEqual(builds, [])
        control["disagree"] = True
        with self.assertRaisesRegex(RuntimeError, "rank-inconsistent"):
            method(instance, mapping)
        control["capture"] = True
        with self.assertRaisesRegex(RuntimeError, "must precede"):
            method(instance, mapping)
        control.update(capture=False, disagree=False)
        instance.first_half_eligible = False
        method(instance, mapping)
        self.assertEqual(builds, [])


if __name__ == "__main__":
    unittest.main()
