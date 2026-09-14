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

"""Execute NVIDIA host contracts without importing optional CUDA/FlashInfer packages.

These tests extract unchanged Python definitions and supply fake device objects.
They do not compile a CuTe kernel or establish distributed numerical correctness.
"""

import ast
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2] / "python/tokenspeed_kernel"
FINALIZE = ROOT / "ops/communication/mnnvl_cutedsl_finalize.py"


def definitions(path, names, **namespace):
    selected = []
    for node in ast.parse(path.read_text()).body:
        defined = {getattr(node, "name", None)}
        if isinstance(node, ast.Assign):
            defined.update(getattr(t, "id", None) for t in node.targets)
        if defined.intersection(names):
            selected.append(node)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class HostContracts(unittest.TestCase):
    def test_finalize_constructor_rejects_capture_before_host_collective(self):
        ns = definitions(
            FINALIZE,
            {"_require_collective_agreement"},
            Any=Any,
            dist=SimpleNamespace(ProcessGroup=object),
            torch=SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "must precede capture"):
            ns["_require_collective_agreement"](None, (3584, 16))

    def test_bt_tuning_validation(self):
        ns = definitions(
            FINALIZE,
            {"MNNVLCuteDSLBTFinalizeTuning", "_validate_tuning_routes"},
            dataclass=dataclass,
        )
        config = ns["MNNVLCuteDSLBTFinalizeTuning"](1024, 2, 256, 1, 224, 448, True)

        def validate(routes):
            ns["_validate_tuning_routes"](
                routes=routes,
                candidate_min_tokens=33,
                candidate_max_tokens=1024,
                hidden_size=3584,
                top_k=16,
            )

        validate((config,))
        for routes in ((), (replace(config, max_tokens=1023),), (config, config)):
            with self.assertRaises(ValueError):
                validate(routes)
        for field, value in (
            ("threads", 33),
            ("elements_per_thread", 3),
            ("prefetch_group", 17),
            ("enable_pdl", 1),
        ):
            with self.assertRaises(ValueError):
                validate((replace(config, **{field: value}),))

    def test_ht_tuning_resource_validation(self):
        ns = definitions(
            FINALIZE,
            {
                "MNNVLCuteDSLHTFinalizeTuning",
                "_validate_ht_tuning_routes",
                "_K3_TP_SIZE",
                "_K3_LATENT_SIZE",
                "_WARP_SIZE",
                "_BF16_VECTOR_SIZE",
            },
            dataclass=dataclass,
            Any=Any,
        )
        config = ns["MNNVLCuteDSLHTFinalizeTuning"](
            8192, None, 448, 1, 10, 2, None, 2, 3, False, True
        )

        def validate(route):
            ns["_validate_ht_tuning_routes"](
                routes=(route,), candidate_min_tokens=1025, candidate_max_tokens=8192
            )

        validate(config)
        for field, value in (
            ("persistent_ctas", 7),
            ("consumer_threads", 1024),
            ("stages", 1),
            ("reduction_warps", 16),
            ("vectors_per_thread", 2),
            ("rms_pipeline_stages", 4),
        ):
            with self.assertRaises(ValueError):
                validate(replace(config, **{field: value}))

    def test_no_optional_backend_imports_at_runtime_boundary(self):
        runtime = ROOT.parents[2] / "python/tokenspeed/runtime/models/kimi_k3_comm.py"
        for node in ast.walk(ast.parse(runtime.read_text())):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(
                    (node.module or "").startswith(
                        ("flashinfer", "cutlass", "tokenspeed_kernel.thirdparty")
                    )
                )


if __name__ == "__main__":
    unittest.main()
