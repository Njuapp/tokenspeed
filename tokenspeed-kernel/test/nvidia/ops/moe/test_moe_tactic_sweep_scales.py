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

"""CPU regression for format-specific synthetic MoE weight scales."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "quant_format,scale_byte,block", [("mxfp", 127, 32), ("nvfp4", 0x38, 16)]
)
def test_synthetic_weight_scales_encode_one(quant_format, scale_byte, block):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/tokenspeed_kernel/ops/moe/flashinfer/moe_tactic_sweep.py"
    )
    tree = ast.parse(path.read_text())
    names = {"_scale_block_size", "_make_weights", "_make_output_scales"}
    namespace = {"torch": torch, "MXFP_SF_BLOCK": 32, "NVFP4_SF_BLOCK": 16}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    _, w13_scale, _, w2_scale = namespace["_make_weights"](
        2, 64, 32, torch.device("cpu"), 42, quant_format=quant_format
    )
    assert w13_scale.shape == (2, 64, 64 // block)
    assert w2_scale.shape == (2, 64, 32 // block)
    for scale in (w13_scale, w2_scale):
        assert torch.all(scale.view(torch.uint8) == scale_byte)
        if quant_format == "nvfp4":
            assert torch.all(scale.float() == 1)
    output_scales = namespace["_make_output_scales"](
        quant_format, 2, torch.device("cpu")
    )
    if quant_format == "nvfp4":
        assert all(torch.all(scale == 1) for scale in output_scales)
    else:
        assert output_scales == (None, None, None)
