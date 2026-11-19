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

"""PCIe all-reduce for two groups of four SM120 GPUs."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.fabric import host_identity

logger = logging.getLogger(__name__)


@triton.jit
def _timer_ns():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _read_ready(ptr, mask, EMPTY: tl.constexpr):
    v = tl.load(ptr, mask, 0, volatile=True)
    missing = tl.sum(
        ((v.to(EMPTY[0], bitcast=True) == EMPTY[1]) & mask).to(tl.int32), 0
    )
    started = _timer_ns()
    elapsed = tl.full((), 0, tl.uint64)
    polls = 0
    # Profiler callbacks and host compilation can delay a peer for seconds.
    # Poll counts do not measure that delay consistently across devices.
    while (missing != 0) & (elapsed < 60000000000):
        polls = (polls + 1) % 1024
        v = tl.load(ptr, mask, 0, volatile=True)
        missing = tl.sum(
            ((v.to(EMPTY[0], bitcast=True) == EMPTY[1]) & mask).to(tl.int32), 0
        )
        if polls == 0:
            elapsed = _timer_ns() - started
    tl.device_assert(missing == 0, "peer payload did not arrive within 60 seconds")
    return v


@triton.jit(do_not_specialize=["N"])
def _pcie_hierarchical_all_reduce(
    X,
    Out,
    Pointers,
    Counters,
    N,
    CAP: tl.constexpr,
    RANK: tl.constexpr,
    B: tl.constexpr,
):
    Pointers = Pointers.to(tl.pointer_type(tl.uint64))
    pid = tl.program_id(0)
    epoch = tl.load(Counters + pid)
    parity = epoch % 3
    chunk = N // 4
    offset = pid.to(tl.int64) * B + tl.arange(0, B)
    mask = offset < chunk
    base = parity.to(tl.int64) * 10 * CAP
    me = tl.load(Pointers + RANK).to(tl.pointer_type(tl.bfloat16)) + base
    local = RANK % 4
    island = RANK // 4
    # Each destination owns one quarter. Only the owning rank reads each partial.
    for dest in tl.static_range(4):
        peer = (
            tl.load(Pointers + island * 4 + dest).to(tl.pointer_type(tl.bfloat16))
            + base
        )
        value = tl.load(X + dest * chunk.to(tl.int64) + offset, mask, 0)
        value = tl.where(value == 0, 0.0, value).to(tl.bfloat16)
        tl.store(peer + local * CAP + offset, value, mask)
    acc = tl.full((B,), 0.0, tl.float32)
    for src in tl.static_range(4):
        value = _read_ready(me + src * CAP + offset, mask, (tl.uint16, 0x8000))
        acc = acc + value.to(tl.float32)
        tl.store(me + src * CAP + offset, -0.0, mask)
    partner = tl.load(Pointers + (RANK + 4) % 8).to(tl.pointer_type(tl.bfloat16)) + base
    cross = (partner + 4 * CAP).to(tl.pointer_type(tl.float32)) + offset
    acc = tl.where(acc == 0, 0.0, acc)
    tl.store(cross, acc, mask)
    mine_cross = (me + 4 * CAP).to(tl.pointer_type(tl.float32)) + offset
    other = _read_ready(mine_cross, mask, (tl.uint32, 0x80000000))
    reduced = (acc + other) if island == 0 else (other + acc)
    tl.store(mine_cross, -0.0, mask)
    reduced = tl.where(reduced == 0, 0.0, reduced).to(tl.bfloat16)
    for dest in tl.static_range(4):
        peer = (
            tl.load(Pointers + island * 4 + dest).to(tl.pointer_type(tl.bfloat16))
            + base
        )
        tl.store(peer + 6 * CAP + local * CAP + offset, reduced, mask)
    for src in tl.static_range(4):
        ptr = me + 6 * CAP + src * CAP + offset
        value = _read_ready(ptr, mask, (tl.uint16, 0x8000))
        tl.store(Out + src * chunk.to(tl.int64) + offset, value, mask)
        tl.store(ptr, -0.0, mask)
    tl.store(Counters + pid, epoch + 1)


class PcieAllReduceWorkspace:
    """Own buffers for an eight-rank SUM reduction over two NUMA groups.

    Calls on one workspace must be serialized on a CUDA stream, including
    graph replay. Buffer contents are private; only the returned output may
    outlive a call. FP32 additions use a fixed two-level tree and round once
    to BF16. This is not the rank-ordered collective used by rl-bitwise.
    """

    def __init__(
        self, group, rank_order: tuple[int, ...], capacity: int, num_ctas: int
    ):
        self.num_ctas = num_ctas
        self.capacity = capacity
        self.cap = triton.cdiv(capacity, 4)
        self.rank = rank_order.index(dist.get_rank(group))
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.buffer = symm_mem.empty(
            (3, 10 * self.cap), dtype=torch.bfloat16, device=self.device
        )
        self.buffer.fill_(-0.0)
        self.buffer[:, 4 * self.cap : 6 * self.cap].view(torch.float32).fill_(-0.0)
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.pointers = torch.tensor(
            [int(self.handle.buffer_ptrs[r]) for r in rank_order],
            dtype=torch.int64,
            device=self.device,
        )
        self.counters = torch.zeros(
            self.num_ctas, dtype=torch.int64, device=self.device
        )
        # Precompile every tile bucket before any peer enters a polling kernel.
        temporary = torch.empty(capacity, dtype=torch.bfloat16, device=self.device)
        maximum_block = max(
            128, triton.next_power_of_2(triton.cdiv(capacity // 4, self.num_ctas))
        )
        block = 128
        while block <= maximum_block:
            _pcie_hierarchical_all_reduce.warmup(
                temporary,
                temporary,
                self.pointers,
                self.counters,
                capacity,
                self.cap,
                self.rank,
                block,
                grid=(self.num_ctas,),
                num_warps=4,
                enable_fp_fusion=False,
                debug=True,
            )
            block *= 2
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def supports(self, x: torch.Tensor) -> bool:
        return (
            x.device == self.device
            and x.dtype == torch.bfloat16
            and 0 < x.numel() <= self.capacity
            and x.numel() % 4 == 0
        )

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
        if not self.supports(x):
            raise ValueError("the PCIe workspace does not support this tensor")
        if out is None:
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        elif (
            out.shape != x.shape
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous and match the input")
        source = x.contiguous()
        count = source.numel()
        block = max(128, triton.next_power_of_2(triton.cdiv(count // 4, self.num_ctas)))
        _pcie_hierarchical_all_reduce[(self.num_ctas,)](
            source,
            out,
            self.pointers,
            self.counters,
            count,
            self.cap,
            self.rank,
            block,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        return out


def create_pcie_all_reduce_workspace(
    group: dist.ProcessGroup, capacity: int
) -> PcieAllReduceWorkspace | None:
    """Create a workspace only for eight peer-accessible SM120 GPUs.

    All ranks in group call this during initialization. The function checks
    host identity and two NUMA groups of four GPUs. It derives rank order
    from the devices, so CUDA_VISIBLE_DEVICES may reorder physical GPUs.
    An unsupported or unknown topology returns None on every rank.
    """
    if dist.get_world_size(group) != 8 or capacity <= 0 or capacity % 4:
        return None
    local = None
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        bus = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        numa = int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text())
        visible = []
        peers = True
        for ordinal in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(ordinal)
            visible.append((p.pci_domain_id, p.pci_bus_id, p.pci_device_id))
            if ordinal != device:
                peers = peers and torch.cuda.can_device_access_peer(device, ordinal)
        local = {
            "host": host_identity(),
            "device": (props.pci_domain_id, props.pci_bus_id, props.pci_device_id),
            "numa": numa,
            "sm_count": props.multi_processor_count,
            "supported": (props.major, props.minor) == (12, 0)
            and props.multi_processor_count >= 32
            and peers
            and numa >= 0,
            "visible": visible,
        }
    except (OSError, RuntimeError, AttributeError, ValueError) as exc:
        logger.info(f"PCIe all-reduce topology is unavailable: {exc}")
    members = [None] * 8
    dist.all_gather_object(members, local, group=group)
    if any(member is None or not member["supported"] for member in members):
        return None
    devices = {member["device"] for member in members}
    if len({member["host"] for member in members}) != 1 or len(devices) != 8:
        return None
    if any(not devices.issubset(set(member["visible"])) for member in members):
        return None
    nodes = sorted({member["numa"] for member in members})
    if len(nodes) != 2 or any(
        sum(member["numa"] == node for member in members) != 4 for node in nodes
    ):
        return None
    order = tuple(sorted(range(8), key=lambda rank: (members[rank]["numa"], rank)))
    # Every rank uses the same fixed grid. One CTA per available SM avoids
    # a polling CTA waiting for another CTA that cannot become resident.
    sm_count = min(member["sm_count"] for member in members)
    num_ctas = min(128, 1 << (sm_count.bit_length() - 1))
    return PcieAllReduceWorkspace(group, order, capacity, num_ctas)
