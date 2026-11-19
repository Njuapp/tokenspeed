# PCIe all-reduce on SM120

The eight-rank workspace serves BF16 SUM reductions that fit its small-batch
capacity. Initialization verifies one host, two NUMA groups of four GPUs,
and peer access. It derives the rank order from the physical devices.

Each group first reduces four rank inputs in FP32 into one quarter of the
tensor per rank. Paired ranks exchange these partials across NUMA groups.
Each group then gathers the BF16 result. Only one quarter of each tensor
crosses the NUMA boundary per rank.

Three rotating Lamport buffers use negative zero as an empty marker. Senders
normalize zero values to positive zero. A receiver clears each consumed
element before it advances. The workspace owns all buffers and counters;
calls must be serialized on one CUDA stream. CUDA Graph replay uses the same
device counters. Initialization compiles all tile buckets before ranks start
polling peer memory.

The workspace fixes one grid size at initialization for all payload shapes.
It uses the largest power of two no greater than the group's smallest SM
count, capped at 128 CTAs. This gives large decode payloads more parallel
PCIe writes. Every rank uses the same grid; each CTA uses four warps.

Receivers use the GPU global timer for a 60-second peer timeout. They sample
the timer once per 1,024 polls. This permits host compilation and profiler
callbacks to delay a rank without tying the timeout to polling throughput.

This tree changes FP32 addition order. It serves the ordinary performance
route only. The explicit rank-ordered collective and forced NCCL routes keep
their existing behavior. Unsupported payloads return to NCCL. No bitwise
invariance or trainer-alignment claim is made for this implementation.
