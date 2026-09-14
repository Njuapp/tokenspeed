# Kimi-K3 BT/HT first-half optimization

This opt-in path changes only routed-expert finalize, the first all-reduce
(AR1), and latent RMSNorm. Enable it before constructing the model:

```bash
export TOKENSPEED_K3_BT_HT=1
```

Unset the variable, or set it to `0`, for main's original dispatch. The flag
does not enable a fused up projection, ReduceScatter or AllGather. Model
forward, scheduler, graph configuration and agentic benchmark files are
unchanged by this PR.

## Dispatch and second-half contract

| Actual/padded kernel M | First half | Second half |
| --- | --- | --- |
| 1–32 | Original TAIL_FUSION eligibility/kernel or original fallback | Original |
| 33–1024 | BT deferred finalize + AR1 + RMSNorm | Original sharded cuBLAS projection, owner shared/residual arithmetic and AR2 |
| 1025–8192 | HT deferred finalize + AR1 + RMSNorm | Same original second half |
| Zero, above capacity, or unsupported configuration | Original dispatch | Original |

M is the token count seen by the kernel, not concurrency or total conversation
length. The policy covers continuous ranges, not a graph-bucket whitelist.
The original selector runs first. Its small-tail choice has priority, and its
ordinary/multimem decision is retained for AR2. In particular, decode and
speculative verification do not acquire a prefill-only multimem AR2 just
because their M crosses 256.

Both deferred tiers consume the producer's existing triple: BF16 expert rows,
BF16 top-k weights and int32 row indices. The producer's scaling remains in
the weights. BT/HT returns normalized `[M,3584]` routed output; shared and
residual tensors never enter these first-half kernels.

For the original multimem second half, shared staging remains before the
first-half collective. After BT/HT completes, each rank adds the residual to
its 896-column shared owner slice in BF16, calls the original cuBLAS `addmm_`,
runs the original staged AR2, then clones the result. For the ordinary second
half, the existing `_project_and_inject_local_block` and `all_reduce` are used.
Projection remains sharded. The second-half arithmetic/rounding order and
output lifetime are not changed.

## Kernel and capability boundaries

Runtime reaches optional backends only through `tokenspeed_kernel.ops`.
The existing FlashInfer BT device implementation is not modified. HT retains
the selected native-3584 specialization: each TP8 latent shard contains 56
BF16x8 vectors, so the final reduction warp is predicated.

The retained presets are:

- BT: 256 finalize threads, 2 elements/thread, prefetch group 1, 224 reduction
  threads, 448 RMS threads, PDL enabled.
- HT: 448 consumer threads, 1 vector/thread, 10 stages, 2 reduction warps,
  2 RMS token groups, 3 RMS pipeline stages, PDL enabled.

There is no kernel retuning in this extraction. The fixed application contract
is BF16, latent 3584, hidden 7168, top-k 16, TP8/EP1, DP1/CP1, sharded up
projection and a deferred-capable producer. It also requires a WORLD-spanning
multicast-capable Blackwell group, the validated FlashInfer 0.6.18 API and PDL.
Optional imports remain lazy behind the kernel boundary.

Rank agreement and the retained joint BT/HT capability vote precede backend
construction. Unsupported configurations fall back uniformly; exceptions
after collective construction starts propagate rather than strand peers via
rank-local fallback.

## Workspace and graph lifetime

BT/HT protocol storage and normalized latent outputs are process-wide, shared
by sequential layers. Construction/compilation happen before graph capture;
attempting cold collective construction during capture is rejected. Kernel
replay does not allocate or enter host collectives. Keep the protocol buffers
alive for every graph replay; concurrent workspace use is not supported.

This branch has no large-M second-half fused workspace, multicast-output ring,
or independent `[8192,7168]` symmetric output allocation per layer. The normal
small-M latent-tail resources and original multimem staging buffers remain.
BT/HT still has its own memory cost: removing the second-half allocations does
not establish memory neutrality or predict an exact KV-capacity recovery.

## Validation and benchmark scope

CPU-only checks, which do not require optional GPU packages:

```bash
PYTHONPATH=python python -m pytest --noconftest \
  test/runtime/test_k3_moe_tail_tier.py \
  test/runtime/test_k3_first_half.py \
  tokenspeed-kernel/test/thirdparty/test_mnnvl_first_half_host_contracts.py -q
```

The tests cover continuous routing, small-M priority, decode's original AR2,
rank disagreement, capture rejection, retained tuning/resource validation,
no second-half allocation, and both BT/HT second-half operation orders.

On a separately allocated and configured TP8 fabric:

```bash
timeout 20m torchrun --standalone --nproc-per-node=8 \
  test/runtime/smoke_k3_first_half.py --output-dir /tmp/k3-first-half-smoke
```

Use the normal rendezvous/node arguments for multi-node TP8. This synthetic
real-communication smoke exercises eager/128 graph replays, changed inputs
and weights, boundary M values and two layer slots under unchanged numerical
limits. It is not a real SiTU-producer or serving benchmark. GPU and full-model
validation of this extracted branch remain pending; old full-fusion results
must not be attributed to it. First-half reduction association can still
change logits, so removing second-half fusion is not a numerical pass.

Formal performance acceptance compares the combined BT/HT first-half change
against unmodified main under the original K3 + EAGLE3 agentic protocol, not
an isolated BT cohort. Preserve the frozen EvalScope client and dataset,
CC 1/2/4/8/16 with 4/8/8/16/32 conversations, held-out warmup at offset 68,
500 generated tokens per turn and genuine generated conversation histories.
The original server protocol uses TP8, FP8 KV, EAGLE3 steps 3/draft 4/top-k 1,
prefill chunk 8192, prefill graph cap 2048, the K3 reasoning parser and disabled
KV store. Do not silently replace it with graph cap 8192 or passthrough parsing.
Freeze any seed and runtime-version controls explicitly and equally for both
arms, and report differences from the historical reference. The repository's
benchmark script/configuration is unchanged here; a strict launcher must
resolve its effective settings against that reference before launch.

This branch makes no new serving speedup or model-quality claim and remains
disabled by default. Numerical/task-quality qualification is separate from
performance-only comparison.
