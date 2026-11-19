"""Attribute each b12x MXFP8 GEMM launch to a model matrix by its position in the layer.

Input: a CSV of one decode step or one prefill batch (device,start_ns,end_ns,name).
A segment starts at each MegaMoE launch. Inside a segment the launches follow the
order the model code issues them, so neighbouring kernels identify the matrix:

  decode : M  s  [shared_down]  (engram)  [wq_a_wkv]  r  [wq_b] ([index wq_b])  attn
           wo_a(cuBLAS)  [wo_b]  gate  k  [shared_up]  -> next M
  prefill: M  [shared_up]  s  [shared_down]  (engram)  [wq_a_wkv]  r  [wq_b] ...  attn
           wo_a(cuBLAS)  [wo_b]  gate  k  -> next M
"""

import collections
import csv
import statistics
import sys

path, mode = sys.argv[1], sys.argv[2]  # mode: decode | prefill
N_BASE = 40
PERIOD = 43 if mode == "decode" else 40
ANCHOR = "_block_step_kernel" if mode == "decode" else "ncclDevKernel_ReduceScatter"
ENGRAM_LAYERS = (1, 14)


def tag(n):
    if "b12x" in n:
        return "B"
    if n.startswith("cake_megamoe"):
        return "M"
    if n.startswith("_rmsnorm_fp8"):
        return "r"
    if n.startswith(
        ("_selected_attention_grouped", "_compact_attention", "decode_body")
    ):
        return "a"
    if n.startswith("_silu_and_mul"):
        return "s"
    if n.startswith("_index_score"):
        return "i"
    if n.startswith("_sqrt_softplus"):
        return "k"
    if n.startswith("ncclDevKernel"):
        return "N"
    return ""


def group(layer):
    if layer >= N_BASE:
        return "draft"
    if mode == "prefill":
        return "enc" if layer < 20 else "dec"
    return "base"


def load(path):
    """Yield (device, start_ns, end_ns, name) from a window CSV or an Nsight SQLite export."""
    if not path.endswith(".sqlite"):
        for r in csv.DictReader(open(path)):
            yield r["device"], int(r["start_ns"]), int(r["end_ns"]), r["name"]
        return
    import sqlite3

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = (
        "select k.deviceId, k.start, k.end, s.value, d.value from CUPTI_ACTIVITY_KIND_KERNEL k"
        " join StringIds s on s.id = k.shortName join StringIds d on d.id = k.demangledName"
    )
    for dev, start, end, short, demangled in con.execute(q):
        name = short
        if "b12x" in demangled or "b12x" in short:
            name = "b12x|" + short
        yield str(dev), start, end, name


by_dev = collections.defaultdict(list)
total = collections.Counter()
for dev, s, e, name in load(path):
    by_dev[dev].append((s, e, name))
    total[dev] += e - s

agg = collections.defaultdict(list)  # (category, group) -> [ns, ...]


def add(cat, layer, ns):
    agg[(cat, group(layer))].append(ns)


for dev, rows in by_dev.items():
    rows.sort()
    # The MoE launch right after the anchor kernel is layer 0.
    anchor = max(st for st, _, n in rows if n.startswith(ANCHOR))
    j0 = sum(1 for st, _, n in rows if n.startswith("cake_megamoe") and st < anchor)
    segs, cur = [], []
    for s, e, n in rows:
        t = tag(n)
        if not t:
            continue
        if t == "M":
            segs.append(cur)
            cur = []
        else:
            cur.append((t, e - s))
    segs.append(cur)
    for k, seg in enumerate(segs):
        moe_layer = (k - 1 - j0) % PERIOD  # the MoE that opened this segment
        attn_layer = (moe_layer + 1) % PERIOD  # the attention inside it
        has_index = any(t == "i" for t, _ in seg)
        seen = set()
        pre, mid = [], []
        for t, d in seg:
            if t != "B":
                if t == "r":
                    for j, dd in enumerate(pre):
                        if j == len(pre) - 1:
                            add("wq_a_wkv", attn_layer, dd)
                        elif attn_layer in ENGRAM_LAYERS:
                            add("engram", attn_layer, dd)
                        else:
                            add("other", attn_layer, dd)
                    pre = []
                if t == "a" and mid:
                    big = max(mid)
                    for dd in mid:
                        main = dd == big or not has_index
                        add("wq_b" if main else "index_wq_b", attn_layer, dd)
                    mid = []
                seen.add(t)
                continue
            if "k" in seen:  # decode: shared up is issued before the MoE launch
                add("shared_up", attn_layer, d)
            elif mode == "prefill" and k > 0 and not seen:
                add("shared_up", moe_layer, d)
                seen.add("up")
            elif (
                k > 0
                and "s" in seen
                and "down" not in seen
                and not (seen - {"s", "up"})
            ):
                add("shared_down", moe_layer, d)
                seen.add("down")
            elif "r" not in seen:
                pre.append(d)
            elif "a" not in seen:
                mid.append(d)
            else:
                add("wo_b", attn_layer, d)
        for dd in pre + mid:
            agg[("unclassified", "-")].append(dd)

ndev = len(by_dev)
if path.endswith(".sqlite"):
    # Whole capture: report per GPU-step, counted from the anchor kernel (5 per step).
    n_anchor = sum(
        1 for rows in by_dev.values() for _, _, n in rows if n.startswith(ANCHOR)
    )
    steps = n_anchor / 5 if mode == "decode" else n_anchor
    print(f"{steps:.0f} GPU-steps in the capture; columns are per GPU-step")
    ndev = steps
all_ns = sum(total.values())
b12x = sum(sum(v) for v in agg.values())
print(
    f"{path}: {ndev} GPUs, all-kernel time {all_ns/ndev/1e6:.2f} ms/GPU, "
    f"b12x {b12x/ndev/1e6:.3f} ms/GPU = {100*b12x/all_ns:.2f}%"
)
print(
    f"{'matrix':13s} {'group':6s} {'launch/GPU':>10s} {'median us':>10s} {'mean us':>8s} {'ms/GPU':>8s} {'share%':>7s} {'of b12x%':>8s}"
)
for (cat, g), v in sorted(agg.items(), key=lambda x: -sum(x[1])):
    ns = sum(v)
    print(
        f"{cat:13s} {g:6s} {len(v)/ndev:10.1f} {statistics.median(v)/1e3:10.1f} {ns/len(v)/1e3:8.1f} "
        f"{ns/ndev/1e6:8.3f} {100*ns/all_ns:7.2f} {100*ns/b12x:8.1f}"
    )
