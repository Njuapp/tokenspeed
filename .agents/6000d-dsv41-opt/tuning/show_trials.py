import json
import sys

path, T = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 9341
rows = {}
for line in open(path):
    r = json.loads(line)
    if r["tokens"] != T or "|recheck" in r["key"] or f"seed{seed}|" not in r["key"]:
        continue
    k = json.dumps(
        dict(sorted((a, b) for a, b in r["knobs"].items() if b is not None)),
        sort_keys=True,
    )
    rows.setdefault(k, {})["power_law" if "|power_law" in r["key"] else "balanced"] = r[
        "p50_us"
    ]
d = rows['{"backend": "p2p_split"}']
out = []
for k, us in rows.items():
    b = us.get("balanced")
    p = us.get("power_law")
    out.append((p if p is not None else 9e9, b, p, k))
for _, b, p, k in sorted(out, key=lambda x: (x[0], x[1] or 9e9)):
    fb = (
        f"{b:8.1f} ({100 * (b / d['balanced'] - 1):+6.1f}%)"
        if b is not None and b != float("inf")
        else "    fail          "
    )
    fp = (
        f"{p:8.1f} ({100 * (p / d['power_law'] - 1):+6.1f}%)"
        if p is not None and p != float("inf")
        else "       -          "
    )
    short = k.replace('"backend": "p2p_split", ', "").replace('"', "")
    print(fb, fp, short[:230])
