"""Check how each wave of bench_closed_wave.py was spread over the attention-DP ranks, and report the result
over the waves that were spread evenly.

The client asks for request i on rank i % dp. The gateway ignores that field and places requests itself,
usually CC / dp per rank but not always. The run scripts sample /get_loads once a second (loads_sampler.py); this
script reports, per wave, the per-rank running-request counts while all CC requests were running, and then the
median over the first 3 measured waves that are VALID and evenly spread (the same median rule as the bench).
usage: check_spread.py BENCH_DIR LOADS_JSONL CC
"""

import collections
import glob
import json
import os
import sys

bench, loads, cc = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(loads)]
rows = [r for r in rows if r.get("running")]
names = [os.path.basename(p)[:-6] for p in glob.glob(f"{bench}/*.jsonl")]
ends = sorted((os.path.getmtime(f"{bench}/{w}.jsonl"), w) for w in names)
previous, even_waves = None, set()
for end, wave in ends:
    full = [
        r
        for r in rows
        if (previous or end - 600) < r["t"] <= end and sum(r["running"]) == cc
    ]
    patterns = collections.Counter(tuple(r["running"]) for r in full)
    dp = len(full[0]["running"]) if full else 0
    even = bool(full) and set(patterns) == {tuple([cc // dp] * dp)}
    if even:
        even_waves.add(wave)
    print(
        f"SPREAD {wave}: {len(full)} samples with all {cc} running; per-rank counts {patterns.most_common(2)}; "
        f"{'even' if even else 'NOT even' if full else 'no sample'}"
    )
    previous = end
summary = json.load(open(f"{bench}/summary.json"))
measured = summary["waves"]
uneven = [f"wave{w['wave']}" for w in measured if f"wave{w['wave']}" not in even_waves]
ok = [
    w for w in measured if w["status"] == "VALID" and f"wave{w['wave']}" in even_waves
][:3]
result = {
    "cc": cc,
    "measured_waves": len(measured),
    "uneven_waves": uneven,
    "even_valid_waves_used": [w["wave"] for w in ok],
}
if len(ok) == 3:
    med = sorted(ok, key=lambda w: w["throughput_tok_s"])[1]
    result.update(
        median_throughput_tok_s=round(med["throughput_tok_s"], 1),
        median_tpot_ms=round(med["tpot_ms"], 2),
        median_accept=round(med["accept"], 3),
        throughput_per_gpu=round(med["throughput_tok_s"] / summary["dp"], 1),
    )
else:
    result["status"] = (
        "fewer than 3 evenly spread valid waves: run again or raise WAVES"
    )
print("EVEN_RESULT " + json.dumps(result))
