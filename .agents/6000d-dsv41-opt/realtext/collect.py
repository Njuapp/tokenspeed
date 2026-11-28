#!/usr/bin/env python3
"""One line per concurrency level of real-text runs, for tables.

Applies the selection rule of this workload to every level of a run directory: the median of the first three
valid waves that the gateway spread evenly over the ranks. Step time is TPOT x accept per wave (the time of one
decode cycle of a request).

A run directory holds either cc<CC>/bench_summary.txt (one measurement per level) or cc<CC>-h<H>/bench_summary.txt
(the level measured on several parts of the prompt set, see README.md; the parts are averaged).

usage: collect.py GPUS RUN_DIR [RUN_DIR ..]     GPUS = number of GPUs of the engine (16 or 8)
"""

import json
import re
import statistics
import sys
from pathlib import Path


def level(path):
    waves, spread = [], {}
    for line in path.read_text().splitlines():
        if line.startswith('{"status"'):
            waves.append(json.loads(line))
        elif line.startswith("SPREAD wave"):
            spread[int(line.split()[1][4:].rstrip(":"))] = line.rstrip().endswith(
                "; even"
            )
    used = [w for w in waves if w["status"] == "VALID" and spread.get(w["wave"])][:3]
    if not used:
        return None

    def median(value):
        return statistics.median(value(w) for w in used)

    return {
        "thr": median(lambda w: w["throughput_tok_s"]),
        "tpot": median(lambda w: w["tpot_ms"]),
        "accept": median(lambda w: w["accept"]),
        "step": median(lambda w: w["tpot_ms"] * w["accept"]),
        "used": [w["wave"] for w in used],
        "waves": len(waves),
    }


def peak_memory(run):
    path = run / "gpu-memory-end.txt"
    if not path.exists():
        return None
    return max(
        int(line.split(",")[1].split()[0])
        for line in path.read_text().splitlines()
        if line.strip()
    )


gpus = int(sys.argv[1])
for name in sys.argv[2:]:
    run = Path(name)
    levels = {}
    for path in run.glob("cc*/bench_summary.txt"):
        match = re.fullmatch(r"cc(\d+)(?:-h(\d+))?", path.parent.name)
        if match:
            levels.setdefault(int(match.group(1)), []).append(
                (int(match.group(2) or 0), path)
            )
    print(f"## {run.name}: {gpus} GPUs, peak memory at the end {peak_memory(run)} MiB")
    for cc in sorted(levels):
        parts = [level(path) for _, path in sorted(levels[cc])]
        if any(part is None for part in parts):
            print(
                f"  cc {cc}: no evenly spread valid wave in {parts.count(None)} part(s)"
            )
            continue

        def mean(key):
            return sum(part[key] for part in parts) / len(parts)

        short = (
            ""
            if all(len(part["used"]) == 3 for part in parts)
            else "  (fewer than 3 even waves)"
        )
        print(
            f"  per GPU {cc / gpus:g}  cc {cc}  tok/s {mean('thr'):.1f}  tok/s per GPU {mean('thr') / gpus:.1f}"
            f"  TPOT ms {mean('tpot'):.2f}  accept {mean('accept'):.3f}  step ms {mean('step'):.2f}"
            f"  waves used {[part['used'] for part in parts]} of {parts[0]['waves']}{short}"
        )
