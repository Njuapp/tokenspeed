#!/usr/bin/env python3
"""Per CC level of real-text runs: RESULT (median of all valid waves), EVEN_RESULT (first 3 even waves),
and each wave with its spread. usage: show.py RUN_DIR [RUN_DIR ..]   (a run directory holds cc<CC>/bench_summary.txt)
"""

import glob
import json
import os
import re
import sys

for d in sys.argv[1:]:
    run = os.path.basename(d.rstrip("/"))
    ready = ""
    print(f"## {run}")
    for path in sorted(
        glob.glob(f"{d}/cc*/bench_summary.txt"),
        key=lambda p: int(re.search(r"cc(\d+)", p).group(1)),
    ):
        cc = int(re.search(r"cc(\d+)", path).group(1))
        res = even = None
        waves = []
        spread = {}
        for line in open(path):
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
            elif line.startswith("EVEN_RESULT "):
                even = json.loads(line[12:])
            elif line.startswith("SPREAD wave"):
                m = re.match(r"SPREAD (wave\d+): .*; (even|NOT even|no sample)", line)
                spread[m.group(1)] = m.group(2)
            elif line.startswith('{"status"'):
                w = json.loads(line)
                waves.append(w)
        per = " ".join(
            f"{w['throughput_tok_s']:.0f}{'' if spread.get('wave%d' % w['wave']) == 'even' else '*'}"
            for w in waves
        )
        r = (
            f"all: {res['median_throughput_tok_s']:.1f} tok/s TPOT {res['median_tpot_ms']:.2f} acc {res['median_accept']:.3f}"
            if res
            else "all: none"
        )
        e = (
            f"even: {even['median_throughput_tok_s']:.1f} tok/s TPOT {even['median_tpot_ms']:.2f} (waves {even['even_valid_waves_used']})"
            if even and "median_throughput_tok_s" in even
            else f"even: fewer than 3 (even waves {even['even_valid_waves_used'] if even else '?'})"
        )
        print(f"CC{cc:<4} {cc // 16:>2}/GPU | {r} | {e} | waves [{per}]  (* = uneven)")
