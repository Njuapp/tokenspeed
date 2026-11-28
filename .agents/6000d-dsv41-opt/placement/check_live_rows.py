#!/usr/bin/env python3
"""Does a decode load profile count real rows only?

Under attention DP every rank replays the decode graph of the busiest rank's batch, so a rank with fewer requests
feeds filler rows through the MoE layers, and an idle rank feeds nothing else. A profile that counts them is
solved into a placement for traffic that does not exist (see ../docs/LOAD_PROFILE_FILLER_ROWS.md). This script
compares what a profile counted with what the requests of the run needed.

Counted: the decode rows of the target model in the profile, divided by the rows of one request in one decode
cycle (6 with DSpark block size 5). Expected: the decode cycles of the requests in the engine logs. A request of
N output tokens with average accept length a ran (N - 1) / a cycles; both come from the "Req: <id> Finish!
Accept_num_tokens_avg: <a>" lines. The output length is 1024 tokens, 32 for the requests of the client's
"warmup32" wave (../realtext/bench_closed_wave.py); the smoke request is ignored.

A rank writes its file at eager forwards only, so the files of the ranks stop at different moments, up to one
wave apart. The counted cycles must therefore lie between the cycles finished before the earliest file and those
finished before the latest one (6% tolerance for requests in flight). A recorder that counts filler rows reads
1.2 to 1.4 times the expected value on our real-text runs.

usage: check_live_rows.py PROFILE_DIR ENGINE_LOG [ENGINE_LOG ..] [--rows-per-cycle 6]
exit 0 when the profile is inside the bracket.
"""

import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

args = sys.argv[1:]
per_cycle = 6
if "--rows-per-cycle" in args:
    i = args.index("--rows-per-cycle")
    per_cycle = int(args[i + 1])
    del args[i : i + 2]
if len(args) < 2:
    sys.exit(__doc__)
profile = args[0]
ranks = [json.load(open(p)) for p in glob.glob(os.path.join(profile, "rank*.json"))]
if not ranks:
    sys.exit("no rank files in " + profile)
groups = ranks[0]["groups"]
group = max(groups, key=lambda g: groups[g]["num_experts"])
written = [r["written_unix"] for r in ranks]
rows = sum(
    r["groups"][group]["layers"][0]["modes"].get("decode", {}).get("rows", 0)
    for r in ranks
)
pattern = re.compile(
    r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z.*Req: (\S+) Finish! Accept_num_tokens_avg: ([0-9.]+)"
)
finished = []
for path in args[1:]:
    for line in open(path, errors="replace"):
        match = pattern.search(line)
        if not match or match.group(2).startswith("chatcmpl"):
            continue
        when = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
        tokens = 32 if match.group(2).startswith("warmup32") else 1024
        finished.append((when.timestamp(), (tokens - 1) / float(match.group(3))))
if not finished:
    sys.exit("no request finish lines in the logs")
low = sum(cycles for when, cycles in finished if when <= min(written))
high = sum(cycles for when, cycles in finished if when <= max(written))
# Each rank's file holds its own requests up to its own write; a rank serves 1 / ranks of the requests.
expected = sum(
    sum(cycles for when, cycles in finished if when <= w) for w in written
) / len(written)
counted = rows / per_cycle


def clock(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%T")


live = sorted({str(r.get("coverage", {}).get("live_rows_only")) for r in ranks})
print(
    f"profile {profile}: {len(ranks)} ranks, group {group}, files written {clock(min(written))} .. "
    f"{clock(max(written))} UTC, live_rows_only {live}"
)
print(
    f"decode cycles counted {counted:.0f}; cycles of the requests finished before the earliest file {low:.0f}, "
    f"before the latest {high:.0f}, expected from each rank's own write time {expected:.0f}; "
    f"counted / expected = {counted / expected:.3f}"
)
ok = 0.94 * low <= counted <= 1.06 * high
print("REAL ROWS ONLY" if ok else "NOT CONFIRMED")
sys.exit(0 if ok else 1)
