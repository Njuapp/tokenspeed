#!/usr/bin/env python3
"""Is the MoE routing recorded in a load profile even over the experts?

Prints, per expert group and mode, the busiest and the least busy expert over the mean, the busiest owner rank
over the mean (contiguous ownership) and the number of experts that received routes. Used to confirm that an
engine started with TOKENSPEED_MOE_ROUTING_SIMULATION=balanced routes evenly.

usage: check_balance.py PROFILE_DIR
exit 1 when the busiest expert of the largest group's decode routes is more than 1.15 times the mean.
"""

import glob
import json
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ranks = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/rank*.json"))]
assert ranks, "no rank files"
ep = ranks[0]["ep_size"]
assert len(ranks) == ep, (len(ranks), ep)
groups = ranks[0]["groups"]
target = max(groups, key=lambda g: groups[g]["num_experts"])
uneven = 0
for group, first in groups.items():
    experts = first["num_experts"]
    per_rank = experts // ep
    for mode in ("decode", "extend"):
        rows = []
        for layer in range(first["num_layers"]):
            total = [0] * experts
            for rank in ranks:
                counted = rank["groups"][group]["layers"][layer]["modes"].get(mode)
                if counted:
                    for expert, count in enumerate(counted["routes"]):
                        total[expert] += count
            routes = sum(total)
            if routes:
                owner = [
                    sum(total[k * per_rank : (k + 1) * per_rank]) for k in range(ep)
                ]
                rows.append(
                    (
                        max(total) * experts / routes,
                        min(total) * experts / routes,
                        max(owner) * ep / routes,
                        sum(1 for count in total if count),
                        routes,
                    )
                )
        if rows:
            worst = max(row[0] for row in rows)
            print(
                f"{group} {mode}: {len(rows)} layers, routes per layer {rows[0][4]}, busiest expert / mean "
                f"{min(row[0] for row in rows):.3f}..{worst:.3f}, least busy / mean {min(row[1] for row in rows):.3f}, "
                f"busiest owner rank / mean {max(row[2] for row in rows):.3f}, "
                f"active experts {min(row[3] for row in rows)}..{max(row[3] for row in rows)} of {experts}"
            )
            if group == target and mode == "decode" and worst > 1.15:
                uneven = 1
print("EVEN" if not uneven else "NOT EVEN")
sys.exit(uneven)
