#!/usr/bin/env python3
"""Rank loads that placement files give on a decode load profile.

For each placement file (or "none" for contiguous ownership): the load of the busiest EP rank over the mean, as
the mean over the layers and for the worst layer, and the share of the routes that leave the rank they start
on. A route goes to the copy of its expert on its own rank when there is one, otherwise to the expert's home
slot. Use it to judge a placement on another profile than the one it was solved from.

usage: eval_placement.py PROFILE_DIR EP_SIZE PLACEMENT.json|none [PLACEMENT.json|none ..]
"""

import glob
import json
import os
import sys

if len(sys.argv) < 4:
    sys.exit(__doc__)
profile, ep = sys.argv[1], int(sys.argv[2])
ranks = sorted(
    (json.load(open(p)) for p in glob.glob(os.path.join(profile, "rank*.json"))),
    key=lambda r: r["rank"],
)
assert len(ranks) == ep, (len(ranks), ep)
groups = ranks[0]["groups"]
group = max(groups, key=lambda g: groups[g]["num_experts"])
layers = groups[group]["num_layers"]
experts = groups[group]["num_experts"]
for name in sys.argv[3:]:
    placement = None if name == "none" else json.load(open(name))
    ratios, cross = [], []
    for layer in range(layers):
        if placement is None:
            slots, physical, home = experts, list(range(experts)), list(range(experts))
        else:
            physical = placement["physical_to_logical"][layer]
            slots = len(physical)
            home = placement["home_slot"][layer] if "home_slot" in placement else None
            if home is None:
                home = [0] * experts
                for slot, expert in enumerate(physical):
                    home[expert] = slot
        per_rank = slots // ep
        # rank -> {expert: its first slot on that rank}
        own = [dict() for _ in range(ep)]
        for slot, expert in enumerate(physical):
            own[slot // per_rank].setdefault(expert, slot)
        load = [0] * ep
        local = total = 0
        for source, rank in enumerate(ranks):
            routes = rank["groups"][group]["layers"][layer]["modes"]["decode"]["routes"]
            for expert, count in enumerate(routes):
                if not count:
                    continue
                slot = own[source].get(expert, home[expert])
                load[slot // per_rank] += count
                total += count
                if slot // per_rank == source:
                    local += count
        ratios.append(max(load) / (sum(load) / ep))
        cross.append(1 - local / total)
    print(
        f"{os.path.basename(name):24s} busiest rank / mean: mean over layers {sum(ratios) / layers:.3f}, "
        f"worst layer {max(ratios):.3f}; routes that cross ranks {sum(cross) / layers:.1%}"
    )
