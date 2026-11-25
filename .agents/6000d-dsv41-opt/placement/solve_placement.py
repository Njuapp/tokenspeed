#!/usr/bin/env python3
"""Turn MoE load profiles (TOKENSPEED_MOE_LOAD_PROFILE) into an expert placement file
(TOKENSPEED_MOE_EXPERT_PLACEMENT).  Python standard library only; runs on the host.

    solve_placement.py PROFILE_DIR --ep-size 8  --output placement-ep8.json
    solve_placement.py PROFILE_DIR --ep-size 8  --replicas 2 --output placement-ep8-r2.json
    solve_placement.py PROFILE_DIR --ep-size 16 --output placement-ep16.json --rows-per-launch 1056
    solve_placement.py --identity --ep-size 8 --output placement-identity.json   (control: contiguous ownership)

PROFILE_DIR holds rank<r>.json of every EP rank (for a two-node EP16 engine copy both nodes' files into one
directory).  Per layer the experts are assigned to the EP ranks so that the largest per-rank route count is as
small as the heuristic gets it: longest-processing-time greedy under the slot limit, then swaps and moves out of
the busiest rank until none helps.

Without --replicas every rank holds the same number of experts and the file is a permutation per layer
(format 1): rank r owns slots [r * n / EP, (r + 1) * n / EP) of physical_to_logical[layer].

With --replicas H (format 2) every rank gets H more slots.  In each layer the
experts whose load exceeds --replica-threshold of the mean rank load, at most H of them, are copied to every
rank, and each rank routes its own tokens' hits on them to its own copy: those routes cross no GPU, and a rank's
load from such an expert is its own tokens' share of it.  The other experts are then balanced around that.  A
layer with more hot experts than H leaves one of them alone on a rank; while that rank is above
--replica-balance x the mean it gives its hits on a replicated expert to another rank's copy (it then holds no
copy).  Slots a layer does not need hold a second copy of one of the rank's own experts that no route reaches.
The profile must come from the EP size the file is solved for (it is used per source rank).

Why the busiest rank matters: every SM of the RTX 6000D may spend 1.18e11 FLOP x operand bits per 8.32 ms window
and then runs arithmetic at 2.6% speed until the window ends.  One MoE launch charges a rank
routes x 6 x hidden x intermediate FLOP x 8 bits (MXFP8 x MXFP4); a rank that needs more than the 156 SMs' quota
inside one launch waits for the next window while the other ranks wait for it.

The report gives per layer, for contiguous ownership ("before"), the permutation alone ("perm") and the result:
  max/mean   the busiest rank's routes over the mean rank's;
  quota      the busiest rank's arithmetic of one launch as a fraction of one window's quota of the whole GPU
             (valid FLOPs; "padded" counts whole tasks of 128 or 64 rows, the upper bound of what the kernel executes);
  cross      the fraction of routes whose expert sits on another GPU than the token;
  expert     the busiest expert alone, and "floor": that expert plus the lightest experts that fill its rank,
             which no permutation can go below.
Layers are flagged when the busiest rank stays above --quota-threshold after placement and when no permutation
can bring it under (replicas are then the remedy).
"""

import argparse
import glob
import json
import os
import sys

PLACEMENT_FORMAT = "tokenspeed-moe-expert-placement/1"
PLACEMENT_FORMAT_REPLICAS = "tokenspeed-moe-expert-placement/2"
PROFILE_FORMAT = "tokenspeed-moe-load-profile/1"


def read_profile_sources(directory, group, modes):
    """({layer: {rank: [routes of that rank's own rows per logical expert]}}, info)."""
    paths = sorted(glob.glob(os.path.join(directory, "rank*.json")))
    if not paths:
        sys.exit(f"no rank*.json in {directory}")
    sources, info, ranks, rows, launches, replayed = {}, None, [], {}, {}, set()
    for path in paths:
        data = json.load(open(path))
        if data.get("format") != PROFILE_FORMAT:
            sys.exit(f"{path}: not a {PROFILE_FORMAT} file")
        groups = data["groups"]
        key = group or max(groups, key=lambda k: groups[k]["num_experts"])
        if key not in groups:
            sys.exit(f"{path}: no group {key} (has {sorted(groups)})")
        g = groups[key]
        this = {
            "group": key,
            "num_experts": g["num_experts"],
            "top_k": g["top_k"],
            "num_layers": g["num_layers"],
            "ep_size": data["ep_size"],
        }
        if info is None:
            info = this
        elif this != info:
            sys.exit(f"{path}: {this} differs from {info}")
        ranks.append(data["rank"])
        # Modes whose routes include CUDA-graph replays (trees that write "coverage"): such a file is only as
        # fresh as the rank's last eager forward.
        replayed.update(data.get("coverage", {}).get("modes_with_graph_replays", []))
        for layer in g["layers"]:
            acc = sources.setdefault(layer["layer"], {}).setdefault(
                data["rank"], [0] * g["num_experts"]
            )
            forwards = 0
            for mode, entry in layer["modes"].items():
                if modes and mode not in modes:
                    continue
                for expert, count in enumerate(entry["routes"]):
                    acc[expert] += count
                rows[mode] = rows.get(mode, 0) + entry["rows"]
                forwards += entry["forwards"]
            # Every rank takes part in every collective launch, so any rank's count is the launch count;
            # the largest guards against a rank whose last window was not written.
            launches[layer["layer"]] = max(launches.get(layer["layer"], 0), forwards)
    if sorted(ranks) != sorted(set(ranks)):
        sys.exit(f"duplicate ranks in {directory}: {sorted(ranks)}")
    info.update(
        ranks=sorted(ranks),
        rows_by_mode=rows,
        files=len(paths),
        launches_by_layer=launches,
        graph_replay_modes=sorted(replayed),
    )
    return sources, info


def read_profiles(directory, group, modes):
    """Sum the per-rank profiles: ({layer: [routes per logical expert]}, info)."""
    sources, info = read_profile_sources(directory, group, modes)
    layers = {
        layer: [sum(column) for column in zip(*by_rank.values())]
        for layer, by_rank in sources.items()
    }
    return layers, info


def rank_loads(assignment, load, ranks):
    out = [0] * ranks
    for expert, rank in enumerate(assignment):
        if rank >= 0:
            out[rank] += load[expert]
    return out


def _improve(load, members, total, capacity):
    """Swaps and moves out of the busiest rank while that lowers the pair's maximum; in place."""
    ranks = len(total)
    for _ in range(100000):
        busiest = max(range(ranks), key=lambda r: (total[r], -r))
        best = None
        for other in range(ranks):
            if other == busiest:
                continue
            gap = total[busiest] - total[other]
            room = len(members[other]) < capacity
            for a in members[busiest]:
                if room and 0 < load[a] < gap:
                    peak = max(total[busiest] - load[a], total[other] + load[a])
                    if best is None or peak < best[0]:
                        best = (peak, a, None, other)
                for b in members[other]:
                    delta = load[a] - load[b]
                    if 0 < delta < gap:
                        peak = max(total[busiest] - delta, total[other] + delta)
                        if best is None or peak < best[0]:
                            best = (peak, a, b, other)
        if best is None or best[0] >= total[busiest]:
            break
        _, a, b, other = best
        members[busiest].remove(a)
        members[other].append(a)
        total[busiest] -= load[a]
        total[other] += load[a]
        if b is not None:
            members[other].remove(b)
            members[busiest].append(b)
            total[busiest] += load[b]
            total[other] -= load[b]


def _greedy(load, experts, ranks, capacity, initial):
    members, total = [[] for _ in range(ranks)], list(initial)
    # Heaviest expert first, onto the lightest rank that still has room
    # (ties: the lower rank, for a deterministic result on every host).
    for expert in sorted(experts, key=lambda e: (-load[e], e)):
        rank = min(
            (r for r in range(ranks) if len(members[r]) < capacity),
            key=lambda r: (total[r], r),
        )
        members[rank].append(expert)
        total[rank] += load[expert]
    return [sorted(m) for m in members], total


def balance(load, ranks):
    """expert -> rank with len(load) / ranks experts per rank, small maximum rank load."""
    n = len(load)
    members, total = _greedy(load, range(n), ranks, n // ranks, [0] * ranks)
    _improve(load, members, total, n // ranks)
    # Number the ranks so that ranks 2i and 2i+1 are the i-th busiest and the i-th lightest: an EP group
    # of half the size (whose rank i is the union of these two) then stays balanced as well.
    order = sorted(range(ranks), key=lambda r: (-total[r], r))
    position = {}
    for i in range(ranks // 2):
        position[order[i]] = 2 * i
        position[order[ranks - 1 - i]] = 2 * i + 1
    if ranks % 2:
        position[order[ranks // 2]] = ranks - 1
    assignment = [-1] * n
    for rank, experts in enumerate(members):
        for expert in experts:
            assignment[expert] = position[rank]
    return assignment


def choose_replicas(load, ranks, cap, threshold):
    """The experts to copy to every rank: load above threshold x mean rank load, heaviest first, at most cap."""
    mean = sum(load) / ranks
    hot = [
        e
        for e in sorted(range(len(load)), key=lambda e: (-load[e], e))
        if load[e] > threshold * mean and load[e] > 0
    ]
    return hot[:cap]


def place_with_replicas(by_rank, ranks, slots_per_rank, replicated, balance_target):
    """Place one layer whose `replicated` experts get a copy on every rank that uses its own.

    by_rank[r][e] is the route count of rank r's own rows to expert e.  A rank serves its own rows' hits on a
    replicated expert from its own copy.  While the busiest rank is above balance_target x the mean rank load it
    gives the hits on one replicated expert away: it then holds no copy and routes them to the expert's home copy
    on another rank (one home per expert, which keeps its own copy).

    Returns (slot row of all ranks concatenated, home slot per expert, per-rank loads, expert -> owner rank or
    -1 for a replicated one, {replicated expert: [serving rank per source rank]}).
    """
    n = len(by_rank[0])
    load = [sum(column) for column in zip(*by_rank)]
    mean = sum(load) / ranks
    hot = set(replicated)
    route = {e: list(range(ranks)) for e in replicated}
    home_rank = {}
    capacity = slots_per_rank - len(replicated)
    initial = [sum(by_rank[r][e] for e in replicated) for r in range(ranks)]
    members, total = _greedy(
        load, [e for e in range(n) if e not in hot], ranks, capacity, initial
    )
    _improve(load, members, total, capacity)
    while True:
        busiest = max(range(ranks), key=lambda r: (total[r], -r))
        if total[busiest] <= balance_target * mean:
            break
        step = None
        for hits, e in sorted(
            ((by_rank[busiest][e], e) for e in replicated), key=lambda t: (-t[0], t[1])
        ):
            if hits <= 0 or route[e][busiest] != busiest or home_rank.get(e) == busiest:
                continue
            target = home_rank.get(e)
            if target is None:
                target = min(
                    (r for r in range(ranks) if r != busiest and route[e][r] == r),
                    key=lambda r: (total[r], r),
                )
            if total[target] + hits < total[busiest]:
                step = (hits, e, target)
                break
        if step is None:
            break
        hits, e, target = step
        home_rank[e] = target
        route[e][busiest] = target
        total[busiest] -= hits
        total[target] += hits
        _improve(load, members, total, capacity)
    rows, home, owner = [], [0] * n, [-1] * n
    for rank in range(ranks):
        regular = sorted(members[rank])
        copies = [e for e in replicated if route[e][rank] == rank]
        # The rank's own experts, then its replicas, then fillers: one more copy of the expert in its first
        # slot, which no route reaches because a rank routes to its first slot holding an expert.
        held = regular + copies
        slots = held + [held[0]] * (slots_per_rank - len(held))
        for index, expert in enumerate(regular):
            home[expert] = rank * slots_per_rank + index
            owner[expert] = rank
        for index, expert in enumerate(copies):
            if home_rank.get(expert, 0) == rank:
                home[expert] = rank * slots_per_rank + len(regular) + index
        rows.extend(slots)
    return rows, home, total, owner, route


def cross_fraction(by_rank, owner, route=None):
    """Fraction of routes served on another rank than the token's (route: serving ranks of replicated experts)."""
    total = sum(sum(row) for row in by_rank)
    local = sum(
        count
        for rank, row in enumerate(by_rank)
        for expert, count in enumerate(row)
        if (owner[expert] if owner[expert] >= 0 else route[expert][rank]) == rank
    )
    return 1.0 - local / total if total else 0.0


def padded_rank_routes(by_rank, owner, route, ranks, pad):
    """Largest per-rank count of task rows: what each rank serves of every expert, rounded up to whole tasks."""
    served = [{} for _ in range(ranks)]
    for source, row in enumerate(by_rank):
        for expert, count in enumerate(row):
            rank = owner[expert] if owner[expert] >= 0 else route[expert][source]
            served[rank][expert] = served[rank].get(expert, 0) + count
    return max(sum(pad(count) for count in of_rank.values()) for of_rank in served)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_dir", nargs="?")
    ap.add_argument(
        "--identity",
        action="store_true",
        help="write the identity placement (contiguous ownership through the placement path); needs no profile",
    )
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument(
        "--ep-size",
        type=int,
        required=True,
        help="EP group size the placement is solved for (8 prefill, 16 decode)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--group",
        default=None,
        help="profile group, e.g. e384k6 (default: the group with the most experts, i.e. the target model)",
    )
    ap.add_argument(
        "--modes",
        default="",
        help="comma list of forward kinds to count (extend, decode, mixed, other); default all",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="layers of the model (default: the profile's)",
    )
    ap.add_argument("--allow-missing-ranks", action="store_true")
    ap.add_argument(
        "--replicas",
        type=int,
        default=0,
        help="replica slots added to every rank; per layer at most this many hot experts are copied to every rank (0: permutation only, format 1)",
    )
    ap.add_argument(
        "--replica-threshold",
        type=float,
        default=0.5,
        help="replicate an expert whose load exceeds this fraction of the mean rank load (0: always the --replicas hottest)",
    )
    ap.add_argument(
        "--replica-balance",
        type=float,
        default=1.25,
        help="a rank busier than this multiple of the mean rank load gives its hits on a replicated expert to another rank's copy "
        "(balance bought with cross-GPU routes; happens when a layer has more hot experts than replica slots)",
    )
    ap.add_argument(
        "--allow-graph-replay-profile",
        action="store_true",
        help="solve although most counted rows are of a forward kind the engine ran from CUDA graphs: such a profile is written "
        "only at the last eager forward and may hold little more than the start-up warm-up",
    )
    ap.add_argument(
        "--rows-per-launch",
        type=int,
        default=None,
        help="rows of one collective launch summed over the EP ranks, for the quota columns (e.g. 16384 for 8 x 2048, 1056 for 16 x 66); "
        "default: the profile's rows / launches, which is not known for CUDA-graph replays",
    )
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument(
        "--operand-bits",
        type=int,
        default=8,
        help="bits charged per FLOP (8 for MXFP8 x MXFP4)",
    )
    ap.add_argument(
        "--sm-quota",
        type=float,
        default=1.18e11,
        help="FLOP x bits one SM may spend per 8.32 ms window",
    )
    ap.add_argument("--sms", type=int, default=156)
    ap.add_argument(
        "--quota-threshold",
        type=float,
        default=0.7,
        help="flag a layer whose busiest rank needs more than this fraction of one window's quota per launch "
        "(valid FLOPs; the rest is room for padded tasks and for the dense GEMMs that share the window; "
        "measured: a rank at 76%% valid / 84%% padded already lengthens the launch from 5.75 to 7.1 ms)",
    )
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    if args.identity:
        n, num_layers = args.num_experts, args.num_layers or 40
        if n % args.ep_size:
            sys.exit(f"{n} experts do not divide over EP {args.ep_size}")
        # Contiguous ownership is the same for every EP size dividing the expert count.
        ep_sizes = [e for e in range(n, 1, -1) if n % e == 0 and e <= 128]
        out = {
            "format": PLACEMENT_FORMAT,
            "num_experts": n,
            "num_layers": num_layers,
            "ep_sizes": ep_sizes,
            "physical_to_logical": [list(range(n)) for _ in range(num_layers)],
            "meta": {"identity": True},
        }
        with open(args.output + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(args.output + ".tmp", args.output)
        print(
            f"wrote identity placement {args.output} ({num_layers} layers x {n} experts)"
        )
        return
    if not args.profile_dir:
        sys.exit("PROFILE_DIR is required unless --identity is given")
    sources, info = read_profile_sources(args.profile_dir, args.group, modes)
    n, num_layers, top_k, ranks = (
        info["num_experts"],
        args.num_layers or info["num_layers"],
        info["top_k"],
        args.ep_size,
    )
    replay_rows = sum(
        count
        for mode, count in info["rows_by_mode"].items()
        if mode in info["graph_replay_modes"]
    )
    if replay_rows:
        share = replay_rows / max(1, sum(info["rows_by_mode"].values()))
        text = (
            f"{share:.0%} of the counted rows are of forward kinds this engine ran from CUDA graphs ({info['graph_replay_modes']}). Graph replays "
            "write no profile: the files hold those kinds up to the last eager forward only, for a graph-only engine the start-up warm-up."
        )
        if share > 0.5 and not args.allow_graph_replay_profile:
            sys.exit(
                f"refusing to solve: {text} Pass --allow-graph-replay-profile if the profile is known to cover served load."
            )
        print(f"WARNING: {text}", file=sys.stderr)
    if n % ranks:
        sys.exit(f"{n} experts do not divide over EP {ranks}")
    if info["ranks"] != list(range(info["ep_size"])) and not args.allow_missing_ranks:
        sys.exit(
            f"profiles of ranks {info['ranks']} found, the engine ran EP {info['ep_size']}; pass --allow-missing-ranks to solve anyway"
        )
    if args.replicas and info["ep_size"] != ranks:
        sys.exit(
            f"replicas need the per-rank profile of the EP size they are solved for: profile EP {info['ep_size']}, --ep-size {ranks}"
        )
    per_rank = n // ranks
    # GEMM1 (hidden -> 2 x intermediate) + GEMM2 (intermediate -> hidden), 2 FLOP per multiply-add.
    flop_bits_per_route = 6 * args.hidden * args.intermediate * args.operand_bits
    quota = args.sms * args.sm_quota

    def by_rank_of(layer):
        rows_of = sources.get(layer, {})
        if info["ep_size"] == ranks:
            return [rows_of.get(r, [0] * n) for r in range(ranks)]
        # Another EP size than the profile's: only the totals are meaningful (source ranks unknown).
        total = (
            [sum(column) for column in zip(*rows_of.values())] if rows_of else [0] * n
        )
        return [total] + [[0] * n for _ in range(ranks - 1)]

    # Pass 1: which experts each layer replicates; the slot count is the same for every layer.
    chosen = {}
    for layer in range(num_layers):
        load = [sum(column) for column in zip(*by_rank_of(layer))]
        chosen[layer] = (
            choose_replicas(load, ranks, args.replicas, args.replica_threshold)
            if args.replicas and sum(load)
            else []
        )
    extra = max((len(c) for c in chosen.values()), default=0)
    slots_per_rank = per_rank + extra
    # A permutation solved for EP E is also balanced for every EP size dividing E: its ranks are unions of E's.
    ep_sizes = (
        [e for e in range(ranks, 1, -1) if ranks % e == 0 and n % e == 0]
        if not extra
        else [ranks]
    )
    same_sources = info["ep_size"] == ranks

    rows, homes, report = [], [], []
    for layer in range(num_layers):
        by_rank = by_rank_of(layer)
        load = [sum(column) for column in zip(*by_rank)]
        total = sum(load)
        if not total:
            print(
                f"layer {layer:2d}: no routes recorded, keeping contiguous ownership",
                file=sys.stderr,
            )
            row = [
                e
                for r in range(ranks)
                for e in list(range(r * per_rank, (r + 1) * per_rank))
                + [r * per_rank] * extra
            ]
            rows.append(row)
            homes.append(
                [(e // per_rank) * slots_per_rank + e % per_rank for e in range(n)]
            )
            report.append({"layer": layer, "routes": 0})
            continue
        mean = total / ranks
        contiguous = [e // per_rank for e in range(n)]
        before = rank_loads(contiguous, load, ranks)
        permutation = balance(load, ranks)
        perm_loads = rank_loads(permutation, load, ranks)
        replicated = chosen[layer]
        route = {}
        if extra:
            row, home, after, owner, route = place_with_replicas(
                by_rank, ranks, slots_per_rank, replicated, args.replica_balance
            )
        else:
            owner, after = permutation, perm_loads
            # Slots of a rank in ascending logical id.
            row = [e for r in range(ranks) for e in range(n) if permutation[e] == r]
            assert sorted(row) == list(range(n))
            home = [0] * n
            for slot, expert in enumerate(row):
                home[expert] = slot
        assert len(row) == ranks * slots_per_rank and all(
            row[home[e]] == e for e in range(n)
        )
        rows.append(row)
        homes.append(home)
        floor_routes = max(mean, max(load) + sum(sorted(load)[: per_rank - 1]))
        entry = {
            "layer": layer,
            "routes": total,
            "replicated": replicated,
            # [source rank, expert, serving rank]: hits a rank gave to another rank's copy to stay balanced.
            "replica_hits_given_away": [
                [r, e, to]
                for e in replicated
                for r, to in enumerate(route[e])
                if to != r
            ],
            "replicated_share_of_routes": round(
                sum(load[e] for e in replicated) / total, 4
            ),
            "max_over_mean_before": round(max(before) / mean, 4),
            "max_over_mean_permutation": round(max(perm_loads) / mean, 4),
            "max_over_mean_after": round(max(after) / mean, 4),
            # No permutation beats this: the busiest expert shares its rank with per_rank - 1 others.
            "floor": round(floor_routes / mean, 4),
            "busiest_expert": max(range(n), key=lambda e: load[e]),
            "busiest_expert_share_of_routes": round(max(load) / total, 4),
            "busiest_expert_over_mean_rank": round(max(load) / mean, 4),
            "experts_with_routes": sum(1 for c in load if c),
        }
        if same_sources:
            entry.update(
                cross_before=round(cross_fraction(by_rank, contiguous), 4),
                cross_permutation=round(cross_fraction(by_rank, permutation), 4),
                cross_after=round(cross_fraction(by_rank, owner, route), 4),
            )
        for e in ep_sizes[1:]:
            group_of = n // e
            entry[f"max_over_mean_after_ep{e}"] = round(
                max(rank_loads([row.index(x) // group_of for x in range(n)], load, e))
                / (total / e),
                4,
            )
        # One launch: the recorded mean launch, or the given row count with the recorded route shares.
        launches = info["launches_by_layer"].get(layer, 0)
        launch_routes = (
            args.rows_per_launch * top_k
            if args.rows_per_launch
            else (total / launches if launches else None)
        )
        flags = []
        if launch_routes is not None:
            scale = launch_routes / total
            fraction = (
                lambda routes: routes * scale * flop_bits_per_route / quota
            )  # noqa: E731
            # The kernel cuts an expert's rows into 128-row tasks, 64-row ones when no rank holds more than 512 rows.
            task_rows = 64 if launch_routes / top_k / info["ep_size"] <= 512 else 128
            pad = lambda count: (
                -(-round(count * scale) // task_rows) * task_rows if count > 0 else 0
            )  # noqa: E731
            padded = (
                padded_rank_routes(by_rank, owner, route, ranks, pad)
                if same_sources
                else max(
                    sum(pad(load[e]) for e in range(n) if owner[e] == r)
                    for r in range(ranks)
                )
            )
            entry.update(
                rows_per_launch=round(launch_routes / top_k, 1),
                launches_recorded=launches,
                quota_fraction_before=round(fraction(max(before)), 4),
                quota_fraction_permutation=round(fraction(max(perm_loads)), 4),
                quota_fraction_after=round(fraction(max(after)), 4),
                quota_fraction_after_padded=round(
                    padded * flop_bits_per_route / quota, 4
                ),
                quota_fraction_mean_rank=round(fraction(mean), 4),
                quota_fraction_busiest_expert=round(fraction(max(load)), 4),
                quota_fraction_floor=round(fraction(floor_routes), 4),
            )
            if entry["quota_fraction_before"] > args.quota_threshold:
                flags.append("above-quota-before")
            if entry["quota_fraction_after"] > args.quota_threshold:
                flags.append("ABOVE-QUOTA-AFTER")
            if entry["quota_fraction_floor"] > args.quota_threshold:
                flags.append("NO-PERMUTATION-FITS")
        if max(load) > 0.5 * mean:
            # One expert carries more than half a balanced rank's load: a permutation is limited by it.
            flags.append("hot-expert")
        entry["flags"] = flags
        report.append(entry)
    out = {
        "format": PLACEMENT_FORMAT_REPLICAS if extra else PLACEMENT_FORMAT,
        "num_experts": n,
    }
    if extra:
        out["num_physical_experts"] = ranks * slots_per_rank
    out.update(num_layers=num_layers, ep_sizes=ep_sizes, physical_to_logical=rows)
    if extra:
        out["home_slot"] = homes
    out["meta"] = {
        "solved_for_ep_size": ranks,
        "profile_dir": os.path.abspath(args.profile_dir),
        "profile": info,
        "modes": modes or "all",
        "replicas": {
            "slots_added_per_rank": extra,
            "cap": args.replicas,
            "threshold": args.replica_threshold,
            "balance": args.replica_balance,
        },
        "quota": {
            "flop_bits_per_route": flop_bits_per_route,
            "window_quota_flop_bits": quota,
            "sms": args.sms,
            "sm_quota": args.sm_quota,
            "threshold": args.quota_threshold,
            "rows_per_launch_given": args.rows_per_launch,
        },
        "layers": report,
    }
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, args.output)
    solved = [r for r in report if r["routes"]]
    print(
        f"profile: group {info['group']}, ranks {info['ranks']}, rows by mode {info['rows_by_mode']}"
    )
    print(
        f"one launch charges a rank routes x {flop_bits_per_route / args.operand_bits / 1e6:.2f} MFLOP x {args.operand_bits} bits; "
        f"window quota of {args.sms} SMs = {quota:.3e} FLOP x bits (mean rank of a 16384-row launch: {12288 * flop_bits_per_route / quota:.1%})"
    )
    print(
        "layer rows/launch | max/mean before  perm  after | quota before  perm  after (padded) | cross before  perm  after | replicated (share) | expert alone  floor | flags"
    )
    for r in solved:
        if "quota_fraction_before" in r:
            q = f"{r['quota_fraction_before']:11.1%} {r['quota_fraction_permutation']:6.1%} {r['quota_fraction_after']:6.1%} {r['quota_fraction_after_padded']:8.1%}"
            tail = f"{r['quota_fraction_busiest_expert']:11.1%} {r['quota_fraction_floor']:6.1%}"
            rows_launch = f"{r['rows_per_launch']:11.0f}"
        else:
            q, tail, rows_launch = (
                f"{'(pass --rows-per-launch)':>34s}",
                f"{'':>18s}",
                f"{'?':>11s}",
            )
        cross = (
            f"{r['cross_before']:11.1%} {r['cross_permutation']:6.1%} {r['cross_after']:6.1%}"
            if "cross_before" in r
            else f"{'':>25s}"
        )
        replicated = f"{','.join(str(e) for e in r['replicated']) or '-':>10s} ({r['replicated_share_of_routes']:.1%}, {len(r['replica_hits_given_away'])} given away)"
        print(
            f"{r['layer']:5d} {rows_launch} | {r['max_over_mean_before']:15.3f} {r['max_over_mean_permutation']:5.3f} {r['max_over_mean_after']:6.3f} | {q} | {cross} | {replicated} | {tail} | {' '.join(r['flags'])}"
        )
    if solved:
        mean_of = lambda key: sum(r[key] for r in solved) / len(solved)  # noqa: E731
        print(
            f"mean over {len(solved)} layers: max/mean {mean_of('max_over_mean_before'):.3f} -> permutation {mean_of('max_over_mean_permutation'):.3f} -> result {mean_of('max_over_mean_after'):.3f}; "
            f"worst layer {max(r['max_over_mean_before'] for r in solved):.3f} -> {max(r['max_over_mean_permutation'] for r in solved):.3f} -> {max(r['max_over_mean_after'] for r in solved):.3f}"
        )
        if "cross_before" in solved[0]:
            weight = sum(r["routes"] for r in solved)
            weighted = (
                lambda key: sum(r[key] * r["routes"] for r in solved) / weight
            )  # noqa: E731
            print(
                f"cross-GPU routes (all recorded routes): {weighted('cross_before'):.1%} -> permutation {weighted('cross_permutation'):.1%} -> result {weighted('cross_after'):.1%}"
            )
        for flag, text in (
            (
                "above-quota-before",
                f"layers whose busiest rank exceeds {args.quota_threshold:.0%} of one window's quota per launch with contiguous ownership",
            ),
            ("ABOVE-QUOTA-AFTER", "layers still above it in the result"),
            (
                "NO-PERMUTATION-FITS",
                "layers no permutation can bring under it (replicas needed)",
            ),
            (
                "hot-expert",
                "layers where one expert carries more than half a balanced rank's load",
            ),
        ):
            hit = [r["layer"] for r in solved if flag in r["flags"]]
            print(f"{text}: {hit if hit else 'none'}")
    if extra:
        print(
            f"replicas: {extra} slot(s) added per rank ({ranks * slots_per_rank} slots in all); layers by number of replicated experts: "
            + ", ".join(
                f"{h}: {sum(1 for c in chosen.values() if len(c) == h)}"
                for h in range(extra + 1)
            )
        )
    print(
        f"wrote {args.output} (format {out['format'].rsplit('/', 1)[1]}, EP sizes {ep_sizes})"
    )


if __name__ == "__main__":
    main()
