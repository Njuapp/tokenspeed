#!/usr/bin/env python3
"""Anchor tuner for the SM120 W4A8 split MegaMoE that judges every kernel configuration on two route
distributions, ``balanced`` and ``power_law``.

It drives the tuning tools of the split kernels' authors and needs them: their ``tune_table.py`` (given as
``--base``; this script takes the knob space, the presets and the knob-to-flag mapping from it), their
standalone runner ``mega_runner`` and their launcher (``--launcher``, the two-host one at EP16). Nothing of
theirs is changed. A trial is theirs: ``--whole_graph --pr_timing --cold_l2``, 20 warm-up and 100 timed
rounds, a round's time is the maximum over the ranks, a trial's time the median over the rounds.

Why two distributions: a table tuned on balanced routes alone (every expert receives the same number of
rows) is not necessarily the fastest on real traffic, whose routes are skewed. ``power_law`` is the runner's
Zipf routing (exponent 1.0): the expert of popularity rank r receives a share proportional to 1/r, the hot
experts land on random ranks.

``--rule sequential`` (the rule the published table follows):

1. The rule-based default configuration is the first baseline.
2. Every candidate is compared with the CURRENT baseline. If it is more than ``--max_regress`` (0.5%) slower
   on balanced or on power_law, it is excluded. Otherwise the two times are averaged, and if the average is
   lower than the baseline's the candidate becomes the baseline.
3. Candidates, in this order: the package's presets, its extra starting points, the anchors of the table
   already shipped (``--shipped``), then one pass over the package's knob dimensions around the baseline.
4. The final baseline, the default and the shipped anchor are measured again with a second route seed.

A candidate is measured on balanced first; when that already excludes it, no power_law trial is run.

``--pl_repeats N`` (we used 3): one power_law trial can move by several percent between two runs of the same
configuration (measured at 256 rows per rank: 2877 / 3033 / 3062 microseconds; at 96 and 128 rows the
repeats agreed within 0.2%), which is more than the 0.5% the rule works with. With N > 1 a power_law time is
the median of N trials, unless the first trial is more than ``--pl_margin`` slower than the baseline (then it
is excluded on that one trial).

Other rules, kept because results of them are published next to this script:

* ``power_law``: the fastest power_law time alone, balanced ignored (how close can the kernels get to another
  kernel on skewed routes).
* ``incumbent`` and ``default``: two earlier readings of the rule. Start from the admissible starting point
  with the best mean relative change; a change must beat the incumbent's mean by ``--min_gain``; excluded if
  slower than the incumbent or the default (``incumbent``), or than the default only (``default``).

Trials are cached in ``<work_dir>/trials.jsonl`` (an interrupted run resumes from it); ``--more_cache`` reads
the trials of other runs. ``--seed_cache`` reuses balanced trials of an earlier single-distribution run after
checking that the default still reads the same (``--max_drift``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROUTES = ("balanced", "power_law")
INF = float("inf")


def load_base(path: str):
    spec = importlib.util.spec_from_file_location("tune_table_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canon(knobs: dict) -> str:
    return json.dumps(
        dict(sorted((k, v) for k, v in knobs.items() if v is not None)), sort_keys=True
    )


class Tuner2:
    def __init__(self, args, base):
        self.args, self.base = args, base
        self.work = Path(args.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.work / "trials.jsonl"
        self.cache, self.old = {}, {}
        for path in args.seed_cache or []:
            for line in Path(path).read_text().splitlines():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    self.old[rec["key"]] = rec
        for path in list(getattr(args, "more_cache", None) or []) + [self.trials_path]:
            if Path(path).exists():
                for line in Path(path).read_text().splitlines():
                    rec = json.loads(line)
                    if "|recheck" not in rec["key"]:
                        key = rec["key"]
                        for old, new in (("|aa1", "|r2"), ("|aa2", "|r3")):
                            if key.endswith(old):
                                key = key[: -len(old)] + new
                        self.cache[key] = dict(rec, key=key)
        self.h, self.i, self.k, self.e = base.MODELS[args.model]
        self.topology = base.DEFAULT_TOPOLOGY[args.ep]
        self.device = (args.num_sms, 8, 8)
        self.reuse = {}  # anchor -> may the old balanced trials be used
        self.spread = (
            {}
        )  # (anchor, knobs, route, seed) -> the repeated trials behind a median
        # rule "power_law": the objective is the power_law time alone (what the kernel can reach there).
        self.routes = (
            ("power_law",) if getattr(args, "rule", "") == "power_law" else ROUTES
        )
        self.trials = 0

    # ------------------------------------------------------------------ one trial
    def key(
        self, tokens: int, knobs: dict, route: str, seed: int, tag: str = ""
    ) -> str:
        key = f"{self.args.model}|ep{self.args.ep}|t{tokens}|seed{seed}|cold|{canon(knobs)}"
        if self.topology != self.base.DEFAULT_TOPOLOGY[self.args.ep] or self.device != (
            self.base.REFERENCE_SMS,
            8,
            8,
        ):
            key += f"|{self.topology}|sms{self.device[0]}"
        if route != "balanced":
            key += f"|{route}"
        if tag:
            key += f"|{tag}"
        return key

    def measure(
        self, tokens: int, knobs: dict, route: str, seed: int = None, tag: str = ""
    ) -> float:
        seed = self.args.seed if seed is None else seed
        key = self.key(tokens, knobs, route, seed, tag)
        if key in self.cache:
            return self.cache[key]["p50_us"]
        if (
            not tag
            and route == "balanced"
            and self.reuse.get(tokens)
            and key in self.old
        ):
            rec = dict(self.old[key], reused=True)
            self.cache[key] = rec
            with open(self.trials_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"[tune2] T={tokens} {route} reused p50={rec['p50_us']:.1f} {canon(knobs)}",
                flush=True,
            )
            return rec["p50_us"]
        runner = [
            "--num_tokens_per_rank",
            str(tokens),
            "--num_topk",
            str(self.k),
            "--num_total_experts",
            str(self.e),
            "--hidden",
            str(self.h),
            "--intermediate",
            str(self.i),
            "--seed",
            str(seed),
            "--data_parallel_size",
            "1",
            "--tensor_parallel_size",
            "1",
            "--route_distribution",
            route,
            "--enable_static_expert_shape",
            "--split_launch",
            "green_graph",
            "--gate_up_clamp",
            "10.0",
            "--perf_run",
            "--skip_ref_check",
            "--use_cuda_events",
            "--whole_graph",
            "--pr_timing",
            "--perf_warmup",
            str(self.args.warmup),
            "--perf_iters",
            str(self.args.rounds),
            "--generic_decode_preset",
            "off",
            "--cold_l2",
        ] + self.base.knob_flags(knobs)
        log = (
            self.work / "logs" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".log")
        )
        log.parent.mkdir(exist_ok=True)
        cmd = (
            [
                self.args.launcher,
                self.args.gpus,
                str(log),
                str(self.args.trial_timeout),
                self.args.src,
            ]
            + list(self.args.launcher_env or [])
            + ["--"]
            + runner
        )
        p50, status = INF, "failed"
        for _ in range(2):
            t0 = time.time()
            rc = subprocess.call(cmd)
            text = log.read_text(errors="replace") if log.exists() else ""
            found = self.base.P50.findall(text)
            if rc == 0 and found:
                p50, status = float(found[-1]), "ok"
                break
            status = f"rc={rc}"
            if "out of memory" in text or "OutOfMemory" in text or "Traceback" in text:
                break
        self.trials += 1
        rec = dict(
            key=key,
            tokens=tokens,
            route=route,
            seed=seed,
            knobs=knobs,
            p50_us=p50,
            status=status,
            seconds=round(time.time() - t0, 1),
            log=str(log),
            time=time.strftime("%F %T"),
        )
        with open(self.trials_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.cache[key] = rec
        print(
            f"[tune2] T={tokens} {route} {status} p50={p50:.1f} {canon(knobs)}",
            flush=True,
        )
        return p50

    # ------------------------------------------------------------------ the rule
    def admissible(self, us: dict, refs: list) -> bool:
        limit = 1.0 + self.args.max_regress
        return all(
            r in us and us[r] <= ref[r] * limit for r in self.routes for ref in refs
        )

    def score(self, us: dict, ref: dict) -> float:
        return sum(us[r] / ref[r] - 1.0 for r in self.routes) / len(self.routes)

    def both(
        self,
        tokens: int,
        knobs: dict,
        refs: list,
        force: bool = False,
        seed: int = None,
        near: list = None,
    ) -> dict:
        """Times per distribution; without ``force`` the second trial is skipped once the first excludes."""
        limit, us = 1.0 + self.args.max_regress, {}
        repeats = getattr(self.args, "pl_repeats", 1)
        margin = getattr(self.args, "pl_margin", 0.04)
        for route in self.routes:
            us[route] = self.measure(tokens, knobs, route, seed)
            if route == "power_law" and repeats > 1 and us[route] != INF:
                # One power_law trial moves by several percent from run to run (measured: the same
                # configuration 2877 / 3033 / 3062 us at 256 rows), far more than max_regress. Unless the
                # first trial is clearly excluded (more than pl_margin slower than a reference), the time
                # is the median of pl_repeats trials.
                if force or not any(
                    us[route] > ref[route] * (1.0 + margin) for ref in (near or refs)
                ):
                    vals = [us[route]] + [
                        self.measure(tokens, knobs, route, seed, tag=f"r{i}")
                        for i in range(2, repeats + 1)
                    ]
                    vals = sorted(v for v in vals if v != INF)
                    us[route] = (
                        vals[len(vals) // 2]
                        if len(vals) % 2
                        else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
                    )
                    self.spread[
                        (tokens, canon(knobs), route, seed or self.args.seed)
                    ] = vals
            if not force and any(us[route] > ref[route] * limit for ref in refs):
                break
        return us

    @staticmethod
    def pct(us: dict, ref: dict) -> dict:
        return {
            r: round(100.0 * (us[r] / ref[r] - 1.0), 2)
            for r in ROUTES
            if r in us and r in ref and us[r] != INF
        }

    def mine(self, tokens: int) -> str:
        import math

        if not self.shipped:
            return f"shipped{tokens}"
        near = min(self.shipped, key=lambda t: (abs(math.log(t) - math.log(tokens)), t))
        return f"shipped{near}"

    # ------------------------------------------------------------------ one anchor
    def tune_anchor(self, tokens: int) -> dict:
        a, b = self.args, self.base
        starts = b.presets(a.ep, tokens, 4096, False, True, a.num_sms)
        for i, extra in enumerate(a.extra_starts or []):
            starts[f"extra{i}"] = json.loads(extra)
        for t, knobs in self.shipped.items():
            starts[f"shipped{t}"] = dict(knobs)
        # Old balanced trials: usable only if the default still reads the same today.
        self.reuse[tokens] = False
        old_key = self.key(tokens, starts["default"], "balanced", a.seed)
        if old_key in self.old and old_key not in self.cache:
            now = self.measure(tokens, starts["default"], "balanced", tag="recheck")
            drift = now / self.old[old_key]["p50_us"] - 1.0
            self.reuse[tokens] = abs(drift) <= a.max_drift
            print(
                f"[tune2] T={tokens} default today {now:.1f} against {self.old[old_key]['p50_us']:.1f} "
                f"({100 * drift:+.2f}%): old balanced trials {'reused' if self.reuse[tokens] else 'ignored'}",
                flush=True,
            )
            if not self.reuse[tokens]:
                # Today's number is the default's balanced time.
                rec = dict(
                    self.cache[
                        self.key(
                            tokens, starts["default"], "balanced", a.seed, "recheck"
                        )
                    ],
                    key=old_key,
                )
                self.cache[old_key] = rec
                with open(self.trials_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        elif old_key in self.cache:
            self.reuse[tokens] = bool(self.cache[old_key].get("reused"))
        default = self.both(tokens, starts["default"], [], force=True)
        if INF in default.values():
            raise SystemExit(
                f"the default failed at T={tokens}; see {self.trials_path}"
            )
        mine = self.mine(tokens)
        scored = {"default": default}
        for name, knobs in starts.items():
            if name != "default":
                scored[name] = self.both(
                    tokens,
                    knobs,
                    [default] if a.rule != "power_law" else [],
                    force=(name == mine),
                    near=[default],
                )
        eligible = {
            n: us
            for n, us in scored.items()
            if self.admissible(us, [default] if a.rule != "power_law" else [])
            and INF not in us.values()
        }
        start = min(eligible, key=lambda n: self.score(eligible[n], default))
        best, best_us = dict(starts[start]), eligible[start]
        print(
            f"[tune2] T={tokens} start {start} {self.pct(best_us, default)} of {len(eligible)} admissible starts",
            flush=True,
        )
        steps = []
        for name, values in b.dimensions(
            a.ep,
            tokens,
            best.get("backend"),
            4096,
            a.num_sms,
            False,
            [json.loads(x) for x in (a.extra_sms or [])],
        ):
            for value in values:
                if value == best.get(name) or (
                    isinstance(value, tuple) and list(value) == best.get(name)
                ):
                    continue
                cand = dict(best, **{name: value})
                if (
                    cand.get("backend") == "p2p_split"
                    and cand.get("k2_tail_reclaim") == "off"
                    and cand.get("k2_launch_factor") == 2
                ):
                    continue  # known hang at the first launch (the package tuner skips it too)
                # rule "incumbent": not slower than the configuration it replaces nor than the default;
                # rule "default": not slower than the rule-based default only (the baseline).
                refs = (
                    [best_us, default]
                    if a.rule == "incumbent"
                    else [default] if a.rule == "default" else []
                )
                us = self.both(tokens, cand, refs, near=[best_us])
                if not self.admissible(us, refs):
                    continue
                better = (
                    self.score(us, best_us) < -a.min_gain
                    if a.rule != "default"
                    else self.score(us, default)
                    < self.score(best_us, default) - a.min_gain
                )
                if better:
                    steps.append(
                        dict(knob=name, value=value, change=self.pct(us, best_us))
                    )
                    print(
                        f"[tune2] T={tokens} keep {name}={value} {self.pct(us, best_us)}",
                        flush=True,
                    )
                    best, best_us = cand, us
        entry = dict(
            tokens=tokens,
            rule=a.rule,
            knobs=best,
            us=best_us,
            default_us=default,
            start=start,
            steps=steps,
            against_default_pct=self.pct(best_us, default),
            score_pct=round(100 * self.score(best_us, default), 2),
            old_balanced_reused=self.reuse[tokens],
        )
        if mine in scored:
            entry["shipped_us"] = scored[mine]
            entry["shipped_against_default_pct"] = self.pct(scored[mine], default)
            entry["against_shipped_pct"] = self.pct(best_us, scored[mine])
        if a.confirm_seed:
            confirm = dict(
                seed=a.confirm_seed,
                best_us=self.both(tokens, best, [], force=True, seed=a.confirm_seed),
                default_us=self.both(
                    tokens, starts["default"], [], force=True, seed=a.confirm_seed
                ),
            )
            confirm["against_default_pct"] = self.pct(
                confirm["best_us"], confirm["default_us"]
            )
            if mine in starts and canon(starts[mine]) != canon(best):
                confirm["shipped_us"] = self.both(
                    tokens, starts[mine], [], force=True, seed=a.confirm_seed
                )
                confirm["against_shipped_pct"] = self.pct(
                    confirm["best_us"], confirm["shipped_us"]
                )
            entry["confirm"] = confirm
        return entry

    def tune_anchor_sequential(self, tokens: int) -> dict:
        """The sequential rule: the default configuration is the first baseline; every
        new configuration is compared with the CURRENT baseline; more than max_regress slower on balanced or
        on power_law -> excluded; otherwise, if the average of its two times is lower than the baseline's
        average, it becomes the baseline. Candidates in order: the package's presets, its extra starts, the
        anchors of the shipped table, then one pass over the package's knob dimensions around the baseline.
        """
        a, b = self.args, self.base
        starts = b.presets(a.ep, tokens, 4096, False, True, a.num_sms)
        for i, extra in enumerate(a.extra_starts or []):
            starts[f"extra{i}"] = json.loads(extra)
        for t, knobs in self.shipped.items():
            starts[f"shipped{t}"] = dict(knobs)
        self.reuse[tokens] = False
        default = self.both(tokens, starts["default"], [], force=True)
        if INF in default.values():
            raise SystemExit(
                f"the default failed at T={tokens}; see {self.trials_path}"
            )
        mine = self.mine(tokens)
        state = dict(knobs=dict(starts["default"]), us=default, name="default")
        steps, tested = [], {"considered": 0, "excluded": 0, "not_faster": 0}

        def mean(us):
            return sum(us[r] for r in ROUTES) / len(ROUTES)

        def consider(name, knobs, force=False):
            us = self.both(tokens, knobs, [state["us"]], force=force)
            tested["considered"] += 1
            if not self.admissible(us, [state["us"]]):
                tested["excluded"] += 1
                return us
            if mean(us) < mean(state["us"]):
                steps.append(
                    dict(
                        candidate=name,
                        change=self.pct(us, state["us"]),
                        mean_us=[round(mean(state["us"]), 1), round(mean(us), 1)],
                    )
                )
                print(
                    f"[tune2] T={tokens} new baseline {name} {self.pct(us, state['us'])} mean "
                    f"{mean(state['us']):.1f} -> {mean(us):.1f} us",
                    flush=True,
                )
                state.update(knobs=dict(knobs), us=us, name=name)
            else:
                tested["not_faster"] += 1
            return us

        shipped_us = None
        for name, knobs in starts.items():
            if name != "default":
                us = consider(name, knobs, force=(name == mine))
                if name == mine:
                    shipped_us = us
        for name, values in b.dimensions(
            a.ep,
            tokens,
            state["knobs"].get("backend"),
            4096,
            a.num_sms,
            False,
            [json.loads(x) for x in (a.extra_sms or [])],
        ):
            for value in values:
                cur = state["knobs"].get(name)
                if value == cur or (isinstance(value, tuple) and list(value) == cur):
                    continue
                cand = dict(state["knobs"], **{name: value})
                if (
                    cand.get("backend") == "p2p_split"
                    and cand.get("k2_tail_reclaim") == "off"
                    and cand.get("k2_launch_factor") == 2
                ):
                    continue  # known hang at the first launch (the package tuner skips it too)
                consider(f"{name}={value}", cand)
        best, best_us = state["knobs"], state["us"]
        repeats = {
            name: self.spread.get((tokens, canon(k), "power_law", a.seed))
            for name, k in (
                ("default", starts["default"]),
                ("best", best),
                ("shipped", starts.get(mine, {})),
            )
        }
        entry = dict(
            tokens=tokens,
            rule=a.rule,
            knobs=best,
            us=best_us,
            default_us=default,
            start="default",
            power_law_trials=repeats,
            steps=steps,
            tested=tested,
            against_default_pct=self.pct(best_us, default),
            score_pct=round(100 * self.score(best_us, default), 2),
            mean_us=dict(default=round(mean(default), 1), best=round(mean(best_us), 1)),
            old_balanced_reused=False,
        )
        if shipped_us is not None:
            entry["shipped_us"] = shipped_us
            entry["shipped_against_default_pct"] = self.pct(shipped_us, default)
            entry["against_shipped_pct"] = self.pct(best_us, shipped_us)
        if a.confirm_seed:
            confirm = dict(
                seed=a.confirm_seed,
                best_us=self.both(tokens, best, [], force=True, seed=a.confirm_seed),
                default_us=self.both(
                    tokens, starts["default"], [], force=True, seed=a.confirm_seed
                ),
            )
            confirm["against_default_pct"] = self.pct(
                confirm["best_us"], confirm["default_us"]
            )
            if mine in starts and canon(starts[mine]) != canon(best):
                confirm["shipped_us"] = self.both(
                    tokens, starts[mine], [], force=True, seed=a.confirm_seed
                )
                confirm["against_shipped_pct"] = self.pct(
                    confirm["best_us"], confirm["shipped_us"]
                )
            entry["confirm"] = confirm
        return entry

    def run(self):
        a = self.args
        self.shipped = {}
        if a.shipped:
            for table in json.loads(Path(a.shipped).read_text())["tables"]:
                if table["signature"] == list(self.base.MODELS[a.model]):
                    for e in table["entries"]:
                        if "p50_us" in e:
                            self.shipped[e["tokens"]] = e["knobs"]
        out = Path(a.output)
        table = json.loads(out.read_text()) if out.exists() else dict(entries=[])
        done = {e["tokens"] for e in table["entries"]}
        for tokens in [int(x) for x in a.anchors.split(",")]:
            if tokens in done:
                continue
            t0, n0 = time.time(), self.trials
            entry = (
                self.tune_anchor_sequential(tokens)
                if a.rule == "sequential"
                else self.tune_anchor(tokens)
            )
            entry["trials_run"], entry["minutes"] = self.trials - n0, round(
                (time.time() - t0) / 60, 1
            )
            table.update(
                schema=1,
                model=a.model,
                signature=list(self.base.MODELS[a.model]),
                ep=a.ep,
                topology=self.topology,
                device=list(self.device),
                protocol=dict(
                    timing="pr_timing whole_graph",
                    cold_l2=True,
                    warmup=a.warmup,
                    rounds=a.rounds,
                    seed=a.seed,
                    routes=list(ROUTES),
                    power_law_exponent=1.0,
                    max_regress=a.max_regress,
                    min_gain=a.min_gain,
                    rule=(
                        "excluded if slower than the incumbent or the default by more than "
                        "max_regress on either route; score = mean relative change"
                        if a.rule == "incumbent"
                        else (
                            "the default is the first baseline; a candidate more than max_regress "
                            "slower than the current baseline on either route is excluded; "
                            "otherwise it becomes the baseline if the average of its two times "
                            "is lower"
                            if a.rule == "sequential"
                            else "excluded if slower than the rule-based default by more than "
                            "max_regress on either route; score = mean relative change "
                            "against the default"
                        )
                    ),
                ),
            )
            table["entries"].append(entry)
            table["entries"].sort(key=lambda e: e["tokens"])
            out.write_text(json.dumps(table, indent=1))
            print(
                f"[tune2] anchor T={tokens} done: {json.dumps(entry['knobs'])} against default "
                f"{entry['against_default_pct']} mean {entry['score_pct']}% against shipped "
                f"{entry.get('against_shipped_pct')} ({entry['trials_run']} trials, {entry['minutes']} min)",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the package's tune_table.py")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ep", type=int, default=16)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--num_sms", type=int, default=156)
    ap.add_argument("--extra_sms", nargs="*", default=None)
    ap.add_argument("--extra_starts", nargs="*", default=None)
    ap.add_argument(
        "--shipped", default=None, help="TokenSpeed's mega_moe_sm120_split_tuned.json"
    )
    ap.add_argument(
        "--seed_cache",
        nargs="*",
        default=None,
        help="trials.jsonl of earlier balanced-only runs",
    )
    ap.add_argument("--launcher_env", nargs="*", default=None)
    ap.add_argument(
        "--rule",
        choices=("incumbent", "default", "power_law", "sequential"),
        default="incumbent",
        help="what a candidate must not be slower than: the configuration it replaces (and the "
        "default), or the rule-based default only",
    )
    ap.add_argument(
        "--pl_repeats",
        type=int,
        default=1,
        help="power_law trials behind one time (median); 1 = single trial",
    )
    ap.add_argument(
        "--pl_margin",
        type=float,
        default=0.04,
        help="a first power_law trial more than this slower than a reference is excluded "
        "without repeats",
    )
    ap.add_argument(
        "--more_cache",
        nargs="*",
        default=None,
        help="trials.jsonl of other runs of today, both routes, used as they are",
    )
    ap.add_argument("--anchors", default="128,256,64,32,16")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=9341)
    ap.add_argument("--confirm_seed", type=int, default=6197)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--min_gain", type=float, default=0.005)
    ap.add_argument("--max_regress", type=float, default=0.005)
    ap.add_argument("--max_drift", type=float, default=0.003)
    ap.add_argument("--trial_timeout", type=int, default=900)
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    Tuner2(args, load_base(args.base)).run()


if __name__ == "__main__":
    main()
