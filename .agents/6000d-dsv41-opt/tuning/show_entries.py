import json
import sys

for path in sys.argv[1:]:
    try:
        t = json.load(open(path))
    except FileNotFoundError:
        print(path, "missing")
        continue
    print("##", path.split("/")[-1])
    for e in t["entries"]:
        print(
            f" T={e['tokens']} rule={e.get('rule','incumbent')} start={e['start']} steps={[(s.get('knob', s.get('candidate')), s.get('value', s.get('change'))) for s in e['steps']]} trials={e['trials_run']} min={e['minutes']}"
        )
        print("   knobs", json.dumps(e["knobs"]))
        print(
            "   us", e["us"], "default", e["default_us"], "shipped", e.get("shipped_us")
        )
        print(
            "   vs default",
            e["against_default_pct"],
            "mean",
            e["score_pct"],
            "| shipped vs default",
            e.get("shipped_against_default_pct"),
            "| vs shipped",
            e.get("against_shipped_pct"),
        )
        c = e.get("confirm") or {}
        print(
            "   confirm seed",
            c.get("seed"),
            "best",
            c.get("best_us"),
            "default",
            c.get("default_us"),
            "vs default",
            c.get("against_default_pct"),
            "shipped",
            c.get("shipped_us"),
        )
