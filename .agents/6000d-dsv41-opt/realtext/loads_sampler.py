"""Sample the gateway's per-DP-rank load once a second (to check how a wave spreads over the attention-DP ranks)."""

import json
import sys
import time
import urllib.request

url, out, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
t_end = time.time() + seconds
with open(out, "w") as f:
    while time.time() < t_end:
        try:
            with urllib.request.urlopen(url + "/get_loads", timeout=5) as r:
                d = json.loads(r.read())
            f.write(
                json.dumps(
                    {
                        "t": time.time(),
                        "running": [x["num_running_reqs"] for x in d["loads"]],
                        "waiting": [x["num_waiting_reqs"] for x in d["loads"]],
                    }
                )
                + "\n"
            )
            f.flush()
        except Exception as e:
            f.write(json.dumps({"t": time.time(), "error": repr(e)[:100]}) + "\n")
        time.sleep(1.0)
