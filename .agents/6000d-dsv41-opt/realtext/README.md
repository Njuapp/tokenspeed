# Real-text decode measurement

The workload behind the real-text numbers of `../launch/README.md` and `../docs/EP16_SPLIT_RESULTS.md`:
closed waves of requests with 8192-token prompts of real text and 1024 output tokens, greedy, with the real
DSpark acceptance, through the HTTP gateway of `ts serve`. Only the decode window of a wave is measured.

| File | What it is |
| --- | --- |
| `make_prompts.py` | Builds a bank of 160 distinct 8192-token prompts from the public ShareGPT V3 dump (`--seed 42`; needs `transformers` and the model's tokenizer) |
| `bench_closed_wave.py` | The client. One wave = CC requests released together; decode window = from the moment every request has its first token until the first request finishes; throughput = output tokens of all requests inside the window / window; TPOT = CC x 1000 / throughput; accept = tokens per decode cycle. A smoke request, a 32-token warm-up wave and one discarded wave precede the measured waves |
| `loads_sampler.py` | Samples the gateway's per-rank load once a second while a wave runs |
| `check_spread.py` | Says per wave whether the gateway spread the requests evenly over the ranks, and prints `EVEN_RESULT`: the median of the first three valid waves that were spread evenly |
| `show.py` | Prints a run directory: per CC level the result of all waves, the `EVEN_RESULT` and every wave |

`make_prompts.py` and `bench_closed_wave.py` are the files of the reproduction package of the MegaMoE kernels'
authors (reformatted by this repository's code-style hooks, no functional change); the rest is ours.

## Steps

1. Prompt bank, once: download `ShareGPT_V3_unfiltered_cleaned_split.json` and run
   `python make_prompts.py --model $OPT_MODEL --sharegpt sharegpt.json --out prompts_8k.json`.
   For a placement (step 4) build a second bank with `--seed 43 --out prompts_8k_s43.json`: it shares no prompt
   with the first one.
2. Engines: `../launch/realtext-engine.sh` on every node (node 1 first, then node 0), then wait until
   `http://127.0.0.1:30000/readiness` on node 0 returns 200. `/health` answers before the engine is loaded.
3. For every concurrency level, on node 0:

   ```bash
   python3 loads_sampler.py http://127.0.0.1:30000 $OUT/cc$CC/loads.jsonl 86400 &
   python3 bench_closed_wave.py --url http://127.0.0.1:30000 --model deepseek-v4.1-flash \
       --bank prompts_8k.json --cc $CC --dp 16 --waves 5 --out $OUT/cc$CC/bench > $OUT/cc$CC/bench_summary.txt
   kill %1
   python3 check_spread.py $OUT/cc$CC/bench $OUT/cc$CC/loads.jsonl $CC >> $OUT/cc$CC/bench_summary.txt
   ```

   `--dp` is the number of attention-DP ranks (16 on two nodes, 8 on one). Use concurrency levels that are
   multiples of it. The gateway places the requests itself and sometimes puts one more on one rank and one
   fewer on another; such a wave runs a different row count on that rank, which is why the reported number is
   the `EVEN_RESULT`.
4. Static expert placement (`../placement/README.md`): serve the held-out bank once with the load profile
   switched on (we used one run of 3 waves at CC 160), solve with `--ep-size 16 --modes decode
   --allow-graph-replay-profile`, and start the measured engines with `TOKENSPEED_MOE_EXPERT_PLACEMENT`.
   The profile and the measurement must not share prompts.

Our two-node runs took one engine start for all concurrency levels; an engine on the persistent kernel was
ready after 5 to 11 minutes, one on the split kernels after 8 to 12.
