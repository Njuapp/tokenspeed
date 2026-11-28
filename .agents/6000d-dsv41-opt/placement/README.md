# Static expert placement

A placement file moves each MoE layer's experts between the EP ranks and can keep copies of the hottest experts
on every rank, so that the ranks receive about the same number of routes and fewer routes cross GPUs. The routes,
the per-route results and the order in which a token's routes are summed do not change. It is a property of the
workload: a file solved for one traffic mix can be useless for another, so solve one for the traffic you serve.

Both MegaMoE backends (`--moe-backend mega_moe` and `mega_moe_split`) take a placement file.

## Steps

1. **Record the load.** Start the engine as usual with two more variables (they can go into the switch set):

   ```bash
   export TOKENSPEED_MOE_LOAD_PROFILE=/path/to/profile-dir
   export TOKENSPEED_MOE_LOAD_PROFILE_INTERVAL=50
   ```

   Every EP rank counts, per layer, the routes its own rows send to each expert and writes
   `profile-dir/rank<r>.json`. The file is written at eager forwards (every 50 of them, or after 60 s). Forwards
   replayed from CUDA graphs, which is how decode steps run, are counted on the GPU and reach the file at the next
   eager forward (the next prefill chunk), so serve the workload long enough and let a prefill follow the decode
   steps you want counted. Recording costs time: do not benchmark with it on.

   A decode profile counts a rank's real rows only. Under attention DP every rank replays the decode graph of
   the busiest rank's batch, so a rank with fewer requests pads with filler rows, and an idle rank replays
   nothing else; all of them route to the same few experts of a layer. Until the fix of 2026-11-27 the profile
   counted them (30% of the decode rows of our two-node real-text profile), and a placement solved from such a
   profile hardly balances the real traffic: `../docs/LOAD_PROFILE_FILLER_ROWS.md`. Check a profile before
   solving from it: `python3 check_live_rows.py profile-dir engine.log [engine-node1.log]` compares the counted
   decode cycles with those of the requests in the engine logs (it must read about 1.00), and the profile's
   `coverage.live_rows_only` says which count it holds.

2. **Solve.** On any machine, standard library only:

   ```bash
   # prefill engine, EP8, up to 8 replicated experts per layer on every rank
   python3 solve_placement.py profile-dir --ep-size 8 --replicas 8 --replica-threshold 0.25 \
       --output placement-ep8-r8.json
   # decode engine: count the decode forwards only
   python3 solve_placement.py profile-dir --ep-size 8 --modes decode --allow-graph-replay-profile \
       --replicas 4 --output placement-decode-ep8-r4.json
   ```

   Without `--replicas` the file is a permutation per layer (format 1). With `--replicas H` every rank gets `H`
   more expert slots and the hottest experts of a layer are copied to every rank (format 2); each replica slot
   costs the weights of one expert per layer on every GPU (about 0.75 GB per replica for the 40 layers of
   DeepSeek-V4.1-Flash). The solver prints a per-layer report: the busiest rank's load over the mean before and
   after, and the share of routes that cross GPUs. The profile must come from the EP size the file is solved for.

   `--replicas H` is a cap. A layer copies only the experts whose load exceeds `--replica-threshold` of the mean
   rank load, and the file gets as many replica slots as the layer with the most of them needs: on our
   real-text traffic `--replicas 8` gives 7 slots per rank at EP16 (496 expert slots) and 3 at EP8 (408), where
   the mean rank load is twice as high; `--replicas 4` gives 4 at EP16 (448) and again 3 at EP8. The last lines
   of the report and the engine log line of `TOKENSPEED_MOE_EXPERT_PLACEMENT` state the slot count.
   `python3 eval_placement.py profile-dir EP none file.json ..` prints what files give on a profile, for
   example on one they were not solved from.

3. **Use.** `export TOKENSPEED_MOE_EXPERT_PLACEMENT=/path/to/placement.json` in the switch set of every EP rank;
   the engine refuses a file solved for another EP size or expert count.

## Notes

- The prefill switch sets of `../switch-sets` name a file solved this way from the prefill gold standard (EP8, 8
  replicas); it is not in this branch.
- On `mega_moe_split` FlashInfer serves at most 64 expert slots per rank when the EP group spans NUMA nodes (56
  with 8 replicas at EP8), and it picks its tuned kernel configuration by the total number of expert slots: a
  file with replicas runs FlashInfer's rule-based selection unless a table is registered for exactly its slot
  count. This branch registers tables for 448 slots at EP16 and for 408, 416 and 448 slots at EP8
  ("Tuned kernel tables" in `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.md`); the engine log
  names the tables. With another slot count, add a table or compare with a permutation-only file.
- Decode results with placement files for both backends: `../docs/EP16_SPLIT_RESULTS.md` (two nodes) and
  `../docs/EP8_VS_EP16.md` (one node against two).
- `check_balance.py profile-dir` says whether the routes of a profile are even over the experts; it confirms
  an engine started with `TOKENSPEED_MOE_ROUTING_SIMULATION=balanced` (`../docs/EP8_VS_EP16.md`).
