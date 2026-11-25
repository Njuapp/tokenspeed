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

3. **Use.** `export TOKENSPEED_MOE_EXPERT_PLACEMENT=/path/to/placement.json` in the switch set of every EP rank;
   the engine refuses a file solved for another EP size or expert count.

## Notes

- The prefill switch sets of `../switch-sets` name a file solved this way from the prefill gold standard (EP8, 8
  replicas); it is not in this branch.
- On `mega_moe_split` FlashInfer serves at most 64 expert slots per rank when the EP group spans NUMA nodes (56
  with 8 replicas at EP8), and it picks its tuned kernel configuration by the total number of expert slots, so a
  file with replicas runs FlashInfer's rule-based selection instead. Compare a permutation-only file with a
  replicated one on that backend.
- Decode results with a placement file for both backends will be added to `../launch/README.md`.
