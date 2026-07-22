# Reproducibility

The result JSON records:

- schema version,
- project version,
- git commit embedded at configure time,
- compiler string,
- full run configuration,
- p-grid,
- wall time,
- search statistics,
- per-p raw values and summary statistics.

Use explicit seeds and p-grids for reproducible experiments:

```bash
./build/aldous_tsp \
  --N 1000 \
  --instances 20 \
  --threads 1 \
  --seed 2024 \
  --campaign-id aldous-main \
  --campaign-shard 0 \
  --replicate-offset 0 \
  --point-seed 2024 \
  --search-seed 2024 \
  --search-policy legacy-balanced \
  --solver-policy-id publication \
  --fidelity-level strong \
  --p-values 0.02,0.03,0.05,0.10,0.20,0.50,1.00 \
  --include-instance-rows \
  --output results-N1000-seed2024.json
```

For strict reproducibility across machines, prefer `--threads 1`. Each instance uses deterministic stream seeds, so the numerical results should be stable for a given compiler and platform, but thread scheduling changes progress order and may change floating-point aggregation order in future extensions.

The output writer uses a unique same-directory temporary file and an atomic commit operation to avoid half-written JSON files. Without `--force`, the final commit is an atomic create-if-absent operation, so concurrent processes cannot both win a no-overwrite race. With `--force`, replacement remains atomic. `--output-durability none|file|full` distinguishes a visible commit, a file-data-synchronized commit, and a file-plus-parent-directory-synchronized commit. The public writer reports whether a failure occurred before commit or after the target became visible, preventing unsafe blind retries.

JSON numbers are formatted with locale-independent `to_chars` semantics. Valid UTF-8 strings are preserved; malformed byte sequences are replaced with the JSON `\uFFFD` replacement character rather than emitting invalid JSON text.


## Per-instance output

Pass `--include-instance-rows` when you want each Monte Carlo instance recorded separately. The top-level `instance_rows` array then includes per-instance normalized values, p/k/value rows, effective KNN build metadata, search counters, per-instance oracle call records, a stable numeric `replicate_id`, and 64-bit point/search stream fingerprints. Without the flag, `instance_rows` is present but empty to keep default result files compact.

## Campaign and RNG identities

`--point-seed` and `--search-seed` separate point-instance randomness from heuristic-search randomness. Both default to `--seed`, so existing commands retain their historical streams. `--replicate-offset` assigns stable replicate IDs to a shard; shard 0 with 24 instances normally uses offset 0, shard 1 uses offset 24, and so on. The local row index remains file-local, while `replicate_id` is campaign-global.

Use the same `campaign_id`, `replicate_id`, and point seed across every `(p,k)` cell that should share common random numbers. Use a different search seed to repeat the heuristic on exactly the same point sets. `solver_policy_id` distinguishes algorithm/budget policies, and `fidelity_level` pairs cheap and strong runs for multifidelity correction. `scripts/run_torus_campaign.py` and `scripts/run_full_study.py` now pass these fields and enable instance rows automatically for campaign batches.

`search_policy_preset` is part of the serialized solver configuration. Record it explicitly in publication commands: `legacy-balanced` is the 0.10 compatibility controller, while held-out presets are versioned automatic policies whose effective per-p allocation is also visible through restart iteration and SA candidate-evaluation telemetry. A free-form `solver_policy_id` should still name the complete campaign policy, including any manual overrides.

`scripts/analyze_campaign.py` resamples complete replicate vectors when all selected files carry identities. It averages repeated search streams within a point set before the main fit, reports point-versus-search variance when repeats exist, and computes a paired cheap-plus-correction estimate when cheap and strong fidelities overlap. It also compares `1/k`, `1/k + 1/k^2`, and `1/sqrt(k)` finite-size laws inside the same bootstrap, reports a combined model/statistical envelope, and runs leave-one-size, leave-one-probability, and nested-`pmax` sensitivity refits. Legacy summary-only files remain readable, but the script explicitly falls back to independent-cell bootstrap because their cross-cell correlation cannot be reconstructed. See [`campaign_analysis.md`](campaign_analysis.md) for interpretation and command-line controls.

## Effective KNN/build metadata

Schema version 14 records both requested and effective KNN behavior. This distinguishes a requested grid backend from a safe brute-force fallback on pathological tiny-coordinate inputs, and records forced grid-cell capping through `knn_grid_cell_capped_instances` plus per-instance `knn_build` details. Build metadata also records configured and effective C++ flags, target compile options, source-level low-memory overrides, and the optimization profile.
