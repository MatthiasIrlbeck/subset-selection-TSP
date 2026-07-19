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
  --p-values 0.02,0.03,0.05,0.10,0.20,0.50,1.00 \
  --output results-N1000-seed2024.json
```

For strict reproducibility across machines, prefer `--threads 1`. Each instance uses deterministic stream seeds, so the numerical results should be stable for a given compiler and platform, but thread scheduling changes progress order and may change floating-point aggregation order in future extensions.

The output writer uses an atomic temporary file + rename pattern to avoid half-written JSON files.


## Per-instance output

Pass `--include-instance-rows` when you want each Monte Carlo instance recorded separately. The top-level `instance_rows` array then includes per-instance normalized values, p/k/value rows, effective KNN build metadata, search counters, and per-instance oracle call records. Without the flag, `instance_rows` is present but empty to keep default result files compact.

## Effective KNN/build metadata

Schema version 12 records both requested and effective KNN behavior. This distinguishes a requested grid backend from a safe brute-force fallback on pathological tiny-coordinate inputs, and records forced grid-cell capping through `knn_grid_cell_capped_instances` plus per-instance `knn_build` details. Build metadata also records configured and effective C++ flags, target compile options, source-level low-memory overrides, and the optimization profile.
