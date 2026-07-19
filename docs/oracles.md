# External oracle integration

The clean refactor now includes optional external post-processing through LKH or Concorde. Oracle execution is disabled by default so ordinary runs remain self-contained and reproducible.

## Modes

```bash
--oracle none       # default, no external process
--oracle auto       # resolve LKH first, then Concorde, from PATH
--oracle lkh        # require LKH
--oracle concorde   # require Concorde
```

Useful options:

```bash
--oracle-format matrix|euc2d
--lkh-path LKH
--concorde-path concorde
--oracle-time-limit 30
--oracle-scale 1000000
--oracle-min-k 17
--oracle-max-k 2500
--oracle-lkh-runs 6
--oracle-lkh-trials 0
--oracle-no-tsp
--oracle-no-subset
--oracle-inline-feedback
--oracle-verbose
```

`matrix` writes an explicit TSPLIB full matrix, preserving the exact Euclidean distances after integer scaling. `euc2d` writes scaled coordinates and lets the external solver apply its EUC_2D rounding convention. For scientific comparisons, keep the format and scale fixed and record them in the output JSON.

## Execution model

The solver creates a temporary working directory under `TMPDIR` when set, otherwise the platform temporary directory. It writes `problem.tsp`; for LKH it also writes `init.tour` and `run.par`. It launches the external process with an optional timeout, parses the returned tour, maps it back to original node IDs, and applies it only if it improves the internal candidate.

Oracle metadata is recorded in JSON:

- `oracle_status`
- `config.oracle_mode`
- `config.oracle_resolved`
- `config.oracle_exec_path`
- `config.oracle_version`
- `config.oracle_format`
- `config.oracle_time_limit_sec`
- `config.oracle_scale`
- `search_stats.oracle_*`

## Testing

The core unit test suite includes a fake LKH executable that:

1. responds to `--version`,
2. emits a valid TSPLIB tour,
3. improves a deliberately bad circle tour,
4. verifies oracle call, solve, and improvement statistics.

The CTest CLI suite also includes a fake-oracle smoke test. These tests exercise the process-launching path without requiring a real LKH/Concorde installation.

## Caveats

External solver output is trusted only after permutation validation. Duplicate nodes, out-of-range nodes, and missing output files are rejected. The oracle currently reports aggregate counts and total gain; detailed per-call failure reasons are not yet stored in the JSON document.
