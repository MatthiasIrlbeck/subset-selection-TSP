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

On POSIX systems, oracle processes are launched with `posix_spawn` in a dedicated process group. When the platform provides `posix_spawn_file_actions_addchdir_np`, the solver is spawned directly with a working-directory file action; otherwise a constant `/bin/sh` command changes directory and immediately `exec`s the exact resolved solver path, with all variable values passed only as positional arguments. The parent prepares argument storage and redirections before spawning, so project C++ code never runs in a post-`fork` child. Standard input is `/dev/null`, descriptors and unreaped children have RAII cleanup, and timeouts terminate and reap the entire process group. Version capture uses nonblocking `poll` and one absolute deadline for both output collection and process reaping, avoiding busy-spins and duplicate timeout windows.

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
- `oracle_call_records[]` with one status/error/timing/gain record per attempted call

## Testing

The core unit test suite includes a fake LKH executable that:

1. responds to `--version`,
2. emits a valid TSPLIB tour,
3. improves a deliberately bad circle tour,
4. verifies oracle call, solve, and improvement statistics.

The CTest CLI suite also includes a fake-oracle smoke test. These tests exercise the process-launching path without requiring a real LKH/Concorde installation.

## Caveats

External solver output is trusted only after permutation validation. Duplicate nodes, out-of-range nodes, and missing output files are rejected. The JSON document records both aggregate oracle counters and detailed `oracle_call_records`, including solver status, executable path, elapsed time, gain, and per-call error text.
