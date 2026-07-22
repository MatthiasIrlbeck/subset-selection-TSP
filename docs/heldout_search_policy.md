# Held-out search-policy presets

Version 0.11 adds two opt-in search controllers selected from a held-out,
matched-worker study. The published 0.10 controller remains the default:

```text
--search-policy legacy-balanced   # default; preserves 0.10 behavior
--search-policy heldout-balanced  # moderate extra compute, lower low/mid-p values
--search-policy heldout-quality   # stronger subset and full-TSP allocation
```

The study baseline is commit
`be000aa16e151a43d8dcfc09d59ac045c848b025`. Candidate schedules were screened
on pilot streams, then evaluated on disjoint point and search streams. The
primary statistic is

\[
\Delta=(L/k)_{\text{candidate}}-(L/k)_{\text{legacy}},
\]

so a negative value favors the candidate. Confidence intervals resample whole
point-set replicate blocks. Compute is accumulated solver worker time rather
than nominal iteration count. The machine-readable decision record, report
hashes, and full numerical summaries are in
`config/heldout_policy_study_summary.json`.

## Controller contract

### `legacy-balanced`

This is the release default. It leaves the established automatic restart,
simulated-annealing, and full-TSP controller unchanged. Fixed-seed jobs that do
not request a new preset retain the 0.10 trajectory.

### `heldout-balanced`

For an ordinary deterministic staged subset solve, the preset applies when:

```text
k >= 40
0.02 <= p <= 0.35
```

It uses:

```text
SA iterations per restart:          20,000
candidate trials per SA iteration:  4
random valid-trial probability:     0.10
temperature endpoints:              configured fixed T0/T1
```

Outside that range, the ordinary literal configuration is used. The full-TSP
controller is unchanged.

### `heldout-quality`

The stronger subset policy uses 30,000 iterations with the same four-candidate
proposal rule. Its validated range is:

```text
open square:  0.02 <= p <= 0.35, k >= 40
periodic:     0.02 <= p <= 0.50, k >= 40
```

At `p = 1`, while the relevant controls remain at release defaults, it screens
16 deterministic starts, promotes four quality-and-edge-diverse starts, and
runs 450 ILS iterations per promoted start.

## Precedence and reproducibility

A preset never silently overrides a manual experiment. Automatic subset
application is disabled when any core SA budget/proposal control differs from
its release default, when automatic temperature calibration is active, when
staged search is disabled, in continuation-only secondary solves, or in
elapsed-time mode. The full-TSP quality section is disabled when candidate
count, promoted-restart count, or ILS depth is explicitly changed.

The requested preset is serialized as `config.search_policy_preset`.
Per-restart `restart_sa_iterations`, temperature fields, and aggregate candidate
evaluation counters expose the effective allocation. Selecting
`legacy-balanced` is the explicit compatibility path.

## Held-out subset results

The tables report equal-cell means across the listed p ranges. Wins, ties, and
losses count paired instance/cell observations.

| Geometry and scale | Policy/range | Blocks | Mean delta | 95% block interval | Wins / ties / losses |
|---|---|---:|---:|---:|---:|
| Periodic, `N=2000` | balanced, `p=0.02..0.15` | 36 | **-0.018957** | **[-0.021653, -0.016180]** | 146 / 3 / 31 |
| Periodic, `N=2000` | balanced, `p=0.20..0.35` | 36 | **-0.009992** | **[-0.012013, -0.007745]** | 117 / 0 / 27 |
| Open, `N=2000` | balanced, `p=0.02..0.35` | 24 | **-0.008387** | **[-0.011294, -0.005349]** | 86 / 1 / 33 |
| Open, `N=2000` | quality, `p=0.02..0.35` | 24 | **-0.013953** | **[-0.016711, -0.011332]** | 102 / 3 / 15 |
| Periodic, `N=2000` | quality, `p=0.40..0.50` | 20 | **-0.011264** | **[-0.013689, -0.009083]** | 53 / 0 / 7 |
| Periodic, `N=5000` | quality, `p=0.02..0.40` | 12 | **-0.011411** | **[-0.014039, -0.009032]** | 67 / 0 / 5 |

The selected boundaries are deliberate:

- Open `p=0.40` reversed the balanced effect: mean delta `+0.004633`, with
  7 wins and 17 losses.
- Periodic quality was essentially neutral at `p=0.60` and harmful at
  `p=0.70`; their combined interval crossed zero and the `p=0.70` mean was
  `+0.002971`.
- Four-candidate 20k search at periodic `p=0.80` was clearly harmful, with mean
  delta `+0.004569` and 2 wins versus 34 losses.
- At `p=0.005` (`k=10` in the boundary study), the multiple-candidate policy was
  essentially neutral. The preset therefore requires `k >= 40` and
  `p >= 0.02`.

## Held-out full-TSP results

The final TSP confirmation compared 16 screened starts with four 450-iteration
promotions against the 12-start, five-promotion, 300-iteration release
controller.

| Geometry and scale | Blocks | Worker ratio | Mean delta in `L/N` | 95% interval | Wins / ties / losses |
|---|---:|---:|---:|---:|---:|
| Periodic, `N=1500` | 18 | 1.030 | **-0.001230** | **[-0.001684, -0.000806]** | 15 / 3 / 0 |
| Open, `N=1000` | 16 | 1.236 | **-0.001351** | **[-0.002167, -0.000543]** | 13 / 1 / 2 |

This controller belongs only to `heldout-quality`; the balanced preset retains
the lower-cost established TSP allocation.

## Temperature decision

Generic restart-local temperature calibration remains rejected. In the held-out
high-p study, a calibrated one-candidate policy worsened every one of 48 paired
cases, and calibrated four-candidate search also had a positive confidence
interval. Both presets therefore retain the fixed geometric temperature
schedule. Their gain comes from comparing several candidate swaps per iteration
and reallocating iteration depth, not from automatic endpoint estimation.

## Interpretation limits

These measurements compare finite-compute heuristics; they are not optimality
proofs. Worker-time ratios depend on hardware, compiler, continuation state, and
the downstream neighborhoods activated by the SA endpoint. A mean improvement
does not guarantee improvement on every point set. New default promotion still
requires disjoint held-out streams, matched-compute accounting, scale and
geometry transfer, and a whole-grid continuation check.
