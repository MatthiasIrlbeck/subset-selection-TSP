# Search-policy tuning

Search-policy defaults should be selected by paired quality at a fixed compute
budget, not by nominal restart or SA-iteration counts. Different policies spend
very different fractions of their budget in seed construction, SA, LNS,
ejection chains, and path relinking.

`scripts/tune_search_policy.py` runs a policy matrix on identical point and
search streams. It compares policies using accumulated solver worker seconds and
block-bootstraps paired per-instance differences by point-stream identity.
Alternating policy order between repetitions limits systematic timing-order
bias.

```bash
python3 scripts/tune_search_policy.py \
  --exe build/aldous_tsp \
  --policies config/search_policy_candidates.json \
  --N 2000 \
  --p-values 0.02,0.4,0.8 \
  --instances 20 \
  --repetitions 3 \
  --threads 4 \
  --restart-threads 2 \
  --sa-iters 5000 \
  --out-dir policy-tuning
```

The report contains:

- observed wall and accumulated worker seconds for every run;
- paired mean `L/k` differences and point-block bootstrap intervals;
- per-cell wins, ties, and losses;
- policies within the requested worker-budget tolerance;
- the quality/worker Pareto frontier;
- the best policy within the declared budget.

The default target is the reference policy's worker time with a 15% tolerance.
Use `--budget-ratio` and `--budget-tolerance` to study a different compute tier.
Every raw solver output and exact command is retained next to
`policy_tuning_report.json`.

## Avoiding tuning bias

Use one set of point streams to choose a policy and a disjoint held-out set to
report its gain. A policy should not become a release default solely because it
won the same finite sample used to search the policy space. Tune separate
regimes for very small, moderate, and high `p`, full TSP, geometry, and scale
when the measured frontier shows material interactions.

The candidate file is ordinary JSON. Each policy supplies an identifier and a
list of CLI arguments; no shell parsing is used. The checked-in candidate set is
a starting design, not a claim that one candidate is universally optimal.
