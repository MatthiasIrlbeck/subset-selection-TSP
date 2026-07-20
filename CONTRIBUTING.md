# Contributing

1. Build with tests enabled.
2. Run CTest before submitting changes.
3. Keep public headers under `include/aldous_tsp` namespaced.
4. Add tests for any solver, parser, or JSON schema behavior change.
5. For performance changes, include before/after benchmark commands and result JSON files.

6. Edit `config/options.json`, not generated option surfaces; run `python3 scripts/generate_options.py --check`.
7. Keep fuzz targets bounded and add a seed corpus entry for every fixed fuzz regression.
