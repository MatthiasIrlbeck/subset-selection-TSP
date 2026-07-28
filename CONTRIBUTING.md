# Contributing

Contributions are welcome, especially reproducible correctness fixes, performance work,
analysis improvements, and documentation clarifications.

## Development setup

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DALDOUS_TSP_ENABLE_WARNINGS=ON \
  -DALDOUS_TSP_ENABLE_WERROR=ON \
  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install Python development tools with:

```bash
python3 -m pip install -r requirements-dev.txt
```

## Source-of-truth rules

- Edit `config/options.json`, not generated option surfaces. Run
  `python3 scripts/generate_options.py` and then `--check`.
- Keep public headers in `include/aldous_tsp/` and namespaced under `aldous_tsp::`.
- Preserve deterministic tie-breaking and fixed-seed behavior unless the change explicitly
  revises that contract.
- Do not weaken input validation, solve postconditions, manifest matching, or atomic-output
  semantics to gain speed.
- Keep current validation evidence under `validation_runs/current/`; move historical evidence
  to `validation_archive/<version>/` and label it clearly.

## Pull requests

A pull request should contain:

1. A precise statement of the problem and the chosen design.
2. Tests for solver, parser, schema, I/O, or analysis behavior that changed.
3. Before/after commands and machine-readable evidence for performance or quality changes.
4. A note on determinism, schema/API compatibility, and scientific interpretation.
5. Documentation and generated surfaces updated in the same change.

Performance claims must compare paired fixed inputs and report both quality and worker/wall
cost. Search-policy tuning must use separate tuning and held-out point streams. A faster
heuristic that silently worsens `L/k` is a regression unless the tradeoff is explicit.

## Additional checks

```bash
python3 -m ruff check scripts tests
python3 scripts/generate_options.py --check
python3 tests/release_security_test.py .
git diff --check
```

For sanitizer and fuzzing instructions, see `docs/fuzzing.md`. For release and provenance
requirements, see `docs/release_security.md`; maintainers should also follow
`docs/releases/publication_checklist.md`.
