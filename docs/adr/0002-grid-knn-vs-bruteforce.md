# ADR 0002: Exact KNN backends

## Decision

Keep both exact KNN backends:

- `grid-exact` as the default production backend,
- `bruteforce` as the validation/debug backend.

## Rationale

The uniform-grid expanding-ring backend restores the original prototype's scalable exact KNN construction strategy for uniformly random Euclidean instances. The brute-force backend remains valuable because it is simple, deterministic, and useful for regression tests and pathological debugging.

The CLI exposes the choice through `--knn-backend grid|bruteforce`. Sampled exact verification is available through `--verify-knn <checks>`.
