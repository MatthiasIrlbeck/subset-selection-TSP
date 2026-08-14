# ADR 0001: Exact coordinate distance backend

## Decision

Store only coordinates and compute distances on demand.

## Rationale

The full distance matrix is `O(N^2)` memory. Coordinate storage is `O(N)` and preserves exact Euclidean distances for arbitrary pairs. The current KNN builder uses brute force for correctness and simple testing; future backends can replace KNN construction without changing solver APIs.
