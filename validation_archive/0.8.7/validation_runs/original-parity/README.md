# Original-vs-current parity run

Status: completed.

The current executable and original upstream executable both built and ran.

Important files:
- parity_manifest.csv
- parity_manifest.json
- tiny-grid-parity-current-original-compatible.json
- tiny-grid-parity-baseline-original.json
- small-hybrid-parity-current-original-compatible.json
- small-hybrid-parity-baseline-original.json
- tiny-controlled-current.json
- tiny-controlled-original.json
- hybrid-controlled-current.json
- hybrid-controlled-original.json

Summary:
- Grid backend vs brute-force backend matched exactly in the harness: max_abs_mean_delta = 0.0.
- Original comparison from parity_manifest.csv:
  - tiny-grid-parity baseline_max_abs_mean_delta ≈ 0.05209208
  - small-hybrid-parity baseline_max_abs_mean_delta ≈ 0.11283812
- Controlled --knn comparison:
  - tiny balanced max_abs_delta ≈ 0.05209208
  - hybrid max_abs_delta ≈ 0.20276506

Interpretation:
These are small deterministic validation scenarios, not publication-scale benchmarks.
They confirm the harness runs against the original executable and show that the cleaned implementation is not bit-for-bit identical to the original, especially in hybrid mode.
