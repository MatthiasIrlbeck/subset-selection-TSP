# Development provenance and responsibility

Version 2.0.0 was developed with extensive assistance from OpenAI language-model tooling under
the direction of the repository maintainer. The historical commit identities `OpenAI` and
`OpenAI Repair` denote automated implementation, audit, and repair assistance. They do not imply
that OpenAI sponsors, endorses, maintains, or accepts responsibility for this repository.

The maintainer selected the research direction, supplied the problem context and acceptance
criteria, reviewed the resulting changes, and is responsible for the released code, documentation,
benchmarks, and scientific claims. Automated assistance does not replace independent review or the
need to reproduce numerical results before relying on them.

The development history is intentionally preserved rather than rewritten for attribution. This
keeps validation provenance, performance-baseline commits, release receipts, and source-revision
records auditable. The repository's tests, sanitizer runs, fuzzing, held-out policy studies, and
release artifacts document what was checked; they are not a claim that the software is free of all
defects.

Contributions should identify substantial automated assistance when it materially shaped a patch,
especially for solver logic, statistical analysis, security boundaries, or performance claims. The
human contributor submitting or approving a change remains responsible for its correctness and
interpretation.
