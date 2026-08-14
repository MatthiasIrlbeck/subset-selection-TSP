# Hosted CI contract

Version 2.0.0 treats hosted cross-platform validation as part of the release contract, not
as an advisory dashboard. A release candidate is not ready to merge while any blocking job
is failing.

The blocking matrix covers:

- GCC and Clang release builds with warnings treated as errors;
- AppleClang on macOS and MSVC on Windows, including the complete registered test suite;
- AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer paths;
- Ruff correctness checks and clang-tidy on handwritten C++ sources;
- CodeQL's `security-extended` suite for C/C++ and Python; and
- the pinned historical hot-path performance gate.

Platform-specific contracts exposed by this matrix are kept explicit in the source:

- macOS uses the current `posix_spawn_file_actions_addchdir` interface when available and
  retains a checked fallback for older systems;
- Windows-safe test and output cleanup closes all streams before deleting or replacing
  their files, and atomic replacement retries only transient sharing/access conflicts;
- generated validation JSON, receipt, and CSV files are normalized to LF so their recorded
  digests are independent of checkout platform;
- parity tools invoke Python reference programs through the active interpreter rather than
  relying on POSIX executable semantics; and
- oracle workspaces and user-selected input files are canonicalized, bounded, and checked
  before filesystem access or external-process launch.

Broad CodeQL quality findings are useful review input, but the blocking security workflow
uses `security-extended`; Ruff and clang-tidy remain the dedicated blocking style and
correctness gates. Narrow suppressions are permitted only after the corresponding path or
input contract has been validated in code.
