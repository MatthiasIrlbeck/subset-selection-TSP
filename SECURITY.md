# Security policy

## Supported versions

| Version | Security support |
| --- | --- |
| 2.x | Supported |
| 1.x and earlier | Unsupported prototype releases |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's **private
vulnerability reporting** for this repository from the Security tab. Include:

- the affected version or commit;
- the operating system and compiler;
- a minimal reproducer or malformed input, when safe to share;
- the expected and observed behavior;
- whether external LKH/Concorde execution is involved.

The maintainer will acknowledge a complete report as soon as practical, assess severity and
scope, and coordinate disclosure after a fix is available. Research-result disagreements,
ordinary crashes on trusted inputs, and performance regressions should use the public issue
templates unless they expose a security boundary.

## Threat model

The CLI validates its own input and output contracts, but it is not a sandbox. Do not run
untrusted external solver binaries, arbitrary campaign scripts, or untrusted files with
elevated privileges. Oracle downloads used by project workflows are size-limited and
SHA-256 verified; local users remain responsible for independently verifying third-party
executables.
