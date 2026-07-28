# Verifiable release and external-input policy

Native builds record the source commit, Git tree, dirty state, and how that identity was obtained. A Git checkout uses Git directly. `git archive` substitutes the same commit and tree into `SOURCE_REVISION`, so an extracted source release never degrades to `git_commit = "unknown"` merely because `.git` is absent.

## Immutable automation inputs

Every reusable GitHub Action is referenced by a reviewed 40-character commit SHA. `tests/release_security_test.py` rejects mutable tags and unreviewed actions.

The real-oracle workflow keeps the official LKH and Concorde URLs in `config/oracle_sources.json`, imposes maximum download sizes, and requires independently approved SHA-256 values through repository variables:

```text
LKH_3_0_14_SHA256
CONCORDE_LINUX24_SHA256
```

The release candidate intentionally keeps this workflow manual-only until both repository variables have been configured and reviewed. A manual dispatch may supply the same digests explicitly. Missing or malformed digests fail the workflow before download; mismatches fail before extraction or execution. Re-enable a schedule only in a reviewed follow-up change after the repository variables are present. The solver hashes an oracle when resolving it and re-hashes the exact executable immediately before every launch. A changed executable is refused. Captured console output and tour files have fixed read-size limits.

## Release artifacts

A signed annotated tag `vX.Y.Z` whose value matches CMake is required for publication. The workflow asks the GitHub Git-data API to verify the tag signature before building the release. It then builds and tests from that tag, creates deterministic ZIP and `tar.gz` source archives, verifies archive revision fallback, and publishes:

- source ZIP and `tar.gz`;
- SHA-256 manifest;
- SPDX 2.3 JSON SBOM;
- in-toto Statement with SLSA provenance v1 predicate;
- GitHub build-provenance attestations signed through GitHub's OIDC-backed attestation service.

The workflow refuses a lightweight, unsigned, or GitHub-unverified tag. The generated SBOM is computed from the extracted release archive—not the checkout—and enumerates the archived source files with SHA-1 and SHA-256 digests plus the SPDX package-verification code. The provenance statement binds every release subject to the source commit and Git tree.

## Local reproduction

```bash
export SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)
version=$(sed -n 's/^project(aldous_tsp VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)
commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')
mkdir -p dist
git archive --format=zip --prefix="subset-selection-TSP-${version}/" \
  -o "dist/subset-selection-TSP-${version}.zip" HEAD
python3 scripts/generate_sbom.py --root . \
  --output "dist/subset-selection-TSP-${version}.spdx.json" \
  --version "$version" --commit "$commit" --tree "$tree"
```

Cryptographic release attestations require the GitHub release workflow because the signing identity is the workflow's OIDC identity.
