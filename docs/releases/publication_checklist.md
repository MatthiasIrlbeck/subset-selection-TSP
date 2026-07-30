# Publishing 2.0.0 without discarding the public v1.0 history

The hardened development history and the current public `main` began as unrelated Git histories.
Do not force-push the hardened branch over `main`.

From a clone that can reach GitHub:

```bash
git remote add hardened /path/to/subset-selection-TSP-2.0.0.git.bundle
git fetch hardened release/v2.0.0-prep
git fetch origin main

git switch -c release/v2.0.0 hardened/release/v2.0.0-prep
# Review the expected public head before joining histories.
test "$(git rev-parse origin/main)" = "9435423964f8cef5222d6802cd3d25e8da9fcace"
git merge --allow-unrelated-histories -s ours origin/main \
  -m "Merge legacy public history before 2.0.0"
```

The merge keeps the hardened source tree while making the three public prototype commits part of
the ancestry. Push `release/v2.0.0`, open a pull request, and run every blocking workflow before
merging. If `origin/main` has changed, inspect the new commits and update the expected head rather
than bypassing the check.

The same guarded merge is automated by `scripts/prepare_public_release_branch.sh`. Run it from the
bundle-cloned `release/v2.0.0-prep` branch while that branch still tracks the bundle remote. The
helper fetches the tracked bundle branch and refuses a dirty tree, detached HEAD, the wrong branch,
any local commit drift from the reviewed bundle tip, an unexpected audited parent, an unexpected
public head, an unreachable historical performance baseline, or an existing destination branch.
It verifies that the merge preserves the exact hardened source tree.

## Mandatory pull-request merge method

Merge the release pull request with GitHub's **Create a merge commit** option. Do not use
**Squash and merge**, **Rebase and merge**, or a linear-history rewrite for this import. The
hardened development ancestry contains validation provenance and the pinned commit in
`config/performance_baseline_commit.txt`; rewriting or discarding that ancestry can make the
historical-performance gate fail in a fresh checkout.

Temporarily keep merge commits enabled and linear-history enforcement disabled for this pull
request. After 2.0.0 is merged, the repository's ordinary merge settings may be reconsidered.

After merge, create a signed annotated tag:

```bash
git switch main
git pull --ff-only
git tag -s v2.0.0 -m "subset-selection-TSP 2.0.0"
git push origin v2.0.0
```

`CITATION.cff` intentionally omits the optional `date-released` field while the
release is still a candidate. The signed tag and GitHub release metadata are the
authoritative publication date; do not commit a guessed date merely to prepare the PR.

The verified-release workflow creates a **draft** and explicitly leaves it non-latest. Review the
attached archives, SBOM, checksum manifest, provenance statement, and GitHub attestations. Then
publish and mark it latest explicitly:

```bash
gh release edit v2.0.0 --draft=false --latest
```

Enable release immutability only after that review. Draft releases cannot be marked latest by the
GitHub Releases API, so these are intentionally separate steps.

## GitHub repository settings

Before the release pull request is merged:

1. Enable private vulnerability reporting and verify that the Security issue-template link opens
   the repository's private advisory form.
2. Enable secret scanning and push protection when the repository settings make them available.
3. Run the release branch once so GitHub registers the new status-check names.
4. Protect `main` with a ruleset that requires a pull request, all blocking CI/CodeQL checks,
   an up-to-date branch, and no force-push or deletion. Do not require linear history until the
   2.0.0 history-import pull request has been merged with a true merge commit.
5. Protect `v*` tags from update and deletion. Require a signed annotated release tag.
6. Configure the independently verified `LKH_3_0_14_SHA256` and
   `CONCORDE_LINUX24_SHA256` repository variables. The release candidate intentionally keeps
   the real-oracle workflow manual-only; restore its schedule only in a reviewed follow-up after
   both variables are present.
7. Review Dependabot's first GitHub Actions and Python dependency pull requests before merging.

The repository files cannot enforce these account-level settings. Record the final ruleset names
and required checks in the release issue or pull request.

## Release-candidate checks

- Confirm the source-archive CTest inventory is completely green, not only the Git checkout.
- Run `python3 scripts/verify_release_bundle.py` against the final `dist/` directory.
- Confirm `validation_runs/current/ARTIFACT_VERSION` and every native current result match 2.0.0
  and schema 16.
- Confirm the current validation receipts match the normalized result bytes.
- Confirm the public API example builds through the installed CMake package.
- Confirm the release notes explain that production results are heuristic upper bounds unless the
  exact small-instance solver is explicitly used.
- Confirm the draft release contains exactly the ZIP, tarball, SPDX SBOM, provenance statement,
  and checksum manifest produced by the verified workflow.
