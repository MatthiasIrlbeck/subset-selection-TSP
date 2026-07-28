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

After merge, create a signed annotated tag:

```bash
git switch main
git pull --ff-only
git tag -s v2.0.0 -m "subset-selection-TSP 2.0.0"
git push origin v2.0.0
```

Create the GitHub release as a draft first, verify all checksums/attestations, mark it as latest,
and only then publish or enable release immutability.
