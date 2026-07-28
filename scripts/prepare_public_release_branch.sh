#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_public_release_branch.sh [options]

Join the hardened development history with the existing public main branch without
changing the hardened source tree.

Options:
  --public-remote NAME       remote containing public main (default: origin)
  --public-branch NAME       public branch (default: main)
  --release-branch NAME      branch to create (default: release/v2.0.0)
  --expected-public-head SHA fail if public main differs from this SHA
  --force-delete-existing    delete an existing local release branch first
EOF
}

public_remote=origin
public_branch=main
release_branch=release/v2.0.0
expected_public_head=9435423964f8cef5222d6802cd3d25e8da9fcace
force_delete=false

while (($#)); do
  case "$1" in
    --public-remote) public_remote=$2; shift 2 ;;
    --public-branch) public_branch=$2; shift 2 ;;
    --release-branch) release_branch=$2; shift 2 ;;
    --expected-public-head) expected_public_head=$2; shift 2 ;;
    --force-delete-existing) force_delete=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is not clean" >&2
  exit 1
fi

git fetch "$public_remote" "$public_branch"
public_ref="$public_remote/$public_branch"
actual_public_head=$(git rev-parse "$public_ref")
if [[ -n "$expected_public_head" && "$actual_public_head" != "$expected_public_head" ]]; then
  echo "public head changed: expected $expected_public_head, found $actual_public_head" >&2
  echo "inspect the new public commits before updating --expected-public-head" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$release_branch"; then
  if [[ "$force_delete" != true ]]; then
    echo "local branch already exists: $release_branch" >&2
    exit 1
  fi
  git branch -D "$release_branch"
fi

git switch -c "$release_branch"
git merge --allow-unrelated-histories -s ours "$public_ref" \
  -m "Merge legacy public history before 2.0.0"

cat <<EOF
Created $release_branch at $(git rev-parse HEAD)
Public history parent: $actual_public_head
The hardened source tree was preserved. Review the merge, push the branch, and open a PR.
EOF
