#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/prepare_public_release_branch.sh [options]

Join the reviewed hardened history with the existing public main branch without
changing the hardened source tree.

Run this from the bundle-cloned release/v2.0.0-prep branch while it still tracks
the bundle remote. The helper fetches that upstream and refuses any local commit
drift before it creates the public-history merge.

The resulting pull request MUST be merged on GitHub with "Create a merge commit".
Squash, rebase, and linear-history merging are incompatible with this import
because the retained development history contains validation and performance-
baseline commits referenced by the repository.

Options:
  --public-remote NAME          remote containing public main (default: origin)
  --public-branch NAME          public branch (default: main)
  --release-branch NAME         branch to create (default: release/v2.0.0)
  --expected-public-head SHA    fail if public main differs from this SHA
  --expected-hardened-branch NAME
                                required reviewed source branch
                                (default: release/v2.0.0-prep)
  --expected-hardened-parent SHA
                                required parent of the reviewed preparation commit
                                (default: 4045b8b7f8ebb7ca52eca9445828f4bb753da035)
  --force-delete-existing       delete an existing local release branch first
USAGE
}

public_remote=origin
public_branch=main
release_branch=release/v2.0.0
expected_public_head=9435423964f8cef5222d6802cd3d25e8da9fcace
expected_hardened_branch=release/v2.0.0-prep
expected_hardened_parent=4045b8b7f8ebb7ca52eca9445828f4bb753da035
force_delete=false

while (($#)); do
  case "$1" in
    --public-remote) public_remote=$2; shift 2 ;;
    --public-branch) public_branch=$2; shift 2 ;;
    --release-branch) release_branch=$2; shift 2 ;;
    --expected-public-head) expected_public_head=$2; shift 2 ;;
    --expected-hardened-branch) expected_hardened_branch=$2; shift 2 ;;
    --expected-hardened-parent) expected_hardened_parent=$2; shift 2 ;;
    --force-delete-existing) force_delete=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not inside a Git working tree" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is not clean" >&2
  exit 1
fi

current_branch=$(git branch --show-current)
if [[ -z "$current_branch" ]]; then
  echo "detached HEAD is not allowed; check out $expected_hardened_branch" >&2
  exit 1
fi
if [[ "$current_branch" != "$expected_hardened_branch" ]]; then
  echo "unexpected hardened branch: expected $expected_hardened_branch, found $current_branch" >&2
  exit 1
fi

upstream_ref=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)
if [[ -z "$upstream_ref" ]]; then
  echo "the hardened branch has no upstream; clone the reviewed Git bundle or set its bundle-remote upstream" >&2
  exit 1
fi
upstream_remote=${upstream_ref%%/*}
upstream_branch=${upstream_ref#*/}
if [[ "$upstream_branch" != "$expected_hardened_branch" ]]; then
  echo "unexpected hardened upstream: expected */$expected_hardened_branch, found $upstream_ref" >&2
  exit 1
fi

git fetch "$upstream_remote" "$upstream_branch"
local_hardened_head=$(git rev-parse HEAD)
reviewed_hardened_head=$(git rev-parse '@{upstream}')
if [[ "$local_hardened_head" != "$reviewed_hardened_head" ]]; then
  echo "local hardened head differs from the reviewed bundle upstream" >&2
  echo "local:    $local_hardened_head" >&2
  echo "reviewed: $reviewed_hardened_head" >&2
  echo "discard local commits or re-audit and regenerate the bundle before continuing" >&2
  exit 1
fi

read -r -a hardened_commit_line <<< "$(git rev-list --parents -n 1 HEAD)"
if ((${#hardened_commit_line[@]} != 2)); then
  echo "reviewed hardened head must be one commit directly above its audited parent" >&2
  exit 1
fi
actual_hardened_parent=${hardened_commit_line[1]}
if [[ -n "$expected_hardened_parent" && "$actual_hardened_parent" != "$expected_hardened_parent" ]]; then
  echo "unexpected hardened parent: expected $expected_hardened_parent, found $actual_hardened_parent" >&2
  echo "use only the final reviewed bundle or explicitly provide its audited parent" >&2
  exit 1
fi

hardened_tree=$(git rev-parse 'HEAD^{tree}')

performance_baseline=""
if [[ -f config/performance_baseline_commit.txt ]]; then
  performance_baseline=$(tr -d '[:space:]' < config/performance_baseline_commit.txt)
  if [[ -z "$performance_baseline" ]] || ! git cat-file -e "$performance_baseline^{commit}" 2>/dev/null; then
    echo "configured performance baseline is missing from the reviewed hardened history" >&2
    exit 1
  fi
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

merged_tree=$(git rev-parse 'HEAD^{tree}')
if [[ "$merged_tree" != "$hardened_tree" ]]; then
  echo "fatal: public-history merge changed the hardened source tree" >&2
  exit 1
fi
if [[ -n "$performance_baseline" ]] && ! git merge-base --is-ancestor "$performance_baseline" HEAD; then
  echo "fatal: performance-baseline commit became unreachable after the merge" >&2
  exit 1
fi

cat <<EOF_SUMMARY
Created $release_branch at $(git rev-parse HEAD)
Reviewed hardened head: $reviewed_hardened_head
Public history parent: $actual_public_head
Preserved source tree: $merged_tree

IMPORTANT MERGE CONTRACT
------------------------
Open a pull request from $release_branch to $public_branch and merge it using a
true merge commit: GitHub's "Create a merge commit" action.

Do NOT use "Squash and merge" or "Rebase and merge", and do not require linear
history for this one import PR. Rewriting the prepared history can make the
recorded validation and performance-baseline commits unreachable.
EOF_SUMMARY
