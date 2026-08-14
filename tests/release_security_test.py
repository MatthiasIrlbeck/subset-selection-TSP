#!/usr/bin/env python3
"""Regression tests for immutable release inputs and archive provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

PINNED_ACTIONS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python": "8d9ed9ac5c53483de85588cdf95a591a75ab9f55",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/attest-build-provenance": "96278af6caaf10aea03fd8d33a09a777ca52d62f",
    "github/codeql-action": "9e0d7b8d25671d64c341c19c0152d693099fb5ba",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_workflow_pins(root: Path) -> None:
    uses_re = re.compile(
        r"\buses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
        r"(?:/[A-Za-z0-9_.-]+)*@([0-9A-Fa-f]+)"
    )
    seen: set[str] = set()
    for workflow in sorted((root / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "uses:" not in line:
                continue
            match = uses_re.search(line)
            require(match is not None, f"{workflow}:{line_number}: action is not SHA-pinned")
            action, revision = match.groups()
            require(len(revision) == 40, f"{workflow}:{line_number}: action pin is not a full SHA")
            expected = PINNED_ACTIONS.get(action)
            require(expected is not None, f"{workflow}:{line_number}: unreviewed action {action}")
            require(revision == expected, f"{workflow}:{line_number}: unexpected pin for {action}")
            seen.add(action)
    require("actions/attest-build-provenance" in seen, "release provenance action is absent")
    require("github/codeql-action" in seen, "CodeQL workflow is absent")
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    require(".verification.verified" in release, "release tags are not signature-verified")
    require('--root "$source_dir"' in release,
            "SBOM is not generated from the archived source bytes")
    require("python3 scripts/create_source_archives.py" in release,
            "release archives do not use the mode-preserving generator")
    require("--notes-file" in release,
            "release publication does not use curated release notes")
    require("--draft" in release and "--latest=false" in release,
            "release workflow does not stage a reviewable draft release")
    require("--fail-on-no-commits" in release,
            "release workflow can publish an accidental duplicate release")
    require("scripts/verify_release_bundle.py" in release,
            "release workflow does not verify its final artifact inventory")

    codeql = (root / ".github" / "workflows" / "codeql.yml").read_text(
        encoding="utf-8"
    )
    require("queries: security-extended" in codeql,
            "blocking CodeQL does not use the focused extended-security suite")
    require("security-and-quality" not in codeql,
            "blocking CodeQL mixes advisory maintainability findings into security")
    require('branches: [main, "release/**"]' not in codeql,
            "CodeQL runs duplicate push and pull-request analyses on release branches")

    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    require("concurrency:" in ci and "cancel-in-progress: true" in ci,
            "CI does not cancel superseded pull-request runs")


def test_oracle_lock(root: Path) -> None:
    lock = json.loads((root / "config" / "oracle_sources.json").read_text(encoding="utf-8"))
    require(lock.get("schema_version") == 1, "oracle source lock has unsupported schema")
    sources = lock.get("sources")
    require(isinstance(sources, dict) and sources, "oracle source lock is empty")
    for name, source in sources.items():
        require(isinstance(source, dict), f"oracle source {name} is malformed")
        require(str(source.get("url", "")).startswith("https://"), f"oracle source {name} is not HTTPS")
        require(isinstance(source.get("max_bytes"), int) and source["max_bytes"] > 0,
                f"oracle source {name} has no size bound")
        require(re.fullmatch(r"[A-Z0-9_]+", str(source.get("digest_variable", ""))) is not None,
                f"oracle source {name} has no digest-variable contract")
    workflow = (root / ".github" / "workflows" / "real-oracles.yml").read_text(encoding="utf-8")
    require("workflow_dispatch:" in workflow, "oracle workflow is not manually dispatchable")
    require(
        re.search(r"(?m)^\s*schedule:\s*$", workflow) is None,
        "real-oracle schedule must remain disabled until approved digest variables exist",
    )
    require("scripts/verified_download.py" in workflow, "oracle workflow bypasses verified downloader")
    require("sha256sum" not in workflow or "--check" in workflow,
            "oracle workflow contains an unchecked checksum command")


def test_release_metadata(root: Path) -> None:
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(aldous_tsp VERSION ([^\s)]+)", cmake)
    require(match is not None, "CMake project version is missing")
    version = match.group(1)

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r'(?m)^version:\s*["\']?([^"\'\s]+)', citation)
    require(citation_match is not None, "CITATION.cff version is missing")
    require(citation_match.group(1) == version, "CITATION.cff and CMake versions differ")
    require(
        re.search(r"(?m)^date-released:", citation) is None,
        "release candidate must not claim a speculative publication date",
    )


def test_sbom_and_provenance(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="aldous-release-security-") as raw:
        temp = Path(raw)
        source = temp / "source"
        source.mkdir()
        (source / "a.txt").write_text("alpha\n", encoding="utf-8")
        (source / "b.bin").write_bytes(b"\x00\x01\x02")
        sbom = temp / "out.spdx.json"
        env = {"SOURCE_DATE_EPOCH": "1700000000"}
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "generate_sbom.py"),
                "--root", str(source),
                "--output", str(sbom),
                "--version", "test",
                "--commit", "1" * 40,
                "--tree", "2" * 40,
            ],
            check=True,
            env=env,
        )
        doc = json.loads(sbom.read_text(encoding="utf-8"))
        require(doc["spdxVersion"] == "SPDX-2.3", "SBOM is not SPDX 2.3")
        require(len(doc["files"]) == 2, "SBOM does not enumerate source files")
        expected_sha256 = hashlib.sha256((source / "a.txt").read_bytes()).hexdigest()
        require(
            any(
                any(
                    checksum["algorithm"] == "SHA256"
                    and checksum["checksumValue"] == expected_sha256
                    for checksum in record["checksums"]
                )
                for record in doc["files"]
            ),
            "SBOM file digest is incorrect",
        )
        sha1_values = sorted(
            hashlib.sha1(path.read_bytes()).hexdigest()
            for path in (source / "a.txt", source / "b.bin")
        )
        expected_verification = hashlib.sha1("".join(sha1_values).encode()).hexdigest()
        actual_verification = doc["packages"][0]["packageVerificationCode"][
            "packageVerificationCodeValue"
        ]
        require(
            actual_verification == expected_verification,
            "SPDX package verification code does not follow the SPDX algorithm",
        )

        artifact = temp / "artifact.zip"
        artifact.write_bytes(b"archive")
        provenance = temp / "provenance.json"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "generate_provenance.py"),
                "--output", str(provenance),
                "--artifact", str(artifact),
                "--artifact", str(sbom),
                "--version", "test",
                "--commit", "1" * 40,
                "--tree", "2" * 40,
            ],
            check=True,
            env=env,
        )
        statement = json.loads(provenance.read_text(encoding="utf-8"))
        require(statement["_type"] == "https://in-toto.io/Statement/v1",
                "provenance is not an in-toto Statement v1")
        require(statement["predicateType"] == "https://slsa.dev/provenance/v1",
                "provenance predicate is not SLSA v1")
        require(len(statement["subject"]) == 2, "provenance subjects are incomplete")


def test_release_bundle_verifier(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="aldous-release-bundle-") as raw:
        temp = Path(raw)
        repository = temp / "repository"
        repository.mkdir()
        (repository / ".gitattributes").write_text(
            "SOURCE_REVISION export-subst\n", encoding="utf-8"
        )
        (repository / "SOURCE_REVISION").write_text(
            "commit=$Format:%H$\n"
            "tree=$Format:%T$\n"
            "refnames=$Format:%D$\n",
            encoding="utf-8",
        )
        (repository / "README.md").write_text("release fixture\n", encoding="utf-8")
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "release-test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.email",
             "release-test@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(repository), "add", "--all"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "--quiet", "-m", "release fixture"],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()

        version = "2.0.0-test.1"
        base = f"subset-selection-TSP-{version}"
        dist = temp / "dist"
        dist.mkdir()
        zip_path = dist / f"{base}.zip"
        tar_path = dist / f"{base}.tar.gz"
        sbom_path = dist / f"{base}.spdx.json"
        provenance_path = dist / f"{base}-provenance.json"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "create_source_archives.py"),
                "--repo", str(repository),
                "--ref", "HEAD",
                "--prefix", f"{base}/",
                "--zip", str(zip_path),
                "--tar-gz", str(tar_path),
            ],
            check=True,
        )
        source = temp / "source"
        import zipfile
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(source)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "generate_sbom.py"),
                "--root", str(source / base),
                "--output", str(sbom_path),
                "--version", version,
                "--commit", commit,
                "--tree", tree,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "generate_provenance.py"),
                "--output", str(provenance_path),
                "--artifact", str(zip_path),
                "--artifact", str(tar_path),
                "--artifact", str(sbom_path),
                "--version", version,
                "--commit", commit,
                "--tree", tree,
            ],
            check=True,
        )
        artifacts = sorted((zip_path, tar_path, sbom_path, provenance_path),
                           key=lambda item: item.name)
        (dist / "SHA256SUMS").write_text(
            "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in artifacts),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(root / "scripts" / "verify_release_bundle.py"),
            "--dist", str(dist),
            "--expected-version", version,
            "--expected-commit", commit,
            "--expected-tree", tree,
        ]
        subprocess.run(command, check=True)

        zip_path.write_bytes(zip_path.read_bytes() + b"tampered")
        failed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        require(failed.returncode != 0,
                "release bundle verifier accepted a checksum-invalid archive")


def is_git_worktree(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def archive_test_repository(root: Path, temp: Path) -> Path:
    if is_git_worktree(root):
        return root

    # A downloaded source archive deliberately has no .git directory.  Build a
    # tiny temporary repository so the mode-preserving archive generator is
    # still tested rather than silently skipped.
    repository = temp / "synthetic-repository"
    (repository / "scripts").mkdir(parents=True)
    for relative in (
        Path(".clang-format"),
        Path(".gitattributes"),
        Path("SOURCE_REVISION"),
        Path("scripts/generate_options.py"),
    ):
        source = root / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "archive-test"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "archive-test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "--all"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "--quiet", "-m", "synthetic archive fixture"],
        check=True,
    )
    return repository


def test_source_archive_modes(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="aldous-tsp-archive-mode-") as tmp:
        temp = Path(tmp)
        repository = archive_test_repository(root, temp)
        first_zip = temp / "first.zip"
        first_tar = temp / "first.tar.gz"
        second_zip = temp / "second.zip"
        second_tar = temp / "second.tar.gz"
        common = [
            sys.executable,
            str(root / "scripts" / "create_source_archives.py"),
            "--repo",
            str(repository),
            "--ref",
            "HEAD",
            "--prefix",
            "archive-test/",
        ]
        subprocess.run(
            common + ["--zip", str(first_zip), "--tar-gz", str(first_tar)],
            check=True,
        )
        subprocess.run(
            common + ["--zip", str(second_zip), "--tar-gz", str(second_tar)],
            check=True,
        )
        require(first_zip.read_bytes() == second_zip.read_bytes(),
                "source ZIP generation is not deterministic")
        require(first_tar.read_bytes() == second_tar.read_bytes(),
                "source tar.gz generation is not deterministic")

        import tarfile
        import zipfile

        expected = {
            "archive-test/.clang-format": 0o644,
            "archive-test/scripts/generate_options.py": 0o755,
        }
        with zipfile.ZipFile(first_zip) as archive:
            for name, mode in expected.items():
                info = archive.getinfo(name)
                require(info.create_system == 3,
                        f"ZIP entry is not marked as Unix: {name}")
                actual = (info.external_attr >> 16) & 0o777
                require(actual == mode,
                        f"ZIP mode mismatch for {name}: {oct(actual)} != {oct(mode)}")
        with tarfile.open(first_tar, "r:gz") as archive:
            for name, mode in expected.items():
                actual = archive.getmember(name).mode & 0o777
                require(actual == mode,
                        f"tar mode mismatch for {name}: {oct(actual)} != {oct(mode)}")


def test_source_archive_fallback(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="aldous-source-archive-") as raw:
        temp = Path(raw)
        source = temp / "source"

        def ignore(directory: str, names: list[str]) -> set[str]:
            ignored = {name for name in names if name == ".git" or name.startswith("build-")}
            ignored.update({"__pycache__", "dist"}.intersection(names))
            return ignored

        shutil.copytree(root, source, ignore=ignore)
        commit = "a" * 40
        tree = "b" * 40
        archive_refnames = 'tag: v-test, tag: v"quoted'
        (source / "SOURCE_REVISION").write_text(
            f"commit={commit}\ntree={tree}\nrefnames={archive_refnames}\n", encoding="utf-8"
        )
        build = temp / "build"
        subprocess.run(
            [
                "cmake", "-S", str(source), "-B", str(build),
                "-DBUILD_TESTING=OFF",
                "-DALDOUS_TSP_BUILD_TESTS=OFF",
                "-DALDOUS_TSP_BUILD_CLI=OFF",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        header = (build / "generated" / "aldous_tsp" / "version.hpp").read_text(encoding="utf-8")
        require(f'kGitCommit = "{commit}"' in header, "source archive lost commit identity")
        require(f'kGitTree = "{tree}"' in header, "source archive lost tree identity")
        require('kRevisionSource = "source-archive"' in header,
                "source archive provenance source was not detected")
        require('kSourceRefNames = "tag: v-test, tag: v\\"quoted";' in header,
                "source archive ref names were not parsed and C++-escaped safely")
        require("kSourceDirty = false" in header, "source archive was incorrectly marked dirty")


def test_public_release_preparation_contract(root: Path) -> None:
    helper_path = root / "scripts" / "prepare_public_release_branch.sh"
    helper = helper_path.read_text(encoding="utf-8")
    checklist = (root / "docs" / "releases" / "publication_checklist.md").read_text(
        encoding="utf-8"
    )
    development = (root / "DEVELOPMENT.md").read_text(encoding="utf-8")
    pr_template = (root / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    require("expected_hardened_branch=release/v2.0.0-prep" in helper,
            "public-history helper does not require the reviewed candidate branch")
    require("expected_hardened_parent=4045b8b7f8ebb7ca52eca9445828f4bb753da035"
            in helper,
            "public-history helper does not bind the candidate to its audited parent")
    require("@{upstream}" in helper and "local hardened head differs" in helper,
            "public-history helper does not reject local candidate drift")
    require("git rev-list --parents -n 1 HEAD" in helper
            and "unexpected hardened parent" in helper,
            "public-history helper does not verify the candidate ancestry")
    require("performance_baseline_commit.txt" in helper,
            "public-history helper does not protect the historical performance baseline")
    require("Create a merge commit" in helper and "Squash and merge" in helper
            and "Rebase and merge" in helper,
            "public-history helper does not print the mandatory merge-method warning")
    require("Mandatory pull-request merge method" in checklist,
            "publication checklist lacks the mandatory merge-method section")
    require("Create a merge commit" in checklist and "linear-history" in checklist,
            "publication checklist does not prohibit ancestry-rewriting merge modes")
    require("Create a merge commit" in pr_template and "never squash/rebase" in pr_template,
            "pull-request template does not flag the release-history merge contract")
    require("OpenAI Repair" in development and "They do not imply" in development,
            "development attribution note is missing or ambiguous")

    if os.name == "nt":
        return

    with tempfile.TemporaryDirectory(prefix="aldous-public-history-") as raw:
        temp = Path(raw)
        candidate_bare = temp / "candidate.git"
        public_bare = temp / "public.git"
        candidate = temp / "candidate"
        public = temp / "public"

        subprocess.run(["git", "init", "--bare", "--quiet", str(candidate_bare)], check=True)
        subprocess.run(["git", "init", "--bare", "--quiet", str(public_bare)], check=True)
        subprocess.run(
            ["git", "init", "--quiet", "-b", "release/v2.0.0-prep", str(candidate)],
            check=True,
        )
        for repository in (candidate, public):
            repository.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(candidate), "config", "user.name", "release-test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(candidate), "config", "user.email",
             "release-test@example.invalid"],
            check=True,
        )

        (candidate / "candidate.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(candidate), "add", "candidate.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(candidate), "commit", "--quiet", "-m", "candidate base"],
            check=True,
        )
        candidate_parent = subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True
        ).strip()

        (candidate / "candidate.txt").write_text("reviewed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(candidate), "add", "candidate.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(candidate), "commit", "--quiet", "-m", "reviewed candidate"],
            check=True,
        )
        candidate_head = subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True
        ).strip()
        candidate_tree = subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()

        subprocess.run(
            ["git", "-C", str(candidate), "remote", "add", "bundle", str(candidate_bare)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(candidate), "push", "--quiet", "-u", "bundle",
             "release/v2.0.0-prep"],
            check=True,
        )

        subprocess.run(["git", "init", "--quiet", "-b", "main", str(public)], check=True)
        subprocess.run(
            ["git", "-C", str(public), "config", "user.name", "release-test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(public), "config", "user.email",
             "release-test@example.invalid"],
            check=True,
        )
        (public / "legacy.txt").write_text("legacy\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(public), "add", "legacy.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(public), "commit", "--quiet", "-m", "legacy"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(public), "remote", "add", "origin", str(public_bare)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(public), "push", "--quiet", "-u", "origin", "main"],
            check=True,
        )
        public_head = subprocess.check_output(
            ["git", "-C", str(public), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(
            ["git", "-C", str(candidate), "remote", "add", "origin", str(public_bare)],
            check=True,
        )

        common_args = [
            "bash", str(helper_path),
            "--public-remote", "origin",
            "--release-branch", "release/v2.0.0-test",
            "--expected-public-head", public_head,
            "--expected-hardened-parent", candidate_parent,
        ]

        (candidate / "local-only.txt").write_text("drift\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(candidate), "add", "local-only.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(candidate), "commit", "--quiet", "-m", "local drift"],
            check=True,
        )
        drift = subprocess.run(
            common_args,
            cwd=candidate,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        require(drift.returncode != 0 and "local hardened head differs" in drift.stderr,
                "public-history helper accepted an unreviewed local candidate commit")

        subprocess.run(
            ["git", "-C", str(candidate), "reset", "--hard", "--quiet", candidate_head],
            check=True,
        )
        merged = subprocess.run(
            common_args,
            cwd=candidate,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require("Create a merge commit" in merged.stdout,
                "public-history helper did not print the merge-method contract")
        merged_tree = subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        parents = subprocess.check_output(
            ["git", "-C", str(candidate), "show", "-s", "--format=%P", "HEAD"],
            text=True,
        ).split()
        require(merged_tree == candidate_tree,
                "history integration changed the hardened source tree")
        require(len(parents) == 2 and parents[0] == candidate_head and public_head in parents,
                "history integration did not produce the required two-parent merge")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} REPOSITORY_ROOT")
    root = Path(sys.argv[1]).resolve()
    test_workflow_pins(root)
    test_oracle_lock(root)
    test_release_metadata(root)
    test_public_release_preparation_contract(root)
    test_sbom_and_provenance(root)
    test_release_bundle_verifier(root)
    test_source_archive_modes(root)
    test_source_archive_fallback(root)
    print("release security self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
