#!/usr/bin/env python3
"""Regression tests for immutable release inputs and archive provenance."""

from __future__ import annotations

import hashlib
import json
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
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_workflow_pins(root: Path) -> None:
    uses_re = re.compile(r"\buses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9A-Fa-f]+)")
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
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    require(".verification.verified" in release, "release tags are not signature-verified")
    require('--root "$source_dir"' in release,
            "SBOM is not generated from the archived source bytes")
    require("python3 scripts/create_source_archives.py" in release,
            "release archives do not use the mode-preserving generator")


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
    require("scripts/verified_download.py" in workflow, "oracle workflow bypasses verified downloader")
    require("sha256sum" not in workflow or "--check" in workflow,
            "oracle workflow contains an unchecked checksum command")


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


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} REPOSITORY_ROOT")
    root = Path(sys.argv[1]).resolve()
    test_workflow_pins(root)
    test_oracle_lock(root)
    test_sbom_and_provenance(root)
    test_source_archive_modes(root)
    test_source_archive_fallback(root)
    print("release security self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
