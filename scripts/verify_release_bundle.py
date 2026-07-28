#!/usr/bin/env python3
"""Verify checksums, provenance, and source-archive equivalence for a release directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import tarfile
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(.+)", line)
        require(match is not None, f"{path.name}:{number}: malformed checksum line")
        digest, name = match.groups()
        require("/" not in name and "\\" not in name, f"unsafe checksum path: {name}")
        require(name not in entries, f"duplicate checksum entry: {name}")
        entries[name] = digest
    return entries


def archive_prefix(names: list[str]) -> str:
    roots = {name.split("/", 1)[0] for name in names if name and not name.startswith("/")}
    require(len(roots) == 1, f"source archive must contain one top-level directory, found {sorted(roots)}")
    return next(iter(roots)) + "/"


def normalized_zip(path: Path) -> tuple[str, dict[str, tuple[str, int, bytes]]]:
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist()]
        prefix = archive_prefix(names)
        entries: dict[str, tuple[str, int, bytes]] = {}
        for info in archive.infolist():
            if info.filename == prefix:
                continue
            relative = info.filename[len(prefix):]
            mode = (info.external_attr >> 16) & 0o7777
            file_type = (info.external_attr >> 16) & 0o170000
            if info.is_dir():
                kind, payload = "dir", b""
            elif file_type == stat.S_IFLNK:
                kind, payload = "symlink", archive.read(info)
            else:
                kind, payload = "file", archive.read(info)
            entries[relative.rstrip("/") if kind == "dir" else relative] = (kind, mode, payload)
        return prefix, entries


def normalized_tar(path: Path) -> tuple[str, dict[str, tuple[str, int, bytes]]]:
    with tarfile.open(path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        prefix = archive_prefix(names)
        entries: dict[str, tuple[str, int, bytes]] = {}
        for member in archive.getmembers():
            if member.name.rstrip("/") == prefix.rstrip("/"):
                continue
            relative = member.name[len(prefix):]
            mode = member.mode & 0o7777
            if member.isdir():
                kind, payload = "dir", b""
                relative = relative.rstrip("/")
            elif member.issym():
                kind, payload = "symlink", member.linkname.encode("utf-8")
            elif member.isfile():
                extracted = archive.extractfile(member)
                require(extracted is not None, f"could not read tar member {member.name}")
                kind, payload = "file", extracted.read()
            else:
                raise RuntimeError(f"unsupported tar member type: {member.name}")
            entries[relative] = (kind, mode, payload)
        return prefix, entries


def source_revision(entries: dict[str, tuple[str, int, bytes]]) -> str:
    record = entries.get("SOURCE_REVISION")
    require(record is not None and record[0] == "file", "SOURCE_REVISION is absent from source archive")
    return record[2].decode("utf-8")


def verify_provenance(path: Path, dist: Path) -> None:
    statement = json.loads(path.read_text(encoding="utf-8"))
    require(statement.get("_type") == "https://in-toto.io/Statement/v1", "invalid in-toto statement type")
    require(statement.get("predicateType") == "https://slsa.dev/provenance/v1", "invalid provenance predicate type")
    subjects = statement.get("subject")
    require(isinstance(subjects, list) and subjects, "provenance has no subjects")
    for subject in subjects:
        name = subject.get("name")
        digest = subject.get("digest", {}).get("sha256")
        require(isinstance(name, str) and isinstance(digest, str), "malformed provenance subject")
        artifact = dist / name
        require(artifact.is_file(), f"provenance subject is missing: {name}")
        require(sha256(artifact) == digest, f"provenance digest mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tree")
    args = parser.parse_args()

    dist = args.dist.resolve()
    require(dist.is_dir(), f"release directory not found: {dist}")
    checksum_path = dist / "SHA256SUMS"
    require(checksum_path.is_file(), "SHA256SUMS is missing")
    expected = parse_checksums(checksum_path)
    actual_files = sorted(path.name for path in dist.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    require(set(expected) == set(actual_files),
            f"checksum inventory mismatch: expected {sorted(expected)}, files {actual_files}")
    for name, digest in expected.items():
        require(sha256(dist / name) == digest, f"SHA-256 mismatch: {name}")

    provenance_files = sorted(dist.glob("*-provenance.json"))
    require(len(provenance_files) == 1, "release must contain exactly one provenance statement")
    verify_provenance(provenance_files[0], dist)

    zip_files = sorted(dist.glob("*.zip"))
    tar_files = sorted(dist.glob("*.tar.gz"))
    require(len(zip_files) == 1 and len(tar_files) == 1, "release must contain one ZIP and one tar.gz")
    zip_prefix, zip_entries = normalized_zip(zip_files[0])
    tar_prefix, tar_entries = normalized_tar(tar_files[0])
    require(zip_prefix == tar_prefix, "ZIP and tar top-level prefixes differ")
    require(zip_entries == tar_entries, "ZIP and tar source trees, modes, or bytes differ")

    if args.expected_version:
        require(zip_prefix == f"subset-selection-TSP-{args.expected_version}/",
                f"unexpected archive prefix: {zip_prefix}")
    revision = source_revision(zip_entries)
    require("$Format" not in revision, "SOURCE_REVISION contains unexpanded Git placeholders")
    if args.expected_commit:
        require(f"commit={args.expected_commit}" in revision, "SOURCE_REVISION commit mismatch")
    if args.expected_tree:
        require(f"tree={args.expected_tree}" in revision, "SOURCE_REVISION tree mismatch")

    sbom_files = sorted(dist.glob("*.spdx.json"))
    require(len(sbom_files) == 1, "release must contain exactly one SPDX SBOM")
    sbom = json.loads(sbom_files[0].read_text(encoding="utf-8"))
    require(sbom.get("spdxVersion") == "SPDX-2.3", "SBOM is not SPDX 2.3")
    if args.expected_version:
        packages = sbom.get("packages") or []
        require(packages and packages[0].get("versionInfo") == args.expected_version,
                "SBOM version does not match release version")

    print(
        f"verified release bundle: {len(actual_files)} artifacts, "
        f"{len(zip_entries)} source entries, prefix {zip_prefix}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"release verification failed: {exc}") from exc
