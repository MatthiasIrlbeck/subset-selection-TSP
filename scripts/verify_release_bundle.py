#!/usr/bin/env python3
"""Verify a complete source-release payload before it is uploaded or published."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_archive_name(name: str) -> PurePosixPath:
    require(name != "", "archive contains an empty member name")
    require("\x00" not in name, f"archive member contains NUL: {name!r}")
    require("\\" not in name, f"archive member uses a backslash: {name!r}")
    require(not name.startswith("/"), f"archive member is absolute: {name!r}")
    path = PurePosixPath(name.rstrip("/"))
    require(not path.is_absolute(), f"archive member is absolute: {name!r}")
    require(path.parts and all(part not in {"", ".", ".."} for part in path.parts),
            f"archive member escapes its root: {name!r}")
    return path


def parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    names: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(.+)", line)
        require(match is not None, f"{path.name}:{number}: malformed checksum line")
        digest, name = match.groups()
        require("/" not in name and "\\" not in name, f"unsafe checksum path: {name}")
        require(name not in entries, f"duplicate checksum entry: {name}")
        entries[name] = digest
        names.append(name)
    require(names == sorted(names), "checksum manifest is not sorted by artifact name")
    return entries


def archive_prefix(names: list[str]) -> str:
    require(names, "source archive is empty")
    paths = [safe_archive_name(name) for name in names]
    roots = {path.parts[0] for path in paths}
    require(len(roots) == 1,
            f"source archive must contain one top-level directory, found {sorted(roots)}")
    return next(iter(roots)) + "/"


def normalized_zip(path: Path) -> tuple[str, dict[str, tuple[str, int, bytes]]]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        raw_names = [info.filename for info in infos]
        require(len(raw_names) == len(set(raw_names)), "ZIP contains duplicate member names")
        prefix = archive_prefix(raw_names)
        entries: dict[str, tuple[str, int, bytes]] = {}
        for info in infos:
            safe_archive_name(info.filename)
            require(info.create_system == 3,
                    f"ZIP member lacks Unix mode metadata: {info.filename}")
            require((info.flag_bits & 0x1) == 0,
                    f"ZIP member is unexpectedly encrypted: {info.filename}")
            if info.filename.rstrip("/") == prefix.rstrip("/"):
                require(info.is_dir(), "ZIP top-level prefix is not a directory")
                continue
            require(info.filename.startswith(prefix),
                    f"ZIP member is outside the top-level prefix: {info.filename}")
            relative = info.filename[len(prefix):]
            require(relative != "", "ZIP contains an empty relative member")
            mode = (info.external_attr >> 16) & 0o7777
            file_type = (info.external_attr >> 16) & 0o170000
            if info.is_dir():
                require(file_type == stat.S_IFDIR,
                        f"ZIP directory has inconsistent mode metadata: {info.filename}")
                kind, payload = "dir", b""
                relative = relative.rstrip("/")
            elif file_type == stat.S_IFLNK:
                kind, payload = "symlink", archive.read(info)
                target = payload.decode("utf-8")
                safe_archive_name(target)
            else:
                require(file_type == stat.S_IFREG,
                        f"ZIP member has unsupported file type: {info.filename}")
                kind, payload = "file", archive.read(info)
            require(relative not in entries, f"duplicate ZIP relative path: {relative}")
            entries[relative] = (kind, mode, payload)
        return prefix, entries


def normalized_tar(path: Path) -> tuple[str, dict[str, tuple[str, int, bytes]]]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        raw_names = [member.name for member in members]
        require(len(raw_names) == len(set(raw_names)), "tar contains duplicate member names")
        prefix = archive_prefix(raw_names)
        entries: dict[str, tuple[str, int, bytes]] = {}
        for member in members:
            safe_archive_name(member.name)
            if member.name.rstrip("/") == prefix.rstrip("/"):
                require(member.isdir(), "tar top-level prefix is not a directory")
                continue
            require(member.name.startswith(prefix),
                    f"tar member is outside the top-level prefix: {member.name}")
            relative = member.name[len(prefix):]
            require(relative != "", "tar contains an empty relative member")
            mode = member.mode & 0o7777
            if member.isdir():
                kind, payload = "dir", b""
                relative = relative.rstrip("/")
            elif member.issym():
                safe_archive_name(member.linkname)
                kind, payload = "symlink", member.linkname.encode("utf-8")
            elif member.isfile():
                extracted = archive.extractfile(member)
                require(extracted is not None, f"could not read tar member {member.name}")
                kind, payload = "file", extracted.read()
            else:
                raise RuntimeError(f"unsupported tar member type: {member.name}")
            require(relative not in entries, f"duplicate tar relative path: {relative}")
            entries[relative] = (kind, mode, payload)
        return prefix, entries


def parse_source_revision(entries: dict[str, tuple[str, int, bytes]]) -> dict[str, str]:
    record = entries.get("SOURCE_REVISION")
    require(record is not None and record[0] == "file",
            "SOURCE_REVISION is absent from source archive")
    text = record[2].decode("utf-8")
    require("$Format" not in text, "SOURCE_REVISION contains unexpanded Git placeholders")
    parsed: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        require("=" in line, f"SOURCE_REVISION:{number}: malformed line")
        key, value = line.split("=", 1)
        require(key in {"commit", "tree", "refnames"},
                f"SOURCE_REVISION:{number}: unknown field {key!r}")
        require(key not in parsed, f"SOURCE_REVISION contains duplicate field {key!r}")
        parsed[key] = value
    require(set(parsed) == {"commit", "tree", "refnames"},
            "SOURCE_REVISION is missing required fields")
    require(_GIT_OID_RE.fullmatch(parsed["commit"]) is not None,
            "SOURCE_REVISION commit is not a SHA-1 object ID")
    require(_GIT_OID_RE.fullmatch(parsed["tree"]) is not None,
            "SOURCE_REVISION tree is not a SHA-1 object ID")
    return parsed


def verify_provenance(
    path: Path,
    dist: Path,
    expected_subjects: set[str],
    version: str,
    commit: str,
    tree: str,
) -> None:
    statement = json.loads(path.read_text(encoding="utf-8"))
    require(statement.get("_type") == "https://in-toto.io/Statement/v1",
            "invalid in-toto statement type")
    require(statement.get("predicateType") == "https://slsa.dev/provenance/v1",
            "invalid provenance predicate type")
    subjects = statement.get("subject")
    require(isinstance(subjects, list) and subjects, "provenance has no subjects")
    observed: set[str] = set()
    for subject in subjects:
        require(isinstance(subject, dict), "malformed provenance subject")
        name = subject.get("name")
        digest = subject.get("digest", {}).get("sha256")
        require(isinstance(name, str) and isinstance(digest, str),
                "malformed provenance subject")
        require(_SHA256_RE.fullmatch(digest) is not None,
                f"malformed provenance SHA-256: {name}")
        require(name not in observed, f"duplicate provenance subject: {name}")
        observed.add(name)
        artifact = dist / name
        require(artifact.is_file(), f"provenance subject is missing: {name}")
        require(sha256(artifact) == digest, f"provenance digest mismatch: {name}")
    require(observed == expected_subjects,
            f"provenance subject inventory mismatch: {sorted(observed)}")

    build_definition = statement.get("predicate", {}).get("buildDefinition", {})
    parameters = build_definition.get("externalParameters", {})
    require(parameters.get("version") == version, "provenance version mismatch")
    require(parameters.get("gitCommit") == commit, "provenance commit mismatch")
    require(parameters.get("gitTree") == tree, "provenance tree mismatch")
    dependencies = build_definition.get("resolvedDependencies")
    require(isinstance(dependencies, list) and len(dependencies) == 1,
            "provenance must identify exactly one source dependency")
    dependency = dependencies[0]
    require(dependency.get("digest") == {"gitCommit": commit, "gitTree": tree},
            "provenance source dependency digest mismatch")
    require(str(dependency.get("uri", "")).endswith(f"@{commit}"),
            "provenance source dependency URI mismatch")


def checksum_map(record: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in record.get("checksums", []):
        require(isinstance(item, dict), "malformed SPDX checksum record")
        algorithm = item.get("algorithm")
        value = item.get("checksumValue")
        require(isinstance(algorithm, str) and isinstance(value, str),
                "malformed SPDX checksum record")
        require(algorithm not in values, f"duplicate SPDX {algorithm} checksum")
        values[algorithm] = value
    return values


def verify_sbom(
    path: Path,
    entries: dict[str, tuple[str, int, bytes]],
    version: str,
    commit: str,
    tree: str,
) -> None:
    sbom = json.loads(path.read_text(encoding="utf-8"))
    require(sbom.get("spdxVersion") == "SPDX-2.3", "SBOM is not SPDX 2.3")
    packages = sbom.get("packages")
    require(isinstance(packages, list) and len(packages) == 1,
            "SBOM must contain exactly one package")
    package = packages[0]
    require(package.get("name") == "aldous-tsp", "SBOM package name mismatch")
    require(package.get("versionInfo") == version, "SBOM version mismatch")
    external_refs = package.get("externalRefs")
    require(isinstance(external_refs, list), "SBOM package external references are absent")
    locators = {str(item.get("referenceLocator", "")) for item in external_refs
                if isinstance(item, dict)}
    require(f"gitoid:tree:sha1:{tree}" in locators, "SBOM tree identity mismatch")
    require(any(locator.endswith(f"@{commit}") for locator in locators),
            "SBOM commit identity mismatch")

    expected_files = {name: payload for name, (kind, _mode, payload) in entries.items()
                      if kind == "file"}
    records = sbom.get("files")
    require(isinstance(records, list), "SBOM file inventory is absent")
    observed: dict[str, str] = {}
    sha1_values: list[str] = []
    for record in records:
        require(isinstance(record, dict), "malformed SPDX file record")
        file_name = record.get("fileName")
        require(isinstance(file_name, str) and file_name.startswith("./"),
                "malformed SPDX file name")
        relative = file_name[2:]
        safe_archive_name(relative)
        require(relative not in observed, f"duplicate SPDX file record: {relative}")
        require(relative in expected_files, f"SBOM records a non-archive file: {relative}")
        checksums = checksum_map(record)
        payload = expected_files[relative]
        expected_sha1 = hashlib.sha1(payload).hexdigest()
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        require(checksums.get("SHA1") == expected_sha1,
                f"SBOM SHA-1 mismatch: {relative}")
        require(checksums.get("SHA256") == expected_sha256,
                f"SBOM SHA-256 mismatch: {relative}")
        observed[relative] = expected_sha256
        sha1_values.append(expected_sha1)
    require(set(observed) == set(expected_files),
            "SBOM file inventory does not match the source archive")
    verification = hashlib.sha1("".join(sorted(sha1_values)).encode()).hexdigest()
    actual_verification = package.get("packageVerificationCode", {}).get(
        "packageVerificationCodeValue")
    require(actual_verification == verification,
            "SBOM package verification code mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    args = parser.parse_args()

    require(_VERSION_RE.fullmatch(args.expected_version) is not None,
            f"invalid expected version: {args.expected_version}")
    require(_GIT_OID_RE.fullmatch(args.expected_commit) is not None,
            "expected commit is not a 40-character SHA-1 object ID")
    require(_GIT_OID_RE.fullmatch(args.expected_tree) is not None,
            "expected tree is not a 40-character SHA-1 object ID")

    dist = args.dist.resolve()
    require(dist.is_dir(), f"release directory not found: {dist}")
    base = f"subset-selection-TSP-{args.expected_version}"
    names = {
        "zip": f"{base}.zip",
        "tar": f"{base}.tar.gz",
        "sbom": f"{base}.spdx.json",
        "provenance": f"{base}-provenance.json",
    }
    expected_artifacts = set(names.values())
    children = list(dist.iterdir())
    require(all(path.is_file() for path in children),
            "release directory contains a non-file entry")
    checksum_path = dist / "SHA256SUMS"
    require(checksum_path.is_file(), "SHA256SUMS is missing")
    actual_artifacts = {path.name for path in children if path.name != "SHA256SUMS"}
    require(actual_artifacts == expected_artifacts,
            f"release artifact inventory mismatch: {sorted(actual_artifacts)}")

    expected_checksums = parse_checksums(checksum_path)
    require(set(expected_checksums) == expected_artifacts,
            "checksum inventory does not match the release artifact inventory")
    for name, digest in expected_checksums.items():
        require(sha256(dist / name) == digest, f"SHA-256 mismatch: {name}")

    zip_prefix, zip_entries = normalized_zip(dist / names["zip"])
    tar_prefix, tar_entries = normalized_tar(dist / names["tar"])
    require(zip_prefix == tar_prefix, "ZIP and tar top-level prefixes differ")
    require(zip_entries == tar_entries, "ZIP and tar source trees, modes, or bytes differ")
    require(zip_prefix == f"{base}/", f"unexpected archive prefix: {zip_prefix}")

    revision = parse_source_revision(zip_entries)
    require(revision["commit"] == args.expected_commit, "SOURCE_REVISION commit mismatch")
    require(revision["tree"] == args.expected_tree, "SOURCE_REVISION tree mismatch")

    verify_sbom(
        dist / names["sbom"],
        zip_entries,
        args.expected_version,
        args.expected_commit,
        args.expected_tree,
    )
    verify_provenance(
        dist / names["provenance"],
        dist,
        {names["zip"], names["tar"], names["sbom"]},
        args.expected_version,
        args.expected_commit,
        args.expected_tree,
    )

    print(
        f"verified release bundle: {len(expected_artifacts)} artifacts, "
        f"{len(zip_entries)} source entries, prefix {zip_prefix}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile,
            tarfile.TarError) as exc:
        raise SystemExit(f"release verification failed: {exc}") from exc
