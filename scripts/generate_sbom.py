#!/usr/bin/env python3
"""Generate a deterministic SPDX 2.3 JSON SBOM for a source tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-name", default="aldous-tsp")
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    return parser.parse_args()


def timestamp() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    now = datetime.fromtimestamp(int(raw), timezone.utc) if raw else datetime.now(timezone.utc)
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_digests(path: Path) -> tuple[str, str]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha1.update(chunk)
            sha256.update(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def iter_files(root: Path, output: Path) -> list[Path]:
    excluded = {".git", "dist", "__pycache__"}
    result: list[Path] = []
    output_resolved = output.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part in excluded or part.startswith("build-") for part in rel.parts):
            continue
        if path.resolve() == output_resolved:
            continue
        result.append(path)
    return sorted(result, key=lambda item: item.relative_to(root).as_posix())


def spdx_id(rel: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9.-]", "-", rel)
    return f"SPDXRef-File-{index}-{clean}"[:240]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    files = iter_files(root, output)
    package_id = "SPDXRef-Package-aldous-tsp"
    file_records = []
    relationships = []
    verification_parts: list[str] = []
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(root).as_posix()
        sha1_digest, sha256_digest = file_digests(path)
        verification_parts.append(sha1_digest)
        identifier = spdx_id(rel, index)
        file_records.append(
            {
                "SPDXID": identifier,
                "fileName": f"./{rel}",
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": sha1_digest},
                    {"algorithm": "SHA256", "checksumValue": sha256_digest},
                ],
                "licenseConcluded": "NOASSERTION",
                "licenseInfoInFiles": ["NOASSERTION"],
                "copyrightText": "NOASSERTION",
                "fileTypes": ["SOURCE"],
            }
        )
        relationships.append(
            {
                "spdxElementId": package_id,
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )
    verification = hashlib.sha1("".join(sorted(verification_parts)).encode()).hexdigest()
    namespace = (
        "https://github.com/MatthiasIrlbeck/subset-selection-TSP/"
        f"sbom/{args.version}/{args.commit}/{verification}"
    )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{args.package_name}-{args.version}-source",
        "documentNamespace": namespace,
        "creationInfo": {
            "created": timestamp(),
            "creators": ["Tool: aldous-tsp-generate-sbom/1"],
        },
        "packages": [
            {
                "name": args.package_name,
                "SPDXID": package_id,
                "versionInfo": args.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "packageVerificationCode": {
                    "packageVerificationCodeValue": verification
                },
                "licenseConcluded": "MIT",
                "licenseDeclared": "MIT",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PERSISTENT-ID",
                        "referenceType": "gitoid",
                        "referenceLocator": f"gitoid:tree:sha1:{args.tree}",
                    },
                    {
                        "referenceCategory": "OTHER",
                        "referenceType": "vcs",
                        "referenceLocator": (
                            "git+https://github.com/MatthiasIrlbeck/"
                            f"subset-selection-TSP@{args.commit}"
                        ),
                    },
                ],
            }
        ],
        "files": file_records,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": package_id,
            },
            *relationships,
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
