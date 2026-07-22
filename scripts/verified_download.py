#!/usr/bin/env python3
"""Download a locked HTTPS artifact and verify its independently supplied SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import urllib.parse
import urllib.request

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_source(lock_path: Path, name: str) -> dict[str, object]:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported oracle source-lock schema")
    sources = data.get("sources")
    if not isinstance(sources, dict) or name not in sources:
        raise ValueError(f"source {name!r} is not present in {lock_path}")
    source = sources[name]
    if not isinstance(source, dict):
        raise ValueError(f"source {name!r} is malformed")
    url = source.get("url")
    max_bytes = source.get("max_bytes")
    if not isinstance(url, str) or urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("locked oracle URL must use HTTPS")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("locked oracle max_bytes must be a positive integer")
    return source


def download(source: dict[str, object], expected: str, output: Path) -> dict[str, object]:
    expected = expected.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    url = str(source["url"])
    max_bytes = int(source["max_bytes"])
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "aldous-tsp-verified-download/1"},
    )
    context = ssl.create_default_context()
    digest = hashlib.sha256()
    total = 0
    temp_name: str | None = None
    try:
        with urllib.request.urlopen(request, context=context, timeout=60) as response:
            final_url = response.geturl()
            if urllib.parse.urlparse(final_url).scheme != "https":
                raise RuntimeError(f"download redirected to a non-HTTPS URL: {final_url}")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise RuntimeError(
                    f"declared artifact size {declared} exceeds locked limit {max_bytes}"
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
            ) as temp:
                temp_name = temp.name
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(
                            f"artifact exceeded locked limit of {max_bytes} bytes"
                        )
                    digest.update(chunk)
                    temp.write(chunk)
                temp.flush()
                os.fsync(temp.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: expected {expected}, observed {actual}"
            )
        os.replace(temp_name, output)
        temp_name = None
        return {
            "source": source,
            "output": str(output),
            "bytes": total,
            "sha256": actual,
        }
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    source = load_source(args.lock, args.source)
    record = download(source, args.expected_sha256, args.output)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
