#!/usr/bin/env python3
"""Create reproducible source ZIP and tar.gz archives from one Git tree.

Git's native ZIP backend can emit DOS-style entries without Unix permission
metadata.  This utility uses Git's tar backend as the single authoritative
source (including export-subst expansion), then derives both distributable
formats while preserving normalized Git modes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import zipfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--tar-gz", dest="tar_gz_path", type=Path, required=True)
    return parser.parse_args()


def _normalized_prefix(prefix: str) -> str:
    stripped = prefix.strip("/")
    if not stripped:
        raise ValueError("archive prefix must not be empty")
    return stripped + "/"


def _git_tar(repo: Path, ref: str, prefix: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            "tar.umask=0022",
            "archive",
            "--format=tar",
            f"--prefix={prefix}",
            ref,
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def _write_tar_gz(tar_bytes: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                compressed.write(tar_bytes)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _zip_datetime(timestamp: int | float) -> tuple[int, int, int, int, int, int]:
    value = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    # ZIP stores seconds in two-second units.  Explicit truncation avoids
    # platform-specific rounding behavior.
    second = value.second - (value.second % 2)
    return (value.year, value.month, value.day, value.hour, value.minute, second)


def _zip_info(member: tarfile.TarInfo) -> zipfile.ZipInfo:
    name = member.name
    if member.isdir() and not name.endswith("/"):
        name += "/"
    info = zipfile.ZipInfo(name, _zip_datetime(member.mtime))
    info.create_system = 3  # Unix
    info.extract_version = 20
    info.create_version = 30
    info.comment = b""
    info.extra = b""
    mode = member.mode & 0o7777
    if member.isdir():
        info.external_attr = ((stat.S_IFDIR | mode) << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
    elif member.issym():
        info.external_attr = (stat.S_IFLNK | mode) << 16
        info.compress_type = zipfile.ZIP_STORED
    elif member.isfile():
        info.external_attr = (stat.S_IFREG | mode) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    else:
        raise ValueError(f"unsupported Git archive entry type: {member.name}")
    return info


def _write_zip(tar_bytes: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as source:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=True,
                strict_timestamps=True,
            ) as destination:
                for member in source.getmembers():
                    info = _zip_info(member)
                    if member.isdir():
                        payload = b""
                    elif member.issym():
                        payload = member.linkname.encode("utf-8")
                    else:
                        extracted = source.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"could not read archive member: {member.name}")
                        payload = extracted.read()
                    destination.writestr(
                        info,
                        payload,
                        compress_type=info.compress_type,
                        compresslevel=9,
                    )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    prefix = _normalized_prefix(args.prefix)
    tar_bytes = _git_tar(repo, args.ref, prefix)
    _write_tar_gz(tar_bytes, args.tar_gz_path.resolve())
    _write_zip(tar_bytes, args.zip_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
