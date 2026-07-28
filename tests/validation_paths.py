from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def current_validation_dir(root: Path) -> Path:
    return root / "validation_runs" / "current"


def archived_validation_dirs(root: Path) -> list[Path]:
    archive = root / "validation_archive"
    if not archive.exists():
        return []
    return sorted(
        path / "validation_runs"
        for path in archive.iterdir()
        if path.is_dir() and (path / "validation_runs").is_dir()
    )


def iter_validation_json(
    root: Path,
    *,
    include_current: bool = True,
    include_archive: bool = True,
) -> Iterator[Path]:
    roots: list[Path] = []
    if include_current:
        current = current_validation_dir(root)
        if current.is_dir():
            roots.append(current)
    if include_archive:
        roots.extend(archived_validation_dirs(root))
    for validation_root in roots:
        yield from sorted(validation_root.rglob("*.json"))
