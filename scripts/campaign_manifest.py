#!/usr/bin/env python3
"""Fail-closed campaign manifests for Aldous subset-TSP studies.

A manifest stores the exact argv, resolved-configuration fingerprint, method
fingerprint, expected problem cell, and expected output path for every planned
batch. Existing outputs are resumable only when the complete result and its
adjacent timing receipt match that entry exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


class ManifestError(RuntimeError):
    """Raised when a campaign plan or completed output violates its contract."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            # Cleanup is idempotent when the temporary file was never created.
            pass
        raise


def probe_fingerprints(executable: str, argv: Iterable[str]) -> dict[str, str]:
    command = [executable, *argv, "--dry-run", "--dump-config"]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ManifestError(
            f"configuration probe failed ({completed.returncode}): {detail}"
        )
    prefix = "ALDOUS_TSP_FINGERPRINTS "
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ManifestError("configuration probe did not emit exactly one fingerprint record")
    try:
        record = json.loads(lines[0][len(prefix):])
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid fingerprint record: {exc}") from exc
    for key in ("configuration_fingerprint", "method_fingerprint"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ManifestError(f"probe emitted invalid {key}")
    return record


def make_entry(
    *,
    cell_id: str,
    output: Path,
    executable: str,
    argv: list[str],
    expected_n: int,
    expected_instances: int,
    expected_p_values: list[float],
    require_oracle: str | None = None,
) -> dict[str, Any]:
    fingerprints = probe_fingerprints(executable, argv)
    return {
        "cell_id": cell_id,
        "output": str(output),
        "executable": executable,
        "argv": list(argv),
        "expected": {
            "N": expected_n,
            "instances": expected_instances,
            "p_values": expected_p_values,
            "required_oracle": require_oracle,
            **fingerprints,
        },
        "status": "planned",
        "result_sha256": None,
    }


def _float_lists_equal(left: object, right: list[float]) -> bool:
    if not isinstance(left, list) or len(left) != len(right):
        return False
    try:
        return all(float(a) == float(b) for a, b in zip(left, right))
    except (TypeError, ValueError):
        return False


def validate_result(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"missing result: {path}")
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse result {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestError(f"result {path} is not a JSON object")
    expected = entry["expected"]
    checks = {
        "schema_version": isinstance(document.get("schema_version"), int)
            and int(document["schema_version"]) >= 16,
        "N": document.get("N") == expected["N"],
        "done": document.get("done") == expected["instances"],
        "target": document.get("target") == expected["instances"],
        "p_values": _float_lists_equal(document.get("p_values"), expected["p_values"]),
    }
    metadata = document.get("campaign_metadata") or {}
    checks["configuration_fingerprint"] = (
        metadata.get("configuration_fingerprint")
        == expected["configuration_fingerprint"]
    )
    checks["method_fingerprint"] = (
        metadata.get("method_fingerprint") == expected["method_fingerprint"]
    )
    required_oracle = expected.get("required_oracle")
    if required_oracle:
        checks["required_oracle"] = (
            document.get("config", {}).get("oracle_resolved") == required_oracle
        )
        checks["oracle_hash"] = bool(
            document.get("config", {}).get("oracle_exec_sha256")
            not in (None, "", "unknown")
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ManifestError(
            f"result {path} violates manifest fields: {', '.join(failed)}"
        )

    receipt_path = Path(str(path) + ".receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"missing or invalid timing receipt {receipt_path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if receipt.get("result_sha256") != digest:
        raise ManifestError(f"receipt digest does not match result {path}")
    if receipt.get("commit_state") != "fully-durable":
        raise ManifestError(f"result {path} is not recorded as fully durable")
    return {"document": document, "sha256": digest, "receipt": receipt}


def planned_payload(
    campaign: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    immutable_entries = [
        {key: value for key, value in entry.items() if key not in {"status", "result_sha256"}}
        for entry in entries
    ]
    return {"campaign": campaign, "entries": immutable_entries}


def new_manifest(campaign: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    payload = planned_payload(campaign, entries)
    return {
        "manifest_version": 1,
        "manifest_fingerprint": canonical_sha256(payload),
        "campaign": campaign,
        "entries": entries,
        "complete": False,
    }


def load_or_create_manifest(
    path: Path, campaign: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    proposed = new_manifest(campaign, entries)
    if not path.exists():
        atomic_write_json(path, proposed)
        return proposed
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read existing manifest {path}: {exc}") from exc
    if existing.get("manifest_fingerprint") != proposed["manifest_fingerprint"]:
        raise ManifestError(
            f"existing manifest {path} does not match the requested campaign plan"
        )
    # Preserve completion states but never permit command or expectation drift.
    proposed_by_id = {entry["cell_id"]: entry for entry in proposed["entries"]}
    for old in existing.get("entries", []):
        current = proposed_by_id.get(old.get("cell_id"))
        if current is not None:
            current["status"] = old.get("status", "planned")
            current["result_sha256"] = old.get("result_sha256")
    proposed["complete"] = bool(existing.get("complete", False))
    atomic_write_json(path, proposed)
    return proposed


def update_entry(
    manifest_path: Path,
    manifest: dict[str, Any],
    cell_id: str,
    *,
    status: str,
    result_sha256: str | None = None,
) -> None:
    matches = [entry for entry in manifest["entries"] if entry["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ManifestError(f"manifest has no unique cell {cell_id}")
    matches[0]["status"] = status
    matches[0]["result_sha256"] = result_sha256
    manifest["complete"] = all(
        entry.get("status") == "complete" for entry in manifest["entries"]
    )
    atomic_write_json(manifest_path, manifest)


def require_complete(manifest: dict[str, Any], *, allow_partial: bool) -> None:
    incomplete = [
        entry["cell_id"] for entry in manifest["entries"]
        if entry.get("status") != "complete"
    ]
    if incomplete and not allow_partial:
        raise ManifestError(
            "campaign incomplete: " + ", ".join(incomplete[:12])
            + (" ..." if len(incomplete) > 12 else "")
        )
