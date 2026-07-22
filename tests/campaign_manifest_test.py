#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def expect_failure(command: list[str], needle: str | None = None) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0, completed.stdout
    if needle:
        text = completed.stdout + completed.stderr
        assert needle in text, text


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: campaign_manifest_test.py <repo-root> <aldous-tsp>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    executable = str(Path(sys.argv[2]).resolve())
    sys.path.insert(0, str(root / "scripts"))
    from campaign_manifest import (  # type: ignore
        ManifestError,
        load_or_create_manifest,
        make_entry,
        new_manifest,
        require_complete,
        update_entry,
        validate_result,
    )

    with tempfile.TemporaryDirectory(prefix="aldous-manifest-") as temporary:
        directory = Path(temporary)
        output = directory / "cell.json"
        argv = [
            "--N", "24", "--instances", "1", "--threads", "1",
            "--p-values", "1", "--sa-iters", "0",
            "--tsp-candidate-starts", "1", "--tsp-restarts", "1",
            "--tsp-ils", "0", "--include-instance-rows",
            "--campaign-id", "manifest-test", "--campaign-shard", "2",
            "--replicate-offset", "7", "--point-seed", "100",
            "--search-seed", "200", "--solver-policy-id", "publication",
            "--fidelity-level", "strong", "--output-durability", "full",
            "--output", str(output),
        ]
        entry = make_entry(
            cell_id="n24-p1", output=output, executable=executable, argv=argv,
            expected_n=24, expected_instances=1, expected_p_values=[1.0],
        )
        completed = subprocess.run([executable, *argv], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        validated = validate_result(output, entry)
        assert len(validated["sha256"]) == 64

        # The exact-cell fingerprint changes with problem identity, while the
        # method fingerprint changes only when a quality-affecting control does.
        other_output = directory / "other.json"
        other_argv = list(argv)
        other_argv[other_argv.index("--output") + 1] = str(other_output)
        n_changed = make_entry(
            cell_id="other-output", output=other_output, executable=executable,
            argv=other_argv, expected_n=24, expected_instances=1,
            expected_p_values=[1.0],
        )
        # Re-probe with only the output changed: exact config changes, method does not.
        assert n_changed["expected"]["configuration_fingerprint"] != entry["expected"]["configuration_fingerprint"]
        assert n_changed["expected"]["method_fingerprint"] == entry["expected"]["method_fingerprint"]
        quality_argv = list(argv)
        quality_argv[quality_argv.index("--tsp-ils") + 1] = "1"
        quality_argv[quality_argv.index("--output") + 1] = str(directory / "quality.json")
        quality = make_entry(
            cell_id="quality", output=directory / "quality.json",
            executable=executable, argv=quality_argv, expected_n=24,
            expected_instances=1, expected_p_values=[1.0],
        )
        assert quality["expected"]["method_fingerprint"] != entry["expected"]["method_fingerprint"]

        bad = copy.deepcopy(entry)
        bad["expected"]["configuration_fingerprint"] = "0" * 64
        try:
            validate_result(output, bad)
            raise AssertionError("configuration mismatch was accepted")
        except ManifestError:
            pass

        receipt_path = Path(str(output) + ".receipt")
        receipt = receipt_path.read_text(encoding="utf-8")
        tampered = json.loads(receipt)
        tampered["result_sha256"] = "f" * 64
        receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            validate_result(output, entry)
            raise AssertionError("receipt mismatch was accepted")
        except ManifestError:
            pass
        receipt_path.write_text(receipt, encoding="utf-8")

        manifest_path = directory / "manifest.json"
        campaign = {"kind": "test", "campaign_id": "manifest-test"}
        manifest = load_or_create_manifest(manifest_path, campaign, [entry])
        update_entry(
            manifest_path, manifest, entry["cell_id"], status="complete",
            result_sha256=validated["sha256"],
        )
        require_complete(manifest, allow_partial=False)
        changed = copy.deepcopy(entry)
        changed["argv"] = [*changed["argv"], "--verbose"]
        try:
            load_or_create_manifest(manifest_path, campaign, [changed])
            raise AssertionError("manifest command drift was accepted")
        except ManifestError:
            pass
        planned_entry = copy.deepcopy(entry)
        planned_entry["status"] = "planned"
        planned_entry["result_sha256"] = None
        incomplete = new_manifest(campaign, [planned_entry])
        try:
            require_complete(incomplete, allow_partial=False)
            raise AssertionError("incomplete campaign was accepted")
        except ManifestError:
            pass
        require_complete(incomplete, allow_partial=True)

        full_study = str(root / "scripts" / "run_full_study.py")
        common = [
            sys.executable, full_study, "--exe", executable,
            "--out-dir", str(directory / "study"), "--instances", "1",
            "--threads", "1", "--ps", "0.5", "--ks", "12",
            "--max-n", "30", "--dry-run",
        ]
        expect_failure([*common, "--stage", "campaign", "--search-policy", "heldout-quality"], "sa-iters-per-n")
        expect_failure([
            *common, "--stage", "campaign", "--search-policy", "legacy-balanced",
            "--sa-iters-per-n", "1", "--allow-oracle-fallback",
        ], "heldout-quality")
        expect_failure([
            *common, "--stage", "campaign", "--search-policy", "heldout-quality",
            "--sa-iters-per-n", "1",
        ], "will not silently fall back")
        completed = subprocess.run([
            *common, "--stage", "campaign", "--search-policy", "heldout-quality",
            "--sa-iters-per-n", "1", "--allow-oracle-fallback",
        ], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        study_manifest = directory / "study" / "campaign" / "campaign-manifest.json"
        assert study_manifest.is_file()

        torus = str(root / "scripts" / "run_torus_campaign.py")
        torus_common = [
            sys.executable, torus, "--exe", executable, "--lkh-path", "missing-lkh",
            "--out-dir", str(directory / "torus"), "--ps", "0.5", "--ks", "12",
            "--instances", "1", "--dry-run",
        ]
        expect_failure([*torus_common, "--search-policy", "heldout-quality"], "sa-iters-per-n")
        expect_failure([
            *torus_common, "--search-policy", "legacy-balanced",
            "--sa-iters-per-n", "1",
        ], "heldout-quality")

    print("campaign manifest contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
