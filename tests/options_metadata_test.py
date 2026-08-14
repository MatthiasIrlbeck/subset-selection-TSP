#!/usr/bin/env python3
"""Cross-check every generated configuration surface against options.json."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: options_metadata_test.py <repo-root> <aldous_tsp-exe>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    exe = Path(sys.argv[2]).resolve()
    metadata = json.loads((root / "config/options.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "schema/results.schema.json").read_text(encoding="utf-8"))

    subprocess.run(
        [sys.executable, str(root / "scripts/generate_options.py"), "--check"],
        cwd=root,
        check=True,
    )

    options = metadata["options"]
    expected_json = {
        option["json_key"]
        for option in options
        if option.get("json_key")
        and option.get("emit_json", True)
        and (option.get("declare", True) or option.get("json_expr"))
    }
    config = schema["$defs"]["config"]
    actual_properties = set(config["properties"])
    actual_required = set(config["required"])
    if actual_properties != expected_json:
        fail(
            "schema config properties differ from metadata: "
            f"missing={sorted(expected_json - actual_properties)}, "
            f"extra={sorted(actual_properties - expected_json)}"
        )
    expected_required = {
        option["json_key"]
        for option in options
        if option.get("json_key")
        and option.get("required", True)
        and (
            (option.get("emit_json", True)
             and (option.get("declare", True) or option.get("json_expr")))
            or (not option.get("declare", True) and option.get("constraints"))
        )
    }
    if actual_required != expected_required:
        fail(
            "schema config required keys differ from metadata: "
            f"missing={sorted(expected_required - actual_required)}, "
            f"extra={sorted(actual_required - expected_required)}"
        )
    if config.get("additionalProperties") is not False:
        fail("generated config schema must reject unknown properties")
    if schema["properties"]["schema_version"].get("const") != metadata["schema_version"]:
        fail("schema version differs from option metadata")

    struct_paths = {
        "run": root / "include/aldous_tsp/generated/run_options_fields.inc",
        "solver": root / "include/aldous_tsp/generated/solver_options_fields.inc",
        "oracle": root / "include/aldous_tsp/generated/external_oracle_fields.inc",
    }
    for struct, path in struct_paths.items():
        text = path.read_text(encoding="utf-8")
        expected_members = {
            option["member"]
            for option in options
            if option.get("declare", True) and option.get("struct") == struct
        }
        actual_members = set(re.findall(r"^\s+[^/\n;]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", text, re.MULTILINE))
        if actual_members != expected_members:
            fail(
                f"generated {struct} fields differ: "
                f"missing={sorted(expected_members - actual_members)}, "
                f"extra={sorted(actual_members - expected_members)}"
            )

    parser_text = (root / "src/generated_options_cli.cpp").read_text(encoding="utf-8")
    config_text = (root / "src/generated_options_core.cpp").read_text(encoding="utf-8")
    docs_text = (root / "docs/generated/options.md").read_text(encoding="utf-8")
    help_result = subprocess.run(
        [str(exe), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help_text = help_result.stdout
    for option in options:
        cli = option.get("cli")
        if cli:
            for flag in [cli, *option.get("aliases", [])]:
                if json.dumps(flag) not in parser_text:
                    fail(f"generated parser does not contain {flag}")
            if cli not in help_text:
                fail(f"generated help does not contain {cli}")
            if f"`{cli}`" not in docs_text:
                fail(f"generated option reference does not contain {cli}")
        key = option.get("json_key")
        if key and option.get("emit_json", True) and (
            option.get("declare", True) or option.get("json_expr")
        ):
            if f'\\"{key}\\"' not in config_text:
                fail(f"generated JSON writer does not contain {key}")

    print(
        f"option metadata parity passed: {len(options)} options, "
        f"{len(expected_json)} JSON fields"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
