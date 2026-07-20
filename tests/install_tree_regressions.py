#!/usr/bin/env python3
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def assert_common_install_tree(prefix: Path) -> None:
    include_dir = prefix / "include" / "aldous_tsp"
    assert (include_dir / "version.hpp").exists(), f"missing generated version.hpp under {include_dir}"
    assert not (include_dir / "version.hpp.in").exists(), "version.hpp.in template must not be installed"
    assert (include_dir / "solver.hpp").exists(), "core public headers must be installed"
    assert (include_dir / "exact_subset.hpp").exists(), "exact subset API must be installed"
    assert (include_dir / "restart.hpp").exists(), "typed restart API must be installed"
    assert (include_dir / "restart_kinds.def").exists(), "restart-kind metadata must be installed"
    assert (prefix / "lib" / "cmake" / "aldous_tsp" / "aldous_tspTargets.cmake").exists(), "exported CMake targets missing"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: install_tree_regressions.py <repo_root> <build_dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    build_dir = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="aldous_install_tree_") as td:
        tmp = Path(td)
        prefix_on = tmp / "install-cli-on"
        run(["cmake", "--install", str(build_dir), "--prefix", str(prefix_on)])
        assert_common_install_tree(prefix_on)
        assert (prefix_on / "include" / "aldous_tsp" / "cli.hpp").exists(), "cli.hpp should be installed when CLI support is built"
        targets_on = (prefix_on / "lib" / "cmake" / "aldous_tsp" / "aldous_tspTargets.cmake").read_text()
        assert "aldous_tsp::cli" in targets_on, "CLI target should be exported when CLI support is built"

        build_off = tmp / "build-cli-off"
        configure = [
            "cmake", "-S", str(root), "-B", str(build_off),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DALDOUS_TSP_BUILD_CLI=OFF",
            "-DALDOUS_TSP_BUILD_TESTS=OFF",
            "-DALDOUS_TSP_LOW_MEMORY_BUILD=ON",
            "-DBUILD_TESTING=OFF",
        ]
        if shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        run(configure)

        # Export/install rules are generated at configure time. Inspecting those
        # files is sufficient to verify the CLI-disabled public surface and
        # avoids recompiling the entire core solely to copy an archive that was
        # already exercised by the CLI-enabled install above.
        target_files = list((build_off / "CMakeFiles" / "Export").glob("**/aldous_tspTargets.cmake"))
        assert len(target_files) == 1, f"expected one generated target export, found {target_files}"
        targets_off = target_files[0].read_text()
        assert "aldous_tsp::core" in targets_off, "core target should be exported"
        assert "aldous_tsp::cli" not in targets_off, "CLI target should not be exported when CLI support is disabled"

        install_script = (build_off / "cmake_install.cmake").read_text()
        assert "cli.hpp" not in install_script, "cli.hpp should not be installed when CLI support is disabled"
        assert "aldous_tsp_cli" not in install_script, "the CLI library should have no install rule when disabled"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
