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
        prefix_off = tmp / "install-cli-off"
        configure = [
            "cmake", "-S", str(root), "-B", str(build_off),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DALDOUS_TSP_BUILD_CLI=OFF",
            "-DALDOUS_TSP_BUILD_TESTS=OFF",
            "-DBUILD_TESTING=OFF",
        ]
        if shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        run(configure)
        run(["cmake", "--build", str(build_off), "--target", "aldous_tsp_core", "--parallel", "2"])
        run(["cmake", "--install", str(build_off), "--prefix", str(prefix_off)])
        assert_common_install_tree(prefix_off)
        assert not (prefix_off / "include" / "aldous_tsp" / "cli.hpp").exists(), "cli.hpp should not be installed when CLI support is disabled"
        targets_off = (prefix_off / "lib" / "cmake" / "aldous_tsp" / "aldous_tspTargets.cmake").read_text()
        assert "aldous_tsp::core" in targets_off, "core target should be exported"
        assert "aldous_tsp::cli" not in targets_off, "CLI target should not be exported when CLI support is disabled"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
