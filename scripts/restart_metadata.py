"""Shared restart-kind metadata for repository analysis scripts.

The authoritative codes and labels live in the public C++ definition file
``include/aldous_tsp/restart_kinds.def``. Keeping Python consumers pointed at
that file prevents a newly added solver kind from being silently excluded by
analysis defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

_KIND_RE = re.compile(
    r'^\s*ALDOUS_TSP_RESTART_KIND\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,'
    r'\s*([0-9]+)\s*,\s*"([^"]+)"\s*\)\s*$'
)


def load_restart_kinds(root: Path | None = None) -> dict[int, str]:
    repo_root = root if root is not None else Path(__file__).resolve().parents[1]
    definition = repo_root / "include" / "aldous_tsp" / "restart_kinds.def"
    kinds: dict[int, str] = {}
    for line_number, line in enumerate(definition.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = _KIND_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"invalid restart kind definition at {definition}:{line_number}")
        code = int(match.group(2))
        label = match.group(3)
        if code in kinds:
            raise RuntimeError(f"duplicate restart kind code {code} in {definition}")
        kinds[code] = label

    expected = list(range(len(kinds)))
    if sorted(kinds) != expected:
        raise RuntimeError(
            f"restart kind codes in {definition} must be contiguous from zero: "
            f"got {sorted(kinds)}"
        )
    return kinds


KIND_NAMES = load_restart_kinds()
DEFAULT_KINDS = tuple(KIND_NAMES)

# Stable restart-sweep codes serialized by RestartSweep in restart.hpp. Old
# result files that predate restart_sweeps are interpreted as primary-only.
SWEEP_NAMES = {0: "primary", 1: "secondary"}
DEFAULT_SWEEPS = tuple(SWEEP_NAMES)

# Stable restart-controller roles serialized by RestartRole in restart.hpp.
# Endpoint fitting defaults to independent diagnostic draws only. Files from
# before role metadata are marked unknown (-1) and remain eligible so historical
# campaigns do not silently disappear; their mixture cannot be decontaminated.
ROLE_NAMES = {
    0: "independent-diagnostic",
    1: "continuation",
    2: "elite-kick",
    3: "anytime",
    4: "raced-production",
}
DEFAULT_ROLES = (0,)
