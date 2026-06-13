"""Path resolution for the datagen stage.

The sweep needs the FACET2-S2E checkout (Bmad lattice / beam / setLattice configs). Its
root is decoupled from this repo's location so the driver can live outside the FACET2-S2E
tree: it is taken from $FACET2_S2E_ROOT, else derived from the installed FACET2_S2E package
(editable install => the package lives at <root>/src/FACET2_S2E, so the root is parents[2]).
"""
import os
from pathlib import Path


def facet2_root() -> Path:
    """Root of the FACET2-S2E checkout (where setLattice_configs/, beams/, bmad/ live)."""
    env = os.environ.get("FACET2_S2E_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"FACET2_S2E_ROOT={root} is not a directory")
        return root
    import FACET2_S2E  # noqa: PLC0415 — deferred so analysis-only use doesn't require it

    return Path(FACET2_S2E.__file__).resolve().parents[2]


def repo_root() -> Path:
    """Root of this repo (.../src/twobunch_s2e_rl/datagen/paths.py -> parents[3])."""
    return Path(__file__).resolve().parents[3]
