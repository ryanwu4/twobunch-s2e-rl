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


# --- generated-output locations (single source of truth; see the results/ reorg) -----------
# Reusable tools (analysis_tools/, surrogate/) write here via these helpers. One-off report
# scripts live in results/<study>/ and instead write beside themselves (Path(__file__).parent).

def results_dir() -> Path:
    """Root of generated outputs."""
    d = repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def campaign_dir(name: str) -> Path:
    """Per-campaign output folder, e.g. results/tightbox_v2_full/."""
    d = results_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def surrogate_dir(model: str) -> Path:
    """Per-model surrogate output folder, e.g. results/surrogate/combined_ft/."""
    d = results_dir() / "surrogate" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def rl_dir(run: str) -> Path:
    """Per-run RL eval/analysis output folder, e.g. results/rl/bptt_dr/ (run = logdir basename)."""
    d = results_dir() / "rl" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def rl_shared_dir() -> Path:
    """Cross-run / shared RL area (reward-norm caches, compare, particle_study): results/rl/_shared/."""
    d = results_dir() / "rl" / "_shared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir() -> Path:
    """Non-figure data outputs (dataset.pkl/csv, etc.)."""
    d = results_dir() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d
