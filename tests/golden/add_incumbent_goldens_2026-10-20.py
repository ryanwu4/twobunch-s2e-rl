"""Freeze the controller incumbents (their surrogates, policies and envs) into tests/golden/ (2026-10-20), on the code as
it stands before the LPS-sharpness surrogate work. Only writes goldens that do not exist yet; existing ones are never
regenerated.

  PYTHONPATH=$PWD/src:$PWD/tests python tests/golden/add_incumbent_goldens_2026-10-20.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compat_artifacts as ca  # noqa: E402

NEW_FLOWS = ["xtcav23e_inj_covB2", "xtcav23e_inj_covB2zs"]
NEW_RUNS = [r for r in ca.POLICY_FLOW if r.startswith("2026-")]


def save(kind, name, arrays):
    out = ca.golden_path(kind, name)
    if out.exists():
        print(f"  exists, kept: {out.relative_to(ca.REPO)}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"  wrote {out.relative_to(ca.REPO)}")


for name in NEW_FLOWS:
    save("flow", name, ca.flow_probe(ca.load_flow(name)))
for run in NEW_RUNS:
    save("policy", run, ca.policy_probe(*ca.load_policy(run)))
    with tempfile.TemporaryDirectory() as tmp:
        save("env", run, ca.env_probe(run, Path(tmp)))
