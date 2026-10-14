"""Freeze the current behaviour of every existing artifact into tests/golden/*.npz.

Run ONCE on the pre-refactor code (commit 6a58ec9 + phase-0 test files), never regenerate to make a
failing compatibility test pass — a diff is a backward-compat break to investigate.

  PYTHONPATH=$PWD/src:$PWD/tests python tests/golden/make_goldens.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compat_artifacts as ca  # noqa: E402


def save(kind, name, arrays):
    out = ca.golden_path(kind, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"  wrote {out.relative_to(ca.REPO)}")


def main():
    for name, rel in ca.CKPTS.items():
        if ca.have(rel):
            save("flow", name, ca.flow_probe(ca.load_flow(name)))
    for run in ca.POLICY_FLOW:
        if ca.have(f"logs/{run}/best_policy.pt"):
            save("policy", run, ca.policy_probe(*ca.load_policy(run)))
    for run, flow in ca.POLICY_FLOW.items():
        if ca.have(f"logs/{run}/cfg.yaml", ca.CKPTS[flow]):
            with tempfile.TemporaryDirectory() as tmp:
                save("env", run, ca.env_probe(run, Path(tmp)))
    for name, rel in ca.SAMPLES.items():
        if ca.have(rel):
            save("sample", name, ca.sample_probe(name))


if __name__ == "__main__":
    main()
