"""ONE-TIME backfill: add drive/witness sigma_z to this run's validate_goal*um.json comparison.

This run was validated in Bmad BEFORE sigma_z was made a first-class pipeline metric
(reward._report_keys + transfer_setpoints._METRICS + validate_bmad._beam_sigma_z). Rather than
re-track the beams in Bmad (~min/goal), this reproduces EXACTLY what the edited validate_bmad.py
would have written, from the already-saved PENT beams:
  - surrogate sigma_z : setpoints_goal*um.json -> surrogate_metrics.{drive,witness}_sigma_z_um
                        (populated by the re-run transfer_setpoints; the same source validate_bmad reads)
  - Bmad sigma_z      : full std(z) of the saved sample_*_PENT.h5 (validate_bmad._beam_sigma_z)
It injects both into surrogate_metrics / bmad_metrics / comparison of each validate JSON and rewrites
validation_summary.csv. Future runs get sigma_z natively and will NOT need this.

Usage: PYTHONPATH=$PWD/src python results/rl/bptt_gc_sigmaz/bmad_validation/backfill_sigma_z.py
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
from pmd_beamphysics import ParticleGroup

HERE = Path(__file__).resolve().parent
SETP = HERE.parent / "setpoints"
SIGZ_KEYS = ("drive_sigma_z_um", "witness_sigma_z_um")


def _beam_sigma_z_um(h5):
    P = ParticleGroup(str(h5)); w = np.unique(P.weight)
    d = float(P[P.weight == w[-1]].z.std()) * 1e6
    wi = float(P[P.weight == w[0]].z.std()) * 1e6 if len(w) >= 2 else float("nan")
    return {"drive_sigma_z_um": d, "witness_sigma_z_um": wi}


def main():
    files = sorted(glob.glob(str(HERE / "validate_goal*um.json")))
    if not files:
        raise SystemExit(f"no validate_goal*um.json in {HERE}")
    for fp in files:
        rec = json.load(open(fp))
        g = int(round(rec["target_um"]))
        sp = json.load(open(SETP / f"setpoints_goal{g}um.json"))["surrogate_metrics"]
        bmad_sz = _beam_sigma_z_um(HERE / f"sample_{g:05d}_PENT.h5")
        for k in SIGZ_KEYS:
            s, b = float(sp[k]), float(bmad_sz[k])
            rec["surrogate_metrics"][k] = s
            rec["bmad_metrics"][k] = b
            rec["comparison"][k] = {"surrogate": s, "bmad": b, "abs_diff": b - s,
                                    "pct_diff": (100.0 * (b - s) / s) if s not in (0.0,) and np.isfinite(b) else None}
        with open(fp, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"goal {g:3d} um  sigma_z surr/bmad  drive {sp['drive_sigma_z_um']:.1f}/{bmad_sz['drive_sigma_z_um']:.1f}"
              f"  witness {sp['witness_sigma_z_um']:.1f}/{bmad_sz['witness_sigma_z_um']:.1f} µm")

    # rewrite the summary CSV from the (now sigma_z-complete) comparison blocks
    recs = [json.load(open(f)) for f in files]
    with open(HERE / "validation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_um", "metric", "surrogate", "bmad", "abs_diff", "pct_diff"])
        for rec in recs:
            for k, c in rec["comparison"].items():
                w.writerow([rec["target_um"], k, c["surrogate"], c["bmad"], c["abs_diff"], c["pct_diff"]])
    print(f"\nupdated {len(files)} validate JSONs + validation_summary.csv with sigma_z")


if __name__ == "__main__":
    main()
