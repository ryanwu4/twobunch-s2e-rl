"""Manifest for the XTCAVF+Cheetah validation-gate campaign (plan phase 2).

Bmad tracks whose beams are saved at BOTH XTCAVF and PENT, so the Cheetah XTCAVF->PENT segment can
be checked against Bmad where the delta controller will operate. Two blocks:

- "controller" (default 120): L1/L2 phase uniform in the controller sub-box (golden +-2 / +-2.5
  deg), every other xtcav23 knob + Q2/Q1/Q0FF Gaussian around golden at readback-error scale.
- "wide" (default 40): plain LHS over the full xtcav23 box, Q2/Q1/Q0FF uniform +-5% of golden,
  for generality beyond the controller region.

Writes data/<name>/manifest.json; run_sweep reuses an existing manifest as-is, so the campaign then
runs with the normal driver:

  python scripts/make_valgate_manifest.py --name xtcav23_valgate_2026-10-14
  python -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/xtcav23_valgate.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from twobunch_s2e_rl.datagen.sweep_params import EXPANDED_PARAMS, XTCAV23_PARAMS

REPO = Path(__file__).resolve().parents[1]

SEGMENT_QUADS = ["Q2FFkG", "Q1FFkG", "Q0FFkG"]           # downstream of XTCAVF (Cheetah segment)
CONTROLS = {"L1PhaseSet": 2.0, "L2PhaseSet": 2.5}        # controller sub-box half-widths [deg]

# readback-error-scale 1-sigma per knob family (plan: "Other knobs (hidden)")
SECTION_GAIN_EV = {"L1EnergyOffset": 210e6, "L2EnergyOffset": 4.165e9, "L3EnergyOffset": 5.5e9}


def readback_sigma(key, golden):
    if key in SECTION_GAIN_EV:
        return 1e-3 * SECTION_GAIN_EV[key]                # 0.1% of the section energy gain
    if "Offset" in key:
        return 1e-4                                       # movers: 0.1 mm
    if key.startswith(("XC", "YC")):
        return 5e-4                                       # FF kickers: 5e-4 kG.m (~10% of golden)
    return 0.01 * abs(golden)                             # magnets: 1%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xtcav23_valgate_2026-10-14")
    ap.add_argument("--n-controller", type=int, default=120)
    ap.add_argument("--n-wide", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    keys = list(XTCAV23_PARAMS)
    golden = {k: XTCAV23_PARAMS[k][2] for k in keys}
    golden.update({q: EXPANDED_PARAMS[q][2] for q in SEGMENT_QUADS})
    lo = {k: XTCAV23_PARAMS[k][0] for k in keys}
    hi = {k: XTCAV23_PARAMS[k][1] for k in keys}

    rows = []
    for _ in range(args.n_controller):
        knobs = {}
        for k in keys + SEGMENT_QUADS:
            if k in CONTROLS:
                v = golden[k] + rng.uniform(-CONTROLS[k], CONTROLS[k])
            else:
                v = golden[k] + rng.normal(0.0, readback_sigma(k, golden[k]))
            if k in lo:
                v = float(np.clip(v, lo[k], hi[k]))
            knobs[k] = float(v)
        rows.append({"knobs": knobs, "block": "controller"})

    lhs = qmc.LatinHypercube(d=len(keys) + 3, seed=args.seed).random(args.n_wide)
    for u in lhs:
        knobs = {k: float(lo[k] + u[i] * (hi[k] - lo[k])) for i, k in enumerate(keys)}
        for j, q in enumerate(SEGMENT_QUADS):
            knobs[q] = float(golden[q] * (0.95 + 0.10 * u[len(keys) + j]))
        rows.append({"knobs": knobs, "block": "wide"})

    manifest = [{"idx": i, "is_baseline_repeat": False, **r} for i, r in enumerate(rows)]
    for j in range(2):                                    # baseline repeats: noise floor
        manifest.append({"idx": len(rows) + j, "knobs": dict(golden), "is_baseline_repeat": True,
                         "block": "baseline"})

    out = REPO / "data" / args.name / "manifest.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} ({len(manifest)} rows)")


if __name__ == "__main__":
    main()
