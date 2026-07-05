"""How low can the driver-witness transverse offset at PENT be pushed?

A small direct optimization (the tracked analog of the FF beta-matching solve): start at the
golden two-bunch working point, vary the offset-relevant knobs, and minimize the PENT
transverse centroid offset with a viability penalty so the optimizer can't "win" by scraping
a bunch. The movers are the differential (driver-vs-witness) lever; kickers are common-mode
and excluded by default. Pass --knobs to widen the actuator set if movers alone plateau.

Reduced particle count is fine here -- the centroid offset is a low-order moment that
converges far faster than emittance. Run wakes-on (default) for a realistic floor; the
transverse wake adds a kick the movers must also cancel (coefficients still provisional).

Usage (bmad-qpad-dev; from twobunch-s2e-rl repo root):
  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.analysis_tools.offset_floor \
      [--npart 20000] [--wakes/--no-wakes] [--knobs movers|movers+strengths|movers+ff] \
      [--maxfev 200] [--span 1.5e-3]
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "32")  # one track at a time -> use the cores

import numpy as np
from scipy.optimize import minimize

from ..datagen.paths import facet2_root
from ..datagen.sweep_params import SWEEP_PARAMS_EXPANDED_EXTRA, SWEEP_PARAMS

BASELINE_CFG = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
MOVERS = ["S1EL_xOffset", "S1EL_yOffset", "S2EL_xOffset", "S2EL_yOffset",
          "S2ER_xOffset", "S2ER_yOffset", "S1ER_xOffset", "S1ER_yOffset"]
STRENGTHS = ["S1ELkG", "S2ELkG", "S3ELkG"]
FF = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]


def golden_of(knob):
    tbl = {**SWEEP_PARAMS, **SWEEP_PARAMS_EXPANDED_EXTRA}
    return tbl[knob][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npart", type=int, default=20000)
    ap.add_argument("--wakes", dest="wakes", action="store_true", default=True)
    ap.add_argument("--no-wakes", dest="wakes", action="store_false")
    ap.add_argument("--knobs", choices=["movers", "movers+strengths", "movers+ff"],
                    default="movers")
    ap.add_argument("--maxfev", type=int, default=200)
    ap.add_argument("--span", type=float, default=1.5e-3, help="mover half-range [m]")
    args = ap.parse_args()

    knobs = {"movers": MOVERS, "movers+strengths": MOVERS + STRENGTHS,
             "movers+ff": MOVERS + FF}[args.knobs]

    import FACET2_S2E as qs
    root = str(facet2_root())
    base = qs.loadConfig("/" + BASELINE_CFG, root)
    tao = qs.initializeTao(
        filePath=root, inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
        csrTF=True, transverseWakes=args.wakes, numMacroParticles=args.npart,
        scratchPath=root + "/tmp/offset_floor", randomizeFileNames=True,
    )

    x0 = np.array([golden_of(k) for k in knobs], dtype=float)
    # mover bounds are golden +/- span; strengths/FF use their expanded-set bounds
    bounds = []
    for k in knobs:
        if k in MOVERS:
            bounds.append((golden_of(k) - args.span, golden_of(k) + args.span))
        else:
            tbl = {**SWEEP_PARAMS, **SWEEP_PARAMS_EXPANDED_EXTRA}
            bounds.append((tbl[k][0], tbl[k][1]))

    history = {"n": 0, "best": np.inf}

    def objective(x):
        merged = {**base, **dict(zip(knobs, x))}
        try:
            qs.setLattice(tao, **merged)
            qs.trackBeam(tao, root, **merged)
            P = qs.getBeamAtElement(tao, "PENT")
        except Exception as e:
            return 1e4
        trans = len(P.x) / args.npart
        if len(np.unique(P.weight)) < 2:          # witness lost -> heavily penalized
            return 5e3 + (1 - trans) * 1e3
        specs = qs.getBeamSpecs(P, targetTwiss="PENT") or {}
        off_um = specs.get("transverseCentroidOffset", np.nan) * 1e6
        if not np.isfinite(off_um):
            return 5e3
        penalty = 0.0 if trans > 0.90 else (0.90 - trans) * 1e4   # keep both bunches alive
        val = off_um + penalty
        history["n"] += 1
        if val < history["best"]:
            history["best"] = val
            print(f"  eval {history['n']:4d}: offset={off_um:7.1f} um  trans={trans:.3f}  *best*",
                  flush=True)
        elif history["n"] % 10 == 0:
            print(f"  eval {history['n']:4d}: offset={off_um:7.1f} um  trans={trans:.3f}",
                  flush=True)
        return val

    print(f"knobs={args.knobs} ({len(knobs)})  npart={args.npart}  wakes={args.wakes}")
    print(f"golden offset = {objective(x0):.1f} um")
    res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxfev": args.maxfev, "xatol": 1e-5, "fatol": 0.5})

    print("\n=== offset floor ===")
    print(f"min offset reached: {res.fun:.1f} um  ({history['n']} tracks)")
    for k, v in zip(knobs, res.x):
        unit = "m" if k in MOVERS else ("kG.m")
        print(f"   {k:14s} {v:+.6g} {unit}   (golden {golden_of(k):+.6g})")


if __name__ == "__main__":
    main()
