"""Beam-based two-bunch matching solve at PENT (the valid anchor; design optics does not transfer).

The anchored re-pilot proved the defaults.yml design-optics FF curve does NOT focus the real
campaign beam. The real FF->PENT map must be found by TRACKING the real beam. Built around the
golden two-bunch baseline: vary FF quads (+ BC20 sextupole strengths, the chromatic correction)
-> track the real two-bunch beam -> minimize the WITNESS core-slice BMAG at PENT vs a target
slice-beta (+ a drive term), viability-penalized.

The 15 cm / 50 cm single-target runs showed the witness IS matchable (BMAG -> ~1.0, full
transmission) with FF ~ golden + small sextupole changes. --scan-betas maps the matched curve
over a range of beta* (warm-started continuation), writing the beam-based anchor to CSV.

Usage:
  single:  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.analysis.beam_matching_solve \
               --target-beta 0.15 --npart 20000 --maxfev 400
  scan:    ... --scan-betas 0.50,0.25,0.15,0.10,0.076 --npart 20000 --maxfev 250
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "32")  # one track at a time -> use the cores

import numpy as np
import torch
from scipy.optimize import minimize

from ..datagen.paths import facet2_root, campaign_dir
from ..surrogate.properties import slice_twiss_bmag

BASELINE_CFG = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
FF = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]
SEXT = ["S1ELkG", "S2ELkG", "S3ELkG"]                       # symmetric (R mirrors L)
BOUNDS = {
    "Q5FFkG": (-256., 0.), "Q4FFkG": (-446., 0.), "Q3FFkG": (0., 457.),
    "Q2FFkG": (0., 167.), "Q1FFkG": (-257., 0.), "Q0FFkG": (0., 167.),
    "S1ELkG": (0., 2590.), "S2ELkG": (-21706., 0.), "S3ELkG": (-2625., 0.),
}
COORD = ["x", "y", "z", "px", "py", "pz"]


def _to_torch(pg):
    return torch.tensor(np.stack([getattr(pg, k) for k in COORD], axis=1),
                        dtype=torch.float64).unsqueeze(0)


def _core_bmag(parts, beta0):
    s = slice_twiss_bmag(parts, n_slices=5, beta0=beta0, alpha0=0.0)
    return float(torch.maximum(s["slice_bmag_x_core"], s["slice_bmag_y_core"])), \
        float(s["slice_beta_x_core"]), float(s["slice_beta_y_core"])


def run_solve(qs, tao, base, root, knobs, bounds, tb, x0, args):
    """One target beta* match (tracking-in-the-loop). Returns (best_x, best_obj, metrics)."""
    hist = {"n": 0, "best": np.inf, "x": np.array(x0, float),
            "wb": np.nan, "wby": np.nan, "db": np.nan}

    def objective(x):
        merged = {**base, **dict(zip(knobs, x))}
        try:
            qs.setLattice(tao, **merged)
            qs.trackBeam(tao, root, **merged)
            P = qs.getBeamAtElement(tao, "PENT")
        except Exception:
            return 1e4
        res = qs.getDriverAndWitness(P)
        if res is None or res[1] is None or len(res[1]) < 50 or len(res[0]) < 50:
            return 5e3
        trans = len(P.x) / args.npart
        wb, _, wby = _core_bmag(_to_torch(res[1]), tb)
        db, _, _ = _core_bmag(_to_torch(res[0]), tb)
        val = wb + args.w_drive * db + (0.0 if trans > 0.9 else (0.9 - trans) * 100.0)
        hist["n"] += 1
        if val < hist["best"]:
            hist.update(best=val, x=np.array(x, float), wb=wb, wby=wby, db=db)
            print(f"  [{tb*100:4.0f}cm] eval {hist['n']:4d}: witness BMAG={wb:5.2f} "
                  f"(slice-beta_y={wby*100:6.1f}cm) drive={db:5.2f} trans={trans:.3f} "
                  f"obj={val:.2f} *best*", flush=True)
        elif hist["n"] % 25 == 0:
            print(f"  [{tb*100:4.0f}cm] eval {hist['n']:4d}: obj={val:.2f}", flush=True)
        return val

    print(f"\n=== target {tb*100:.1f} cm ({len(knobs)} knobs) ===", flush=True)
    print(f"  start objective = {objective(np.array(x0, float)):.2f}", flush=True)
    opts = ({"maxfev": args.maxfev, "xatol": 1e-4, "fatol": 0.05}
            if args.method == "Nelder-Mead" else {"maxfev": args.maxfev})
    minimize(objective, np.array(x0, float), method=args.method, bounds=bounds, options=opts)
    print(f"  -> floor obj {hist['best']:.2f}: witness BMAG {hist['wb']:.2f} "
          f"(slice-beta_y {hist['wby']*100:.1f}cm), drive BMAG {hist['db']:.2f}  "
          f"({hist['n']} tracks)", flush=True)
    return hist["x"], hist["best"], {"wb": hist["wb"], "wby": hist["wby"], "db": hist["db"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-beta", type=float, default=0.15, help="witness target slice-beta [m]")
    ap.add_argument("--scan-betas", default="", help="comma list of beta* [m]; enables curve scan")
    ap.add_argument("--knobs", choices=["ff", "ff+sext"], default="ff+sext")
    ap.add_argument("--w-drive", type=float, default=0.3)
    ap.add_argument("--npart", type=int, default=20000)
    ap.add_argument("--wakes", dest="wakes", action="store_true", default=True)
    ap.add_argument("--no-wakes", dest="wakes", action="store_false")
    ap.add_argument("--maxfev", type=int, default=400)
    ap.add_argument("--method", default="Nelder-Mead")
    ap.add_argument("--out", default=None, help="curve CSV (scan mode)")
    args = ap.parse_args()

    import FACET2_S2E as qs
    root = str(facet2_root())
    base = qs.loadConfig("/" + BASELINE_CFG, root)
    tao = qs.initializeTao(
        filePath=root, inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
        csrTF=True, transverseWakes=args.wakes, numMacroParticles=args.npart,
        scratchPath=root + "/tmp/beam_matching", randomizeFileNames=True,
    )
    knobs = FF + (SEXT if args.knobs == "ff+sext" else [])
    bounds = [BOUNDS[k] for k in knobs]

    betas = ([float(b) for b in args.scan_betas.split(",") if b.strip()]
             if args.scan_betas else [args.target_beta])
    betas = sorted(betas, reverse=True)          # start at largest (closest to golden), continue down
    print(f"knobs={args.knobs} npart={args.npart} wakes={args.wakes} | betas(cm)="
          f"{[round(b*100,1) for b in betas]}  (warm-started continuation)")

    x0 = np.array([float(base[k]) for k in knobs])
    results = []
    for tb in betas:
        xopt, fun, m = run_solve(qs, tao, base, root, knobs, bounds, tb, x0, args)
        results.append((tb, xopt, m))
        x0 = xopt                                # warm-start the next (smaller) beta*

    if args.scan_betas:
        import csv
        out = args.out or str(campaign_dir("beam_matching") / "beam_matched_curve.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["beta_m"] + knobs + ["witness_bmag", "witness_beta_y_m", "drive_bmag"])
            for tb, x, m in sorted(results):
                w.writerow([f"{tb:.4f}"] + [f"{v:.4f}" for v in x]
                           + [f"{m['wb']:.3f}", f"{m['wby']:.4f}", f"{m['db']:.3f}"])
        print(f"\nwrote beam-based matched curve -> {out}")
        print("beta_m -> witness BMAG | knobs:")
        for tb, x, m in sorted(results):
            print(f"  {tb*100:5.1f}cm  BMAG {m['wb']:.2f}  "
                  + " ".join(f"{k[:-2]}={v:.1f}" for k, v in zip(knobs, x)))
    else:
        tb, x, m = results[0]
        print("\n=== matching floor ===")
        for k, v in zip(knobs, x):
            print(f"   {k:9s} {v:+.4g}   (golden {float(base[k]):+.4g})")


if __name__ == "__main__":
    main()
