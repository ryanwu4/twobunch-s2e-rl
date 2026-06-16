"""Persistent Bmad/Tao worker for Bmad-in-the-loop RL eval.

Run by the **bmad-qpad-dev** python (FACET2_S2E / pytao only import there) -- NOT importable in
the slac-rl torch env. It initializes one persistent Tao, then reads JSON knob requests on
stdin, tracks L0AFEND->PENT, and returns the drive/witness PENT clouds + charge-based surviving
fractions as an npz (path emitted on stdout behind a sentinel, so Tao's stdout noise is ignored).

Tracking reuses run_sweep's exact calls (setLattice / trackBeam / getBeamAtElement /
getDriverAndWitness) and preprocess's particle extraction (_stack / _subsample), so the clouds
match the campaign convention the surrogate was trained on (x,y,z,px,py,pz; z = -c*dt; drive =
higher-weight subset). The slac-rl side computes the observables (per_bunch/inter_bunch).

Protocol (line-based on stdin/stdout):
  <- "<<<READY>>>"                         once Tao is up
  -> {"knobs": {<PARAM_KEYS>: <physical>}} one request per line  (or "STOP")
  <- "<<<RESULT>>>:<path-to-npz>"          per request
npz keys: drive (nd,6) f32, witness (nw,6) f32, T_drive, T_witness (f32, charge/full clipped),
n_drive, n_witness (i32), ok (bool), error (str, on failure).
"""
import os

# Match run_sweep: pin threads before pytao loads (avoid OpenMP oversubscription).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import sys
import tempfile

import numpy as np

from ..datagen.paths import facet2_root
from ..datagen.sweep_params import PARAM_KEYS
from ..surrogate.preprocess import _stack, _subsample

READY = "<<<READY>>>"
RESULT = "<<<RESULT>>>:"


def _extract(qs, P, p, drive_full_nc, witness_full_nc, rng):
    """PENT ParticleGroup -> drive/witness (<=p, 6) clouds + charge-based survival fractions."""
    w = np.unique(P.weight)
    if len(w) >= 2:
        PD, PW = qs.getDriverAndWitness(P)        # canonical split (driver = higher weight)
    else:
        PD, PW = (P, None)                         # single surviving bunch == drive

    def cloud(pg):
        if pg is None or len(pg.x) == 0:
            return np.zeros((0, 6), np.float32)
        c = _stack(pg)
        return _subsample(c, p, rng) if c.shape[0] >= p else c

    def frac(pg, full):
        if pg is None or len(pg.x) == 0:
            return np.float32(0.0)
        return np.float32(np.clip((pg.charge * 1e9) / full, 0.0, 1.05))   # charge[C]*1e9 -> nC

    return {
        "drive": cloud(PD), "witness": cloud(PW),
        "T_drive": frac(PD, drive_full_nc), "T_witness": frac(PW, witness_full_nc),
        "n_drive": np.int32(len(PD.x) if PD is not None else 0),
        "n_witness": np.int32(len(PW.x) if PW is not None else 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-config", required=True)
    ap.add_argument("--num-macro", type=int, default=20000)
    ap.add_argument("--csr", type=int, default=1)
    ap.add_argument("--wakes", type=int, default=1)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--drive-full-nc", type=float, required=True)
    ap.add_argument("--witness-full-nc", type=float, required=True)
    ap.add_argument("--P", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)

    import FACET2_S2E as qs
    s2e = facet2_root()
    baseline = qs.loadConfig("/" + args.baseline_config, str(s2e))

    def make_tao():
        return qs.initializeTao(
            filePath=str(s2e), inputBeamFilePathSuffix=baseline["inputBeamFilePathSuffix"],
            csrTF=bool(args.csr), transverseWakes=bool(args.wakes),
            numMacroParticles=args.num_macro, scratchPath=args.scratch,
            randomizeFileNames=True)

    tao = make_tao()
    rng = np.random.default_rng(0)
    print(READY, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        try:
            req = json.loads(line)
            merged = {**baseline, **{k: float(req["knobs"][k]) for k in PARAM_KEYS}}
            qs.setLattice(tao, **merged)
            qs.trackBeam(tao, str(s2e), **merged)
            P = qs.getBeamAtElement(tao, "PENT")
            res = _extract(qs, P, int(req.get("P", args.P)),
                           args.drive_full_nc, args.witness_full_nc, rng)
            res["ok"] = np.bool_(True)
            res["error"] = ""
        except Exception as e:                     # noqa: BLE001 -- report, rebuild Tao, continue
            res = {"drive": np.zeros((0, 6), np.float32), "witness": np.zeros((0, 6), np.float32),
                   "T_drive": np.float32(0.0), "T_witness": np.float32(0.0),
                   "n_drive": np.int32(0), "n_witness": np.int32(0),
                   "ok": np.bool_(False), "error": f"{type(e).__name__}: {e}"}
            try:
                tao = make_tao()
            except Exception as e2:                # noqa: BLE001
                res["error"] += f" | tao reinit failed: {e2}"
        path = tempfile.mktemp(suffix=".npz", dir=args.scratch)
        np.savez(path, **res)
        print(RESULT + path, flush=True)


if __name__ == "__main__":
    main()
