"""Build the flow-surrogate training table from a campaign data dir.

Walks data/<subdir>/sample_*.json (+ sample_*_PENT.h5) and writes
processed/twobunch_flow.h5:
  knobs          (N, 8)     float32  -- the 8 sweep knobs (raw)
  drive_parts    (N, P, 6)  float32  -- drive 6D cloud (x,y,z,px,py,pz), 0 if absent
  witness_parts  (N, P, 6)  float32  -- witness 6D cloud, 0 if not density-trainable
  drive_present  (N,)       bool     -- PDrive specs present (drive survived + characterized)
  witness_viable (N,)       bool     -- PWitness specs present (witness survived)
  drive_density  (N,)       bool     -- drive has >= P live particles (usable for NLL)
  witness_density(N,)       bool     -- witness viable AND >= P live particles
  drive_frac     (N,)       float32  -- drive surviving fraction (charge / full), clipped
  witness_frac   (N,)       float32  -- witness surviving fraction (0 if destroyed)
  n_drive,n_witness (N,)    int32    -- live macroparticle counts
  idx            (N,)       int32    -- sample index (traceability)

+ processed/twobunch_flow_norm.json: knob bounds (from sweep_params), per-bunch
StandardScaler (drive/witness mean[6], std[6]), full per-bunch charges, P, coord order.

Split into drive/witness by the two unique per-particle weights (driver = higher weight),
matching FACET2_S2E.getDriverAndWitness. Witness density/coords are kept only where the
json marks the witness viable; the ~31% destroyed witnesses still contribute the
feasibility (viability/transmission) labels. Viable witnesses with < P live particles
(~4% of viable -- near-destroyed, low-surviving-fraction remnants, NOT high-emittance
survivors) are dropped from density training but still feed the feasibility heads.

Run in the slac-rl env:
  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.preprocess [--subdir full] [--P 1024]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
from pmd_beamphysics import ParticleGroup

from ..datagen.paths import repo_root
from ..datagen.sweep_params import PARAM_KEYS, SWEEP_PARAMS
from . import COORD_KEYS, DEFAULT_P, ELECTRON_MC2_EV

PENT = "PENT"


def _stack(pg: ParticleGroup) -> np.ndarray:
    """(n, 6) in COORD_KEYS order: (x, y, z, px, py, pz). pos [m], mom [eV/c]."""
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float32)


def _subsample(coords: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(coords.shape[0], p, replace=False)
    return coords[idx]


def parse_one(json_path: str, p: int, rng: np.random.Generator) -> dict | None:
    with open(json_path) as f:
        d = json.load(f)
    if not d.get("success"):
        return None
    spec = d.get("specs", {}).get(PENT, {})
    knobs = np.array([d["knobs"][k] for k in PARAM_KEYS], dtype=np.float32)

    drive_present = spec.get("PDrive_norm_emit_x") is not None
    witness_viable = spec.get("PWitness_norm_emit_x") is not None
    drive_charge = float(spec.get("PDrive_charge_nC") or 0.0)
    witness_charge = float(spec.get("PWitness_charge_nC") or 0.0)

    rec = {
        "idx": int(d["idx"]),
        "knobs": knobs,
        "drive_present": drive_present,
        "witness_viable": witness_viable,
        "drive_charge": drive_charge,
        "witness_charge": witness_charge,
        "drive_parts": np.zeros((p, 6), np.float32),
        "witness_parts": np.zeros((p, 6), np.float32),
        "drive_density": False,
        "witness_density": False,
        "n_drive": 0,
        "n_witness": 0,
    }

    if not (drive_present or witness_viable):
        return rec  # specs_error: feasibility-only (no particles extracted)

    h5_path = json_path.replace(".json", f"_{PENT}.h5")
    if not os.path.exists(h5_path):
        return rec
    try:
        P = ParticleGroup(h5_path)
    except Exception:
        return rec
    w = np.unique(P.weight)

    # drive = higher-weight subset (matches getDriverAndWitness); witness = lower.
    if len(w) >= 2:
        drive_pg = P[P.weight == w[-1]]
        witness_pg = P[P.weight == w[0]] if witness_viable else None
    else:
        drive_pg, witness_pg = P, None  # single surviving bunch is the drive

    if drive_present:
        nd = len(drive_pg.x)
        rec["n_drive"] = nd
        if nd >= p:
            rec["drive_parts"] = _subsample(_stack(drive_pg), p, rng)
            rec["drive_density"] = True
    if witness_viable and witness_pg is not None:
        nw = len(witness_pg.x)
        rec["n_witness"] = nw
        if nw >= p:
            rec["witness_parts"] = _subsample(_stack(witness_pg), p, rng)
            rec["witness_density"] = True
    return rec


def _scaler(records, key_parts, key_density, mode="intrabunch"):
    """Per-dim mean/std over density-trainable bunches of one species.

    mode='pooled'     : std of pooled raw particles (v1) -- for the witness this is dominated
                        by inter-sample centroid spread (~25x the intra-bunch size), so the
                        standardized intra-bunch shape becomes tiny and the whitening Sigma_k
                        target is badly conditioned.
    mode='intrabunch' : std of pooled PER-SAMPLE-CENTERED particles (v2 default) -> std reflects
                        the typical intra-bunch size, so standardized intra-bunch shape is ~O(1)
                        and the whitening regresses a well-scaled (mu_k, Sigma_k). Mean is the
                        pooled raw mean either way (the placement frame).
    """
    # float64 accumulation: pz ~1e10 eV/c summed over millions of float32 particles loses
    # ~340 MeV of precision in the mean (= ~21 sigma of the *witness* pz spread, but only
    # ~2 sigma for the fatter drive) -- which corrupts the witness LPS frame. Cast first.
    parts = [r[key_parts].astype(np.float64) for r in records if r[key_density]]
    if not parts:
        return np.zeros(6, np.float32), np.ones(6, np.float32)
    pooled = np.concatenate(parts, axis=0)
    mean = pooled.mean(axis=0)
    if mode == "pooled":
        std = pooled.std(axis=0)
    else:
        centered = np.concatenate([p - p.mean(axis=0, keepdims=True) for p in parts], axis=0)
        std = centered.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-12).astype(np.float32)


def preprocess(subdir="full", p=DEFAULT_P, max_samples=None, seed=0, out=None,
               scaler="intrabunch", verbose=True):
    data_dir = repo_root() / "data" / subdir
    files = sorted(glob.glob(str(data_dir / "sample_*.json")))
    if max_samples:
        files = files[:max_samples]
    if not files:
        raise SystemExit(f"No sample_*.json under {data_dir}")
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"Parsing {len(files)} samples from {data_dir} (P={p})")

    records = []
    for i, fp in enumerate(files):
        rec = parse_one(fp, p, rng)
        if rec is not None:
            records.append(rec)
        if verbose and (i + 1) % 500 == 0:
            nd = sum(r["drive_density"] for r in records)
            nw = sum(r["witness_density"] for r in records)
            print(f"  {i+1}/{len(files)}: kept {len(records)} (drive-dens {nd}, witness-dens {nw})")

    n = len(records)
    drive_mean, drive_std = _scaler(records, "drive_parts", "drive_density", scaler)
    witness_mean, witness_std = _scaler(records, "witness_parts", "witness_density", scaler)
    drive_full = max((r["drive_charge"] for r in records), default=1.0) or 1.0
    witness_full = max((r["witness_charge"] for r in records), default=1.0) or 1.0

    def frac(c, full):
        return float(np.clip(c / full, 0.0, 1.05))

    out = out or str(repo_root() / "processed" / "twobunch_flow.h5")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, "w") as h:
        h.create_dataset("knobs", data=np.stack([r["knobs"] for r in records]), compression="gzip")
        h.create_dataset("drive_parts", data=np.stack([r["drive_parts"] for r in records]),
                         compression="gzip", chunks=(1, p, 6))
        h.create_dataset("witness_parts", data=np.stack([r["witness_parts"] for r in records]),
                         compression="gzip", chunks=(1, p, 6))
        for col, dt in [("drive_present", bool), ("witness_viable", bool),
                        ("drive_density", bool), ("witness_density", bool)]:
            h.create_dataset(col, data=np.array([r[col] for r in records], dtype=dt))
        h.create_dataset("drive_frac", data=np.array([frac(r["drive_charge"], drive_full) for r in records], np.float32))
        h.create_dataset("witness_frac", data=np.array([frac(r["witness_charge"], witness_full) for r in records], np.float32))
        h.create_dataset("n_drive", data=np.array([r["n_drive"] for r in records], np.int32))
        h.create_dataset("n_witness", data=np.array([r["n_witness"] for r in records], np.int32))
        h.create_dataset("idx", data=np.array([r["idx"] for r in records], np.int32))
        h.attrs["P"] = p
        h.attrs["coord_order"] = ",".join(COORD_KEYS)

    norm = {
        "knob_keys": PARAM_KEYS,
        "knob_low": [float(SWEEP_PARAMS[k][0]) for k in PARAM_KEYS],
        "knob_high": [float(SWEEP_PARAMS[k][1]) for k in PARAM_KEYS],
        "drive_mean": drive_mean.tolist(), "drive_std": drive_std.tolist(),
        "witness_mean": witness_mean.tolist(), "witness_std": witness_std.tolist(),
        "drive_full_charge_nC": drive_full, "witness_full_charge_nC": witness_full,
        "coord_keys": list(COORD_KEYS), "electron_mc2_ev": ELECTRON_MC2_EV,
        "P": int(p), "n_records": n, "scaler": scaler,
    }
    with open(out.replace(".h5", "_norm.json"), "w") as f:
        json.dump(norm, f, indent=2)

    if verbose:
        nd = sum(r["drive_density"] for r in records)
        nw = sum(r["witness_density"] for r in records)
        wv = sum(r["witness_viable"] for r in records)
        print(f"Wrote {out}: N={n}, drive-density={nd}, witness-viable={wv}, witness-density={nw}")
        print(f"  full charge nC: drive {drive_full:.4f}, witness {witness_full:.4f}")
    return norm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdir", default="full")
    ap.add_argument("--P", type=int, default=DEFAULT_P)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scaler", choices=["pooled", "intrabunch"], default="intrabunch",
                    help="per-bunch std basis (v2 default 'intrabunch'; v1 used 'pooled')")
    args = ap.parse_args()
    preprocess(subdir=args.subdir, p=args.P, max_samples=args.max_samples,
               seed=args.seed, out=args.out, scaler=args.scaler)


if __name__ == "__main__":
    main()
