"""Consolidate the per-sample JSONs from a campaign data dir into one flat table.

Usage:
  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.analysis.build_dataset [data_subdir]
    data_subdir defaults to "full" -> reads data/full/sample_*.json

Output: artifacts/dataset.pkl  (pandas DataFrame, one row per sample)
        artifacts/dataset.csv  (same, human-inspectable)

Columns:
  meta:   idx, success, is_baseline_repeat, num_macro_particles, csrTF,
          transverseWakes, wall_s
  knobs:  the 8 sweep knobs (input parameters)
  specs:  {POINT}__{metric}  for POINT in {BEGBC20, MFFF, PENT}
          sliced_BMAG_* lists are reduced to _max / _mean derived columns.
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from ..datagen.paths import repo_root

POINTS = ["BEGBC20", "MFFF", "PENT"]


def data_dir(subdir="full"):
    return repo_root() / "data" / subdir


def artifacts_dir():
    d = repo_root() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def flatten_sample(d):
    row = {}
    for k in ("idx", "success", "is_baseline_repeat", "num_macro_particles",
              "csrTF", "transverseWakes", "wall_s"):
        row[k] = d.get(k)
    for k, v in d["knobs"].items():
        row[k] = v
    for pt in POINTS:
        spec = d["specs"].get(pt, {})
        for mk, mv in spec.items():
            if isinstance(mv, list):
                arr = np.asarray(mv, dtype=float)
                with warnings.catch_warnings():  # all-NaN slice is expected (absent bunch)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row[f"{pt}__{mk}_max"] = np.nanmax(arr) if arr.size else np.nan
                    row[f"{pt}__{mk}_mean"] = np.nanmean(arr) if arr.size else np.nan
            else:
                row[f"{pt}__{mk}"] = mv
    return row


def build(subdir="full"):
    ddir = data_dir(subdir)
    files = sorted(glob.glob(str(ddir / "sample_*.json")))
    print(f"Found {len(files)} sample JSONs in {ddir}")
    if not files:
        raise SystemExit(f"No sample_*.json under {ddir}")
    rows = []
    for i, fn in enumerate(files):
        with open(fn) as fh:
            rows.append(flatten_sample(json.load(fh)))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i+1}")
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    print(f"DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")

    art = artifacts_dir()
    out_pkl, out_csv = art / "dataset.pkl", art / "dataset.csv"
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_pkl}\nWrote {out_csv}")
    return df


def health_report(df):
    print("\n=== column non-finite/zero counts (numeric cols, flagged only) ===")
    num = df.select_dtypes(include=[np.number])
    health = []
    for c in num.columns:
        v = num[c].to_numpy(dtype=float)
        health.append((c, int(np.isnan(v).sum()), int(np.isinf(v).sum()), int((v == 0).sum())))
    hdf = pd.DataFrame(health, columns=["col", "n_nan", "n_inf", "n_zero"])
    flagged = hdf[(hdf.n_nan > 0) | (hdf.n_inf > 0) | (hdf.n_zero > 0)]
    print(flagged.to_string(index=False) if len(flagged) else "no NaN/inf/zero")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subdir", nargs="?", default="full", help="data subdir (default: full)")
    args = p.parse_args()
    df = build(args.subdir)
    health_report(df)


if __name__ == "__main__":
    main()
