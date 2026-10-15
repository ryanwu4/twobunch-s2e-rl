"""Replay the online XTCAV features (twobunch_s2e_rl.online.features, the training-matched rules) on real
E300 DAQ data, per scan step, with the deployed preprocessing chain (xtcav_real_features.py: hand crop,
hot pixels, noise threshold, median filter, per-step off-shot row correction).

Per step: band_stats on every +V and -V image (valley rule on each image), median aggregation per sign,
half-difference spacing with the signed calibration (decision D3). Also the per-image single-sign
separations, their shot-to-shot scatter (-> the env's per-image shot noise), and a bootstrap of the
step-level estimate at M images per sign (-> the env's per-step noise).

  PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nice -n 10 /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/xtcav_online_replay.py E300_15671 --out results/online_replay_2026-10-15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xtcav_real_features as rf  # noqa: E402  (deployed chain + calibration)

from twobunch_s2e_rl.online.features import band_stats, step_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=15, help="images per sign in the bootstrap step estimate")
    ap.add_argument("--limit-per-step", type=int, default=None)
    a_ = ap.parse_args()
    exp, num = a_.dataset.rsplit("_", 1)
    ds = rf.DATASET(str(rf.XTCAV_ROOT), exp, num)
    scan_vals = np.asarray(ds.params["scanVals"], float)
    cls = rf.xtcav.classify_tcav_shots(ds)
    steps = rf.xtcav.get_shot_steps(ds)
    tier0 = json.loads((rf.XTCAV_ROOT / "sim2real" / "results" / "tier0_real_features" / a_.dataset
                        / "summary.json").read_text())
    cal = tier0["calibration"]["cal_um_per_deg_used"]       # same calibration as Tier 0 (no re-fit)
    a = rf.xtcav.XTCAVAnalysis(ds, cam="DTOTR2", d_nom=rf.D_NOM, chargeCal=1.0, hand_crop=rf.HAND_CROP,
                               auto_crop=rf.AUTO_CROP, hot_pix_threshold=rf.HOT_PIX_THRESHOLD,
                               noise_threshold=rf.NOISE_THRESHOLD, median_filter=rf.MEDIAN_FILTER)
    a.set_tcav_calibration(cal)
    S = float(a.cal_scaled)                                 # screen um per um of z (magnitude)
    refs = rf.xtcav.build_step_distortion_references(a, cls, steps_per_shot=steps)
    plus = set(int(i) for i in cls["ix_plus"])
    rng = np.random.default_rng(0)
    out = {"dataset": a_.dataset, "cal_um_per_deg": cal, "cal_scaled": S, "M": a_.M, "steps": {}}
    for step in sorted(set(int(s) for s in np.unique(steps))):
        st = {"plus": [], "minus": []}
        ix = []
        for grp in (cls["ix_plus"], cls["ix_neg"]):           # cap per sign, not overall
            ix += [int(i) for i in grp if steps[int(i)] == step][:a_.limit_per_step]
        n_fail = 0
        for i in ix:
            try:
                ref = rf.xtcav.select_reference_for_shot(refs, i, steps)
                _, img, _ = rf.corrected_shot(a, i, ref)
                b = band_stats(img.data, 1e-3 * np.asarray(img.x, float))
            except Exception:
                b = None
            if b is None:
                n_fail += 1
                continue
            st["plus" if i in plus else "minus"].append(b)
        if not st["plus"] or not st["minus"]:
            continue
        # x in mm-on-screen -> z via S; +V streak +S, -V streak -S (signed, D3)
        f = step_features(st["plus"], st["minus"], streak=S, floor_m=60e-6)
        sep = {k: np.array([(d["x_d"] - d["x_w"]) / S for d in v]) * (1 if k == "plus" else -1)
               for k, v in st.items()}                  # single-image spacing estimates [m], signed
        boot = []
        for _ in range(200):
            pp = [st["plus"][j] for j in rng.integers(0, len(st["plus"]), a_.M)]
            mm = [st["minus"][j] for j in rng.integers(0, len(st["minus"]), a_.M)]
            boot.append(step_features(pp, mm, streak=S)["bunch_spacing"])
        iqr = lambda v: float(np.subtract(*np.percentile(v, [75, 25])) / 1.349)   # robust sigma
        out["steps"][step] = {
            "scan_val": float(scan_vals[step - 1]) if 1 <= step <= len(scan_vals) else None,
            "n_plus": len(st["plus"]), "n_minus": len(st["minus"]), "n_gate_fail": n_fail,
            "spacing_um": 1e6 * f["bunch_spacing"], "drive_sigma_z_um": 1e6 * f["drive_sigma_z"],
            "drive_charge_fraction": f["drive_charge_fraction"], "valid": f["valid"],
            "single_plus_median_um": 1e6 * float(np.median(sep["plus"])),
            "single_minus_median_um": 1e6 * float(np.median(sep["minus"])),
            "single_image_robust_sigma_um": 1e6 * 0.5 * (iqr(sep["plus"]) + iqr(sep["minus"])),
            "step_bootstrap_sigma_um": 1e6 * float(np.std(boot)),
        }
        print(step, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in out["steps"][step].items()}), flush=True)
    o = Path(a_.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{a_.dataset}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
