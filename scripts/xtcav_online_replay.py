"""Replay the online XTCAV features (twobunch_s2e_rl.online.features, the training-matched rules) on real
E300 DAQ data, per scan step, with the deployed preprocessing chain (xtcav_real_features.py: hand crop,
hot pixels, noise threshold, median filter, per-step off-shot row correction).

Per step: one energy cut from the step's +V images (decision D2; --cut image = per-image valley),
band_stats on every +V and -V image with it, median aggregation per sign,
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

from twobunch_s2e_rl.online.features import band_stats, step_cut, step_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=15, help="images per sign in the bootstrap step estimate")
    ap.add_argument("--limit-per-step", type=int, default=None)
    ap.add_argument("--cut", default="step_median", choices=["step", "step_median", "image"],
                    help="step_median (default): median of the per-image +V cuts, shared with -V (D2); step: valley "
                         "of the summed +V projection (replay v3); image: per-image valley (v1/v2)")
    ap.add_argument("--qa-steps", default="", help="comma list of steps: save median +V/-V images + cuts")
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
    qa_imgs = {}
    out = {"dataset": a_.dataset, "cal_um_per_deg": cal, "cal_scaled": S, "M": a_.M, "steps": {}}
    for step in sorted(set(int(s) for s in np.unique(steps))):
        st = {"plus": [], "minus": []}
        qa = {"plus": [], "minus": []} if str(step) in a_.qa_steps.split(",") else None
        ix = []
        for grp in (cls["ix_plus"], cls["ix_neg"]):           # cap per sign, not overall
            ix += [int(i) for i in grp if steps[int(i)] == step][:a_.limit_per_step]
        n_fail, loaded = 0, []
        for i in ix:
            try:
                ref = rf.xtcav.select_reference_for_shot(refs, i, steps)
                _, img, _ = rf.corrected_shot(a, i, ref)
                loaded.append(("plus" if i in plus else "minus", img.data.copy(), 1e-3 * np.asarray(img.x, float)))
            except Exception:
                n_fail += 1
        cut = None
        if a_.cut in ("step", "step_median") and loaded:
            shp = loaded[0][1].shape
            cut, ok = step_cut([d for sg, d, _ in loaded if sg == "plus" and d.shape == shp],
                               method="median" if a_.cut == "step_median" else "sum")
            if not ok:
                n_fail += len(loaded)
                loaded = []
        for sg, d, x in loaded:
            b = band_stats(d, x, cut_row=cut if (cut is not None and d.shape == loaded[0][1].shape) else None)
            if b is None:
                n_fail += 1
                continue
            st[sg].append(b)
            if qa is not None and d.shape == loaded[0][1].shape:
                qa[sg].append((d, b["cut_row"]))
        if not st["plus"] or not st["minus"]:
            continue
        # x in mm-on-screen -> z via S; +V streak +S, -V streak -S (signed, D3)
        f = step_features(st["plus"], st["minus"], streak=S, floor_m=60e-6, edge_frac_max=0.01)
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
            "drive_charge_fraction": f["drive_charge_fraction"], "valid": f["valid"], "edge_frac": f["edge_frac"],
            "cut_sens_um": 1e6 * f["cut_sens"],
            "single_plus_median_um": 1e6 * float(np.median(sep["plus"])),
            "single_minus_median_um": 1e6 * float(np.median(sep["minus"])),
            "single_image_robust_sigma_um": 1e6 * 0.5 * (iqr(sep["plus"]) + iqr(sep["minus"])),
            "step_bootstrap_sigma_um": 1e6 * float(np.std(boot)),
            "offsets_plus_um": [round(1e6 * (d["x_d"] - d["x_w"]) / S, 1) for d in st["plus"]],
            "offsets_minus_um": [round(1e6 * (d["x_d"] - d["x_w"]) / S, 1) for d in st["minus"]],
            "cut_rows_plus": [d["cut_row"] for d in st["plus"]],
            "cut_rows_minus": [d["cut_row"] for d in st["minus"]],
        }
        if qa is not None:
            qa_imgs[step] = qa
        print(step, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in out["steps"][step].items()}), flush=True)
    o = Path(a_.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{a_.dataset}.json").write_text(json.dumps(out, indent=1))
    if qa_imgs:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(qa_imgs), 4, figsize=(14, 2.8 * len(qa_imgs)), squeeze=False)
        for r, (step, qa) in enumerate(sorted(qa_imgs.items())):
            for c, sg in enumerate(("plus", "minus")):
                if not qa[sg]:
                    continue
                med = np.median([im for im, _ in qa[sg]], axis=0)
                ax[r, 2 * c].imshow(med, aspect="auto", origin="lower", cmap="magma")
                for _, k in qa[sg][:20]:
                    ax[r, 2 * c].axhline(k, color="c", lw=0.4, alpha=0.5)
                ax[r, 2 * c].set_title(f"step {step} {sg}: median of {len(qa[sg])}", fontsize=8)
                ax[r, 2 * c + 1].plot(med.sum(axis=1))
                ax[r, 2 * c + 1].set_title("y projection (cuts: cyan on image)", fontsize=8)
        fig.tight_layout()
        fig.savefig(o / f"{a_.dataset}_qa.png", dpi=110)


if __name__ == "__main__":
    main()
