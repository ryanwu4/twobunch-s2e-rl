"""Harness audit: re-extract env-matching XTCAV features per TCAV-on shot (read-only on data).

For every TCAV-on DTOTR2 shot (|A| > 5 MV, any phase) of one E300 dataset, using the deployed
preprocessing + per-step off-shot row correction exactly as Tier 0 (imports the Tier 0 driver's
helpers, unmodified), compute on the energy-gated image (low-y band = drive, high-y = witness):
  sep_signed_um   z_drive_mean - z_witness_mean   (signed; z = x / cal_scaled, same for both signs)
  dsig_um         drive-band charge-weighted rms in z (per image)
  wsig_um         witness-band rms
  fd              drive charge fraction Q_low / (Q_low + Q_high)
  dE_MeV          drive - witness y-centroid difference converted with the nominal dispersion
The env's half-difference spacing is then (mean sep(+V) - mean sep(-V)) / 2 and its drive sigma_z
is sqrt(0.5 (var+ + var-)), computed downstream.

Run: PYTHONPATH=<repo>/src python reextract.py E300_15671 --chunk i --nchunks n
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

REPO = Path("/home/rwu4/photoinjector-rl/twobunch-s2e-rl")
sys.path.insert(0, str(REPO / "scripts"))
import xtcav_real_features as t0  # noqa: E402  (Tier 0 driver, imported read-only)

xtcav, DATASET = t0.xtcav, t0.DATASET
OUT = REPO / "results" / "harness_audit_2026-10-14" / "data"
T0 = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/tier0_real_features")


def band_feats(img, x_mm, y_mm, cal_scaled, smooth=2.0):
    img = np.asarray(img, float)
    yproj = ndimage.gaussian_filter(img.sum(axis=1), sigma=smooth)
    pks, _ = signal.find_peaks(yproj)
    if len(pks) < 2:
        return None
    prom = signal.peak_prominences(yproj, pks)[0]
    top2 = np.sort(pks[np.argsort(prom)[::-1][:2]])
    between = np.arange(top2[0], top2[1] + 1)
    sr = int(between[np.argmin(yproj[between])])
    z = 1e3 * np.asarray(x_mm, float) / cal_scaled
    y = np.asarray(y_mm, float)
    out = {"split_row": sr, "prom_ratio": float(np.sort(prom)[::-1][1] / max(np.sort(prom)[::-1][0], 1e-30))}
    for tag, sl in (("lo", slice(0, sr)), ("hi", slice(sr, None))):
        sub = img[sl, :]
        px = sub.sum(0)
        py = sub.sum(1)
        s = px.sum()
        if s <= 0:
            return None
        zm = np.average(z, weights=px)
        out[f"Q_{tag}"] = float(s)
        out[f"z_{tag}"] = float(zm)
        out[f"sz_{tag}"] = float(np.sqrt(np.average((z - zm) ** 2, weights=px)))
        out[f"y_{tag}"] = float(np.average(y[sl], weights=py))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    args = ap.parse_args()
    exp, num = args.dataset.rsplit("_", 1)
    summ = json.load(open(T0 / args.dataset / "summary.json"))
    cal_use = summ["calibration"]["cal_um_per_deg_used"]

    ds = DATASET(str(t0.XTCAV_ROOT), exp, num)
    cls = xtcav.classify_tcav_shots(ds)
    sps = xtcav.get_shot_steps(ds)
    a = xtcav.XTCAVAnalysis(ds, cam="DTOTR2", d_nom=t0.D_NOM, chargeCal=1.0,
                            hand_crop=t0.HAND_CROP, auto_crop=t0.AUTO_CROP,
                            hot_pix_threshold=t0.HOT_PIX_THRESHOLD,
                            noise_threshold=t0.NOISE_THRESHOLD, median_filter=t0.MEDIAN_FILTER)
    a.set_tcav_calibration(cal_use)
    refs = xtcav.build_step_distortion_references(a, cls, steps_per_shot=sps, verbose=False)
    amp, ph = np.asarray(cls["amp"], float), np.asarray(cls["phase"], float)
    on = np.where(amp > 5.0)[0]
    on = on[np.isin(sps[on], np.unique(sps)[args.chunk::args.nchunks])]
    keys = ["ind", "step", "amp", "phase", "ok", "Q_lo", "Q_hi", "z_lo", "z_hi", "sz_lo", "sz_hi",
            "y_lo", "y_hi", "split_row", "prom_ratio", "E_bend", "dy_mm", "ref_step", "charge_nC"]
    R = {k: np.full(len(on), np.nan) for k in keys}
    for j, ind in enumerate(on):
        R["ind"][j], R["step"][j], R["amp"][j], R["phase"][j] = ind, sps[ind], amp[ind], ph[ind]
        try:
            ref = xtcav.select_reference_for_shot(refs, int(ind), sps)
            shot, img, corrected = t0.corrected_shot(a, int(ind), ref)
            # 'cal_scaled' sign: same convention as Tier 0 (no per-sign flip)
            f = band_feats(img.data, img.x, img.y, a.cal_scaled)
        except Exception:
            continue
        R["E_bend"][j], R["dy_mm"][j], R["charge_nC"][j] = shot["E_bend"], shot["dy"], shot["charge_nC"]
        R["ref_step"][j] = float(ref.get("step", np.nan)) if isinstance(ref, dict) and ref.get("step") is not None else np.nan
        if f is None:
            R["ok"][j] = 0
            continue
        R["ok"][j] = 1
        for k in ("Q_lo", "Q_hi", "z_lo", "z_hi", "sz_lo", "sz_hi", "y_lo", "y_hi", "split_row", "prom_ratio"):
            R[k][j] = f[k]
    np.savez_compressed(OUT / f"reext_{args.dataset}_c{args.chunk:02d}.npz", cal_scaled=a.cal_scaled,
                        cal_um_per_deg=cal_use, n_off_per_step=json.dumps(
                            {int(s): int(np.sum(sps[cls['ix_off']] == s)) for s in np.unique(sps)}), **R)
    print(args.dataset, args.chunk, "done", len(on), int(np.nansum(R["ok"])))


if __name__ == "__main__":
    main()
