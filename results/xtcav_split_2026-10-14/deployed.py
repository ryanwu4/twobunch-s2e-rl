"""Run the deployed E300 two-bunch algorithms (vendored, xtcav_metrics.py) on rendered sim DTOTR
images of the valgate beams, graded against the weight-tag truth."""
import json, sys, numpy as np, torch
from pmd_beamphysics import ParticleGroup
from scipy import constants
from twobunch_s2e_rl.xtcav_metrics import (energy_gated_image_features, find_bunch_separation,
                                           split_drive_witness_charge)
from xtcav_segment.convert import particlegroup_to_beam
from xtcav_segment.segment import back_drift, build_segment_from_spec, load_spec, set_tdc
C = constants.c; PX = 30.5e-6
spec = load_spec("configs/segment/segment_baseline.json")
e_ref, L = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
seg, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
camp = "data/xtcav23_valgate_2026-10-14"
man = json.load(open(f"{camp}/manifest.json"))
def trk(P, V):
    set_tdc(seg, V, 0.0); b, _ = particlegroup_to_beam(P, e_ref)
    o = seg.track(back_drift(b, L)); return o.x.numpy().copy(), o.y.numpy().copy()
out = []
for row in man:
    i = row["idx"]
    P = ParticleGroup(f"{camp}/sample_{i:05d}_XTCAVF.h5")
    q = np.asarray(P.weight); w = np.unique(q)
    if len(w) < 2: continue
    isw = q == w[0]; Q = q.sum() * 1e9
    x0, y0 = trk(P, 0.0); xp, yp = trk(P, 22e6)
    S = np.polyfit(P.z, xp - x0, 1)[0]
    zt = P.z
    truth = {"sep_mean": 1e6 * abs(zt[~isw].mean() - zt[isw].mean()),
             "sep_median": 1e6 * abs(np.median(zt[~isw]) - np.median(zt[isw])),
             "fd": q[~isw].sum() / q.sum()}
    rec = {"i": i, "block": "baseline" if row.get("is_baseline_repeat") else row["block"], "truth": truth}
    # off-shot row-distortion correction: subtract the TCAV-off per-row x centroid
    ry0 = np.floor(y0 / PX).astype(int); ryp = np.floor(yp / PX).astype(int)
    u, inv = np.unique(ry0, return_inverse=True)
    cent = dict(zip(u, np.bincount(inv, q * x0) / np.bincount(inv, q)))
    shift = np.array([cent.get(r, 0.0) for r in ryp]); shift -= np.average(x0, weights=q)
    for tag, x in (("raw", xp), ("rowcorr", xp - shift)):
        xe = np.arange(x.min() - 5 * PX, x.max() + 6 * PX, PX); ye = np.arange(yp.min() - 5 * PX, yp.max() + 6 * PX, PX)
        img, _, _ = np.histogram2d(yp, x, [ye, xe], weights=q)
        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
        zc = xc / S                               # sign: S<0 flips; z = head +
        order = np.argsort(zc); z_um = 1e6 * zc[order]
        proj = img.sum(0)[order]
        I_kA = proj / (abs(z_um[1] - z_um[0]) * 1e-6 / C) * 1e-3
        sep = find_bunch_separation(z_um, I_kA)
        r = {"peak_ok": bool(sep["ok"])}
        if sep["ok"]:
            pk = np.sort(sep["peak_positions_um"])
            zw = 1e6 * np.median(zt[isw])
            # is one of the two peaks the witness? (within 2 sigma of the witness median)
            r["peaks_are_drive_witness"] = bool(np.min(np.abs(pk - zw)) < 2e6 * zt[isw].std() + 20)
            r["sep_peak"] = float(sep["separation_um"])
            sp = split_drive_witness_charge(z_um, I_kA, peaks=sep)
            if sp["ok"]:
                zs = sp["z_split_um_used"]          # truth orientation: drive is the head (larger z)
                pred_w = 1e6 * zt < zs
                r["zsplit_mis"] = float(q[pred_w != isw].sum() / q.sum())
        eg = energy_gated_image_features(img, 1e3 * xc, 1e3 * yc, S, Q)
        r["egate_ok"] = bool(eg["ok"])
        if eg["ok"]:
            r["egate_sep_median"] = eg["separation_median_um"]
            r["egate_sep_mean"] = eg["separation_mean_um"]
            r["egate_fd"] = max(eg["band_low_y"]["Q_nC"], eg["band_high_y"]["Q_nC"]) / Q
        rec[tag] = r
    out.append(rec); print(i, rec["block"], json.dumps({k: rec[k] for k in ("truth", "rowcorr")}, default=float)[:300], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
