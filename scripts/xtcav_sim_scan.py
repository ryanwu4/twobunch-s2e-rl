"""Tier 1 sim2real: matched FACET2-S2E replication of the E300 L2 phase scans.

For one dataset, tracks the 33-step L2 scan at the operating point reconstructed in
machine_config.json (built by xtcav_tier1_configs.py): per-step L1 from the klystron
readbacks, L2 = L2_ref + scanVal (L2_ref is the ONE fitted parameter, see TIER1_PLAN §3),
FF/spectrometer quads from LGPS BACTs, B5D36 from its BACT, XTCAV at ±22 MV zero
crossings. Produces feature tables in the same schema family as Tier 0 so Tier 2 can
diff them directly. Features are computed by the SAME code path as the real data
(twobunch_s2e_rl.xtcav_metrics; deployed conventions).

Per step: full track L0AFEND->DTOTR with XTCAV off (~36 s) -> off/unstreaked features ->
checkpoint at ENDBC20 -> two 0.5 s retracks at +/-22 MV -> pixelated (30.5 um/px)
DTOTR2-like features. One 5-point RF phase-scan calibration per run (~3 s) supplies
cal_scaled_sim, used for all steps (mirrors one-cal-per-dataset on the real side).

Runtimes: --smoke (1 step + cal) ~1.5 min; full 33-step dataset ~22 min.

Run (bmad-qpad-dev env, from twobunch-s2e-rl root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_sim_scan.py \
    E300_15671 --l2ref -38 --quad-map value_match [--steps 1,17,33] [--smoke]
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import FACET2_S2E as qs

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from xtcav_dtotr_test import (  # noqa: E402
    CHECKPOINT, PIXEL_UM, S2E_ROOT, CONFIG_FILE,
    make_pixel_image, phase_scan_calibration, set_xtcav,
)
from twobunch_s2e_rl.xtcav_metrics import (  # noqa: E402
    C_LIGHT, energy_gated_image_features, find_bunch_separation, fit_bigaussian,
    split_drive_witness_charge,
)

XTCAV_ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
OUT_ROOT = XTCAV_ROOT / "sim2real" / "results" / "tier1_sim_scans"
BEAM_KEEP_STEPS = (1, 17, 33)


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def step_overrides(mc, base, step, l2ref, quad_map, l1_abs=None):
    ov = {
        "L1PhaseSet": (l1_abs if l1_abs is not None else
                       mc["L1_anchor_deg"] + mc["L1_delta_per_step_deg"][str(step)]),
        "L2PhaseSet": l2ref + mc["L2_scan_vals_deg"][step - 1],
        "B5D36GeVc": mc["B5D36GeVc"],
        "Q19851kG": mc["Q19851kG"],
        "Q19871kG": mc["Q19871kG"],
    }
    if quad_map != "golden":
        for ele, entry in mc["quad_mappings"][quad_map].items():
            ov[ele] = entry["kG"]
    return {**base, **ov}


def image_features(P, cal_scaled):
    """Pixelated-screen features via the shared (real-data-identical) code path."""
    img, x_edges, y_edges = make_pixel_image(P, PIXEL_UM)
    x_mm = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mm = 0.5 * (y_edges[:-1] + y_edges[1:])
    Q_nC = P.charge * 1e9

    out = {"charge_nC": float(Q_nC)}
    eg = energy_gated_image_features(img, x_mm, y_mm, cal_scaled, Q_nC)
    out["eg_ok"] = float(eg["ok"])
    if eg["ok"]:
        out["eg_Q_low_nC"] = eg["band_low_y"]["Q_nC"]
        out["eg_Q_high_nC"] = eg["band_high_y"]["Q_nC"]
        out["eg_sep_mean_um"] = eg["separation_mean_um"]
        out["eg_sep_median_um"] = eg["separation_median_um"]

    # deployed-style I(z): x-projection, z = x/cal, peak at 0, integral I dt = Q
    from scipy import integrate, ndimage
    proj = img.sum(axis=0)
    z_um = 1e3 * x_mm / cal_scaled
    z_um = z_um - z_um[int(np.argmax(ndimage.gaussian_filter(proj, 2.0)))]
    order = np.argsort(z_um)
    z_um, proj = z_um[order], proj[order]
    denom = integrate.trapezoid(proj, z_um * 1e-6)
    I_kA = (proj * Q_nC * 1e-9 * C_LIGHT / denom * 1e-3
            if denom > 0 else np.zeros_like(proj))

    sep = find_bunch_separation(z_um, I_kA)
    fit = fit_bigaussian(z_um, I_kA, peaks=sep)
    split = split_drive_witness_charge(z_um, I_kA, peaks=sep)
    if sep["ok"]:
        out["sep_peak_um"] = sep["separation_um"]
    if fit["ok"]:
        out["bigauss_R2"] = fit["R_squared"]
        out["sigma1_um"] = fit["sigma1_um"]
        if fit["two_bunch"]:
            out["sep_bigauss_um"] = fit["separation_um"]
            out["sigma2_um"] = fit["sigma2_um"]
    if split["ok"]:
        out["Q_minsplit_lo_nC"] = split["Q_drive_nC"]
        out["Q_minsplit_hi_nC"] = split["Q_witness_nC"]

    # energy proxies from particles (sim-only ground-truth-ish covariates)
    out["E_mean_GeV"] = float(np.average(P.energy, weights=P.weight) * 1e-9)
    out["E_rms_GeV"] = float(np.sqrt(np.average(
        (P.energy * 1e-9 - out["E_mean_GeV"]) ** 2, weights=P.weight)))
    out["z_um_grid"] = z_um
    out["current_kA"] = I_kA
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--l2ref", type=float, required=True,
                    help="absolute L2PhaseSet at scanVal=0 (the fitted parameter)")
    ap.add_argument("--quad-map", choices=["value_match", "alias_file", "golden"],
                    default="value_match")
    ap.add_argument("--steps", type=str, default="all")
    ap.add_argument("--nmacro", type=float, default=1e4)
    ap.add_argument("--smoke", action="store_true", help="step 1 + calibration only")
    ap.add_argument("--outtag", type=str, default=None)
    ap.add_argument("--l1-abs", type=float, default=None,
                    help="absolute L1PhaseSet overriding anchor+deltas "
                         "(e.g. from the BC11 sigma_z servo)")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.dataset
    mc = json.load(open(out_dir / "machine_config.json"))
    tag = args.outtag or f"l2ref{args.l2ref:+.2f}_{args.quad_map}"
    run_dir = out_dir / tag
    scratch = run_dir / "scratch"
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    steps = ([1] if args.smoke else
             list(range(1, len(mc["L2_scan_vals_deg"]) + 1)) if args.steps == "all"
             else [int(s) for s in args.steps.split(",")])
    V = mc["xtcav_voltage_V"]

    base = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))
    tao = qs.initializeTao(
        filePath=str(S2E_ROOT),
        inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
        csrTF=True, numMacroParticles=args.nmacro,
        scratchPath=str(scratch), randomizeFileNames=True,
    )

    # makeBeamActiveBeamFile (checkpoint reuse) overwrites the active beam file
    # with the ENDBC20 beam; the NEXT step's full track would then inject a
    # 10 GeV beam at L0AFEND and blow up in the injector. Keep the pristine
    # L0AFEND beam aside and restore it at the top of every step.
    import shutil
    pristine = run_dir / "input_L0AFEND.h5"
    shutil.copy(tao.activeFilePath, pristine)

    results = {"dataset": args.dataset, "l2ref": args.l2ref,
               "quad_map": args.quad_map, "nmacro": int(args.nmacro),
               "steps": {}}
    cal_scaled = None
    rows = []
    t_run = time.perf_counter()

    for step in steps:
        shutil.copy(pristine, tao.activeFilePath)  # restore L0AFEND input beam
        cfg = step_overrides(mc, base, step, args.l2ref, args.quad_map,
                             l1_abs=args.l1_abs)
        t0 = time.perf_counter()
        qs.setLattice(tao, **cfg)
        set_xtcav(tao, 0.0, 0.0)
        qs.trackBeam(tao, str(S2E_ROOT), trackEnd="DTOTR", **cfg)
        t_full = time.perf_counter() - t0

        P_off = qs.getBeamAtElement(tao, "DTOTR")
        if len(P_off.x) < 0.5 * args.nmacro:
            print(f"step {step}: WARNING - beam mostly lost "
                  f"({len(P_off.x)}/{int(args.nmacro)}); recording and continuing")
        P_ckpt = qs.getBeamAtElement(tao, CHECKPOINT)
        qs.makeBeamActiveBeamFile(P_ckpt, tao=tao)

        # one calibration per run, from the first tracked step's checkpoint
        if cal_scaled is None:
            cal = phase_scan_calibration(tao, cfg, V)
            cal_scaled = cal["cal_scaled"]
            results["phase_scan_calibration"] = cal
            print(f"cal_scaled_sim = {cal_scaled:.2f} um/um "
                  f"({cal['cal_um_per_deg']:.0f} um/deg) [real ~15.3 @ 1113 um/deg]")

        step_res = {"config": {k: cfg[k] for k in
                               ["L1PhaseSet", "L2PhaseSet", "B5D36GeVc"]},
                    "t_full_track_s": t_full,
                    "n_live_off": int(len(P_off.x))}
        # off/unstreaked: spot size on streak axis (resolution numerator)
        step_res["off_sigma_x_um"] = 1e6 * qs.smallestIntervalImpliedSigma(P_off.x)

        for sign, stag in [(+1, "plus"), (-1, "neg")]:
            set_xtcav(tao, sign * V, 0.0)
            qs.trackBeam(tao, str(S2E_ROOT), trackStart=CHECKPOINT,
                         trackEnd="DTOTR", **cfg)
            P = qs.getBeamAtElement(tao, "DTOTR")
            feats = image_features(P, cal_scaled)
            z_grid = feats.pop("z_um_grid")
            I_kA = feats.pop("current_kA")
            step_res[stag] = feats
            rows.append({"step": step, "scan_val": mc["L2_scan_vals_deg"][step - 1],
                         "streak_sign": float(sign), **feats})
            np.savez(run_dir / f"profile_step{step:02d}_{stag}.npz",
                     z_um=z_grid, current_kA=I_kA)
            if step in BEAM_KEEP_STEPS and sign > 0:
                P.write(str(run_dir / f"beam_DTOTR_step{step:02d}_plus.h5"))

        results["steps"][str(step)] = step_res
        print(f"step {step}: track {t_full:.0f} s, "
              f"eg_sep(+)={step_res['plus'].get('eg_sep_median_um', float('nan')):.1f} um, "
              f"Q_hi/lo={step_res['plus'].get('eg_Q_high_nC', float('nan')):.2f}/"
              f"{step_res['plus'].get('eg_Q_low_nC', float('nan')):.2f} nC")

    # flat arrays, Tier-0-compatible key names
    if rows:
        keys = sorted({k for r in rows for k in r})
        flat = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                for k in keys}
        np.savez_compressed(run_dir / "features.npz", **flat)

    results["total_s"] = time.perf_counter() - t_run
    with open(run_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=to_jsonable)

    # QA: separation + charge split vs scan value
    if not args.smoke and rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
        sv = flat["scan_val"]; pl = flat["streak_sign"] > 0
        ax1.plot(sv[pl], flat["eg_sep_median_um"][pl], "o-", label="sim eg sep (median)")
        ax1.set_xlabel("L2 scan value [deg]"); ax1.set_ylabel("[um]"); ax1.legend()
        ax2.plot(sv[pl], flat["eg_Q_high_nC"][pl], "o-", label="Q high band")
        ax2.plot(sv[pl], flat["eg_Q_low_nC"][pl], "o-", label="Q low band")
        ax2.set_xlabel("L2 scan value [deg]"); ax2.set_ylabel("[nC]"); ax2.legend()
        fig.tight_layout(); fig.savefig(run_dir / "qa_scan.png"); plt.close(fig)

    print(f"\ndone in {results['total_s']:.0f} s -> {run_dir}")


if __name__ == "__main__":
    main()
