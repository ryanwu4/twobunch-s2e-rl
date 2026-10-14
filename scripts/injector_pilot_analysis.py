"""Injector pilot analysis: benchmark, paired injector effects vs the sensitivity-study prediction,
and the golden-h5 (old campaigns) vs Impact-T-golden bridge offset.

Usage (bmad-qpad-dev env: torch + pmd_beamphysics; repo root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/injector_pilot_analysis.py [--data data/injector_pilot_2026-10-14] \
      [--out results/injector_pilot_2026-10-14] [--study data/injector_sensitivity_2026-10-14]

Observables at XTCAVF are computed from the saved clouds with the SAME function as the
sensitivity study (scripts/injector_sensitivity.py::observables; surrogate/properties.py kernels,
drive = higher weight). Prediction = linear model sum_k s_k (x_k - golden_k), slopes s_k from the
study's absolute-arrival OAT runs (N = 2e4, grid 32, both signs -> central / least-squares slope;
shown converged vs 1e5 / grid 64 in that study). It is applied at all three shared linac points:
at the golden linac point it tests the reproduction + additivity of the study, off golden it
tests whether the injector response transfers across the linac box (i.e. how joint the campaign
design must be).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twobunch_s2e_rl.datagen.sweep_params import INJECTOR_KEYS, injector_golden

OBS = ["bunch_spacing_um", "drive_sigma_z_um", "witness_sigma_z_um", "charge_ratio_w_over_d",
       "energy_difference_MeV", "drive_energy_spread_pct", "witness_energy_spread_pct",
       "drive_norm_emit_x_um", "drive_norm_emit_y_um", "witness_norm_emit_x_um",
       "witness_norm_emit_y_um"]
SHOW = ["bunch_spacing_um", "drive_sigma_z_um", "witness_sigma_z_um", "charge_ratio_w_over_d",
        "energy_difference_MeV"]


def _sens_module():
    p = Path(__file__).with_name("injector_sensitivity.py")
    spec = importlib.util.spec_from_file_location("injector_sensitivity", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_rows(data: Path) -> pd.DataFrame:
    from pmd_beamphysics import ParticleGroup
    sens = _sens_module()
    rows = []
    for f in sorted(glob.glob(str(data / "sample_*.json")) + glob.glob(str(data / "bridge_*.json"))):
        d = json.loads(Path(f).read_text())
        r = {"idx": d["idx"], "file": Path(f).name, "block": d["block"], "beam": d["beam"],
             "injector_point": d["injector_point"], "linac_id": d["linac_id"],
             "success": d.get("success", False), "bmad_wall_s": d.get("wall_s"),
             "pid": d.get("pid"), "timestamp": d.get("timestamp")}
        r.update({f"inj:{k}": v for k, v in d["injector_settings"].items()})
        r.update({f"lin:{k}": d["linac_knobs"][k] for k in ("L1PhaseSet", "L2PhaseSet")})
        inj = d.get("injector", {})
        r["dt_l0afend_ps"] = inj.get("dt_l0afend_ps", 0.0)
        r["impact_wall_s"] = inj.get("impact_wall_s")
        r["q_in_drive_nC"] = d.get("injected_charge_nC", {}).get("drive")
        r["q_in_witness_nC"] = d.get("injected_charge_nC", {}).get("witness")
        h5 = Path(f).with_name(Path(f).stem + "_XTCAVF.h5")
        if r["success"] and h5.exists():
            P = ParticleGroup(str(h5))
            w = np.unique(P.weight)
            nd = int(round(r["q_in_drive_nC"] / 1e9 / w[-1])) if r["q_in_drive_nC"] else 1
            nw = int(round(r["q_in_witness_nC"] / 1e9 / w[0])) if r["q_in_witness_nC"] else 1
            r.update(sens.observables(P, nd, nw))
        rows.append(r)
    return pd.DataFrame(rows)


def study_slopes(study: Path) -> pd.DataFrame:
    """Per-variable slope d obs / d knob (knob units) from the study's absolute-mode OAT runs."""
    E = pd.read_csv(study / "effects.csv")
    E = E[(E.group == "oat") & (E.arrival == "absolute")]
    golden = injector_golden()
    out = {}
    for var, sub in E.groupby("variable"):
        if var not in golden:
            continue
        dx = []
        for d in sub["delta"]:
            d = json.loads(d)
            dx.append(d["abs"] if "abs" in d else golden[var] * d["rel"])
        dx = np.array(dx)
        out[var] = {o: float(np.polyfit(np.r_[0.0, dx], np.r_[0.0, sub[f"d_{o}"].values], 1)[0])
                    for o in OBS if f"d_{o}" in sub}
    return pd.DataFrame(out).T   # rows = variable


def paired_effects(df: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    """Delta vs the Impact-T golden-injector row at the same linac point, + linear prediction."""
    golden = injector_golden()
    ok = df[df.success & (df.beam == "impact")]
    out = []
    for lid, sub in ok.groupby("linac_id"):
        ref = sub[sub.injector_point == "golden"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, r in sub.iterrows():
            if r.injector_point == "golden":
                continue
            row = {"idx": r.idx, "linac_id": lid, "injector_point": r.injector_point,
                   "block": r.block, "dt_l0afend_ps": r.dt_l0afend_ps}
            dx = {k: r[f"inj:{k}"] - golden[k] for k in INJECTOR_KEYS}
            for o in OBS:
                if o not in r or pd.isna(r[o]) or pd.isna(ref.get(o)):
                    continue
                row[f"d_{o}"] = r[o] - ref[o]
                row[f"pred_{o}"] = sum(slopes.loc[k, o] * dx[k] for k in INJECTOR_KEYS
                                       if k in slopes.index and o in slopes.columns)
            out.append(row)
    return pd.DataFrame(out)


def bridge_offsets(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.success]
    out = []
    for lid, sub in ok.groupby("linac_id"):
        b = sub[sub.beam == "golden_h5"]
        g = sub[(sub.beam == "impact") & (sub.injector_point == "golden")]
        if b.empty or g.empty:
            continue
        b, g = b.iloc[0], g.iloc[0]
        row = {"linac_id": lid, "L1PhaseSet": g["lin:L1PhaseSet"], "L2PhaseSet": g["lin:L2PhaseSet"]}
        for o in OBS:
            if o in g and pd.notna(g[o]) and pd.notna(b.get(o)):
                row[f"impact_{o}"] = g[o]
                row[f"golden_h5_{o}"] = b[o]
                row[f"rel_{o}"] = (g[o] - b[o]) / b[o] if b[o] else np.nan
        out.append(row)
    return pd.DataFrame(out)


def plot(P: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    obs = ["bunch_spacing_um", "drive_sigma_z_um", "charge_ratio_w_over_d", "energy_difference_MeV"]
    fig, axs = plt.subplots(1, len(obs), figsize=(4.2 * len(obs), 4))
    for ax, o in zip(axs, obs):
        for lid, sub in P.groupby("linac_id"):
            m = sub.injector_point.str.startswith("oat")
            ax.scatter(sub[f"pred_{o}"][~m], sub[f"d_{o}"][~m], s=18, label=f"linac {lid} (LHS inj)")
            ax.scatter(sub[f"pred_{o}"][m], sub[f"d_{o}"][m], s=40, marker="x", label=f"linac {lid} (OAT)")
        lim = np.nanmax(np.abs(np.r_[P[f"pred_{o}"], P[f"d_{o}"]])) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_xlabel(f"predicted Δ (study slopes)")
        ax.set_ylabel(f"Bmad Δ vs Impact-T golden")
        ax.set_title(o)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "paired_effects_pred_vs_bmad.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/injector_pilot_2026-10-14")
    ap.add_argument("--out", default="results/injector_pilot_2026-10-14")
    ap.add_argument("--study", default="data/injector_sensitivity_2026-10-14")
    a = ap.parse_args()
    data, out, study = Path(a.data), Path(a.out), Path(a.study)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(data)
    df.to_csv(out / "pilot_rows.csv", index=False)
    S = study_slopes(study)
    S.to_csv(out / "study_slopes.csv")
    P = paired_effects(df, S)
    P.to_csv(out / "paired_effects.csv", index=False)
    B = bridge_offsets(df)
    B.to_csv(out / "bridge_offsets.csv", index=False)

    # Impact-T benchmark from the cached beams
    infos = [json.loads(Path(p).read_text()) for p in glob.glob(str(data / "beams" / "*" / "injector_info.json"))]
    imp = np.array([i["wall_s"] for i in infos])
    bm = df.loc[df.success, "bmad_wall_s"].astype(float).values
    summ = {
        "n_rows": int(len(df)), "n_success": int(df.success.sum()),
        "n_impact_beams": len(infos),
        "impact_wall_s": {"median": float(np.median(imp)), "min": float(imp.min()), "max": float(imp.max())} if len(imp) else None,
        "impact_run_s_median": float(np.median([i["impact_run_s"] for i in infos])) if infos else None,
        "bmad_wall_s": {"median": float(np.median(bm)), "min": float(bm.min()), "max": float(bm.max())} if len(bm) else None,
        "n_ambiguous_max": int(max(i["n_ambiguous"] for i in infos)) if infos else None,
        "witness_viable_frac": float(df.loc[df.success, "witness_sigma_z_um"].notna().mean()) if "witness_sigma_z_um" in df else None,
    }
    # prediction quality per observable and linac point (OAT + LHS rows)
    qual = {}
    for o in SHOW:
        if f"d_{o}" not in P:
            continue
        q = {}
        for lid, sub in P.groupby("linac_id"):
            d, p = sub[f"d_{o}"].values, sub[f"pred_{o}"].values
            m = np.isfinite(d) & np.isfinite(p)
            if m.sum() < 2:
                continue
            res = d[m] - p[m]
            q[int(lid)] = {"rms_measured": float(np.sqrt(np.mean(d[m] ** 2))),
                           "rms_residual": float(np.sqrt(np.mean(res ** 2))),
                           "max_abs_measured": float(np.abs(d[m]).max()),
                           "fit_gain": float(np.dot(p[m], d[m]) / max(np.dot(p[m], p[m]), 1e-30))}
        qual[o] = q
    summ["prediction_quality"] = qual
    (out / "pilot_summary.json").write_text(json.dumps(summ, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(json.dumps(summ, indent=2))
        cols = ["linac_id", "injector_point"] + [c for o in SHOW for c in (f"d_{o}", f"pred_{o}")]
        print(P[[c for c in cols if c in P]].round(3).to_string(index=False))
        print(B.round(4).T.to_string())
    if len(P):
        plot(P, out)


if __name__ == "__main__":
    main()
