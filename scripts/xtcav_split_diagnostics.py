"""Why the energy-gated bunch split fails in sim and how the z-lobe split behaves instead.

Takes the Bmad beams of the xtcav23 validation-gate campaign (saved at XTCAVF, 100k particles,
CSR + wakes), tracks each through the Cheetah XTCAVF -> DTOTR segment with the TCAV at +V (22 MV),
and compares three ways of placing the split on each axis against the macroparticle truth tag:

  midpoint  threshold halfway between the two bunch centroids (what screen-mode training uses)
  valley    deployed-style: minimum of the smoothed pixel projection between its two most
            prominent peaks (xtcav_metrics.energy_gated_image_features does this on y)
  optimal   the best any single threshold could do (lower bound on the misassignment)

for the energy axis y (energy gate) and the streak axis x (z-lobe split). Misassignment = fraction
of total charge put in the wrong bunch.

  PYTHONPATH=$PWD/src python scripts/xtcav_split_diagnostics.py --out results/xtcav_split_2026-10-14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

from xtcav_segment.convert import particlegroup_to_beam  # noqa: E402
from xtcav_segment.segment import back_drift, build_segment_from_spec, load_spec, set_tdc  # noqa: E402

PIXEL_M = 30.5e-6
REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "data" / "xtcav23_valgate_2026-10-14"
SPEC = REPO / "configs" / "segment" / "segment_baseline.json"
C_D, C_W = "#1f77b4", "#ff7f0e"


def misassign(v, q, is_w, thr, w_side):
    """Charge fraction on the wrong side of `thr` (witness expected on side sign(w_side))."""
    pred_w = np.sign(w_side) * (v - thr) > 0
    return float(q[pred_w != is_w].sum() / q.sum())


def thresholds(v, q, is_w):
    md, mw = np.average(v[~is_w], weights=q[~is_w]), np.average(v[is_w], weights=q[is_w])
    side = mw - md
    out = {"midpoint": 0.5 * (md + mw)}
    # deployed-style valley on the pixel projection
    lo, hi = np.percentile(v, [0.05, 99.95])
    edges = np.arange(lo - 5 * PIXEL_M, hi + 5 * PIXEL_M, PIXEL_M)
    proj = ndimage.gaussian_filter(np.histogram(v, edges, weights=q)[0], 2.0)
    pks, _ = signal.find_peaks(proj)
    if len(pks) >= 2:
        prom = signal.peak_prominences(proj, pks)[0]
        top2 = np.sort(pks[np.argsort(prom)[::-1][:2]])
        k = top2[0] + np.argmin(proj[top2[0]:top2[1] + 1])
        out["valley"] = 0.5 * (edges[k] + edges[k + 1])
        out["valley_prom_ratio"] = float(np.sort(prom)[-2] / np.sort(prom)[-1])
    else:
        out["valley"] = np.nan
        out["valley_prom_ratio"] = 0.0
    # best single threshold (scan between the centroids and a bit beyond)
    grid = np.linspace(min(md, mw) - abs(side), max(md, mw) + abs(side), 801)
    errs = [misassign(v, q, is_w, t, side) for t in grid]
    out["optimal"] = float(grid[int(np.argmin(errs))])
    mis = {k: (misassign(v, q, is_w, out[k], side) if np.isfinite(out[k]) else np.nan)
           for k in ("midpoint", "valley", "optimal")}
    return out, mis, side


def analyse(i, segment, e_ref, l_tdc):
    from pmd_beamphysics import ParticleGroup
    P = ParticleGroup(str(CAMPAIGN / f"sample_{i:05d}_XTCAVF.h5"))
    w = np.unique(P.weight)
    if len(w) < 2:
        return None
    beam, _ = particlegroup_to_beam(P, e_ref)
    out = segment.track(back_drift(beam, l_tdc))
    x, y = out.x.detach().numpy(), out.y.detach().numpy()
    q = np.asarray(P.weight)
    is_w = q == w[0]                                          # drive = higher weight (S2E convention)
    delta = (np.asarray(P.energy) - e_ref) / e_ref
    z = np.asarray(P.z)
    rec = {"i": i}
    for axis, v in (("energy", y), ("zlobe", x)):
        thr, mis, side = thresholds(v, q, is_w)
        rec[axis] = {"thr": thr, "mis": mis}
    dd, dw = delta[~is_w], delta[is_w]
    rec.update(dE_rel=float(dd.mean() - dw.mean()), sig_d=float(dd.std()), sig_w=float(dw.std()),
               spacing_um=float(1e6 * (z[~is_w].mean() - z[is_w].mean())),
               sigz_d_um=float(1e6 * z[~is_w].std()), sigz_w_um=float(1e6 * z[is_w].std()),
               q_w_frac=float(q[is_w].sum() / q.sum()))
    return rec, dict(x=x, y=y, q=q, is_w=is_w, z=z, delta=delta)


def panel(fig_path, title, rec, d):
    x, y, q, is_w, z, delta = (d[k] for k in ("x", "y", "q", "is_w", "z", "delta"))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    for m, c, lab in ((~is_w, C_D, "drive"), (is_w, C_W, "witness")):
        a.scatter(1e6 * z[m][::5], 1e2 * delta[m][::5], s=0.3, c=c, alpha=0.4, label=lab, rasterized=True)
    a.set(xlabel="z at XTCAVF [µm] (head +)", ylabel="δ = ΔE/E [%]", title="Longitudinal phase space at XTCAVF (truth tag)")
    a.legend(markerscale=20)
    a = ax[0, 1]
    for m, c in ((~is_w, C_D), (is_w, C_W)):
        a.scatter(1e3 * x[m][::5], 1e3 * y[m][::5], s=0.3, c=c, alpha=0.4, rasterized=True)
    for k, ls in (("midpoint", "-"), ("valley", "--")):
        a.axhline(1e3 * rec["energy"]["thr"][k], color="k", ls=ls, lw=1, label=f"energy gate ({k})")
        a.axvline(1e3 * rec["zlobe"]["thr"][k], color="m", ls=ls, lw=1, label=f"z-lobe ({k})")
    a.set(xlabel="x at DTOTR [mm] (streak axis, +22 MV)", ylabel="y at DTOTR [mm] (energy axis)",
          title="DTOTR image (truth tag)")
    a.legend(fontsize=7, loc="best")
    for a, axis, v, lab in ((ax[1, 0], "energy", y, "y at DTOTR [mm] (energy axis)"),
                            (ax[1, 1], "zlobe", x, "x at DTOTR [mm] (streak axis)")):
        lo, hi = np.percentile(v, [0.05, 99.95])
        edges = np.arange(lo, hi + PIXEL_M, PIXEL_M) * 1e3
        a.hist([1e3 * v[~is_w], 1e3 * v[is_w]], edges, weights=[q[~is_w], q[is_w]], stacked=True,
               color=[C_D, C_W], label=["drive", "witness"], histtype="stepfilled", alpha=0.7)
        a.hist(1e3 * v, edges, weights=q, histtype="step", color="k", lw=0.8, label="screen projection")
        mis, thr = rec[axis]["mis"], rec[axis]["thr"]
        for k, ls in (("midpoint", "-"), ("valley", "--"), ("optimal", ":")):
            if np.isfinite(thr[k]):
                a.axvline(1e3 * thr[k], color="k", ls=ls, lw=1.2, label=f"{k}: {100 * mis[k]:.1f}% misassigned")
        name = "Energy gate" if axis == "energy" else "z-lobe split"
        a.set(xlabel=lab, ylabel="charge / pixel [arb.]", title=f"{name}: projection (1 px = 30.5 µm)")
        a.legend(fontsize=8)
    fig.suptitle(f"{title}\nspacing {rec['spacing_um']:.0f} µm, ΔE/E {100 * rec['dE_rel']:.2f}%, "
                 f"σδ drive/witness {100 * rec['sig_d']:.2f}/{100 * rec['sig_w']:.2f}%, "
                 f"σz drive/witness {rec['sigz_d_um']:.0f}/{rec['sigz_w_um']:.0f} µm")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def summary(fig_path, recs, blocks):
    E = np.array([[r["energy"]["mis"][k] for k in ("midpoint", "valley", "optimal")] for r in recs])
    Z = np.array([[r["zlobe"]["mis"][k] for k in ("midpoint", "valley", "optimal")] for r in recs])
    sep_E = np.array([abs(r["dE_rel"]) / (r["sig_d"] + r["sig_w"]) for r in recs])
    sep_Z = np.array([abs(r["spacing_um"]) / (r["sigz_d_um"] + r["sigz_w_um"]) for r in recs])
    ctl = np.array([b == "controller" for b in blocks])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    a = ax[0]
    bins = np.linspace(0, 0.5, 51)
    a.hist(100 * E[ctl, 2], 100 * bins, alpha=0.6, label="energy gate (optimal threshold)")
    a.hist(100 * E[ctl, 0], 100 * bins, alpha=0.6, label="energy gate (midpoint)")
    a.hist(100 * Z[ctl, 0], 100 * bins, alpha=0.6, label="z-lobe (midpoint)")
    a.axvline(10, color="k", ls="--", lw=1, label="validity cut (10%)")
    a.set(xlabel="charge misassigned [%]", ylabel="beams", title=f"Controller region ({ctl.sum()} Bmad beams)")
    a.legend(fontsize=8)
    a = ax[1]
    for m, mk, lab in ((ctl, "o", "controller region"), (~ctl, "^", "wide LHS / baseline")):
        a.scatter(sep_E[m], 100 * E[m, 2], marker=mk, s=14, label=lab)
    a.axhline(10, color="k", ls="--", lw=1)
    a.set(xlabel="|ΔE| / (σE,drive + σE,witness)", ylabel="energy gate, best threshold: misassigned [%]",
          title="Energy gate is limited by overlap in E")
    a.legend(fontsize=8)
    a = ax[2]
    for m, mk, lab in ((ctl, "o", "controller region"), (~ctl, "^", "wide LHS / baseline")):
        a.scatter(sep_Z[m], 100 * Z[m, 0], marker=mk, s=14, label=lab)
    a.axhline(10, color="k", ls="--", lw=1)
    a.set(xlabel="|spacing| / (σz,drive + σz,witness)", ylabel="z-lobe, midpoint: misassigned [%]",
          title="z-lobe split is limited by overlap in z")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "xtcav_split_2026-10-14"))
    ap.add_argument("--voltage", type=float, default=22e6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(SPEC)
    e_ref, l_tdc = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
    segment, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
    set_tdc(segment, a.voltage, 0.0)
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
    recs, blocks = [], []
    for row in manifest:
        i = row["idx"]
        if not (CAMPAIGN / f"sample_{i:05d}_XTCAVF.h5").exists():
            continue
        r = analyse(i, segment, e_ref, l_tdc)
        if r is None:
            continue
        recs.append(r[0])
        blocks.append("baseline" if row.get("is_baseline_repeat") else row.get("block", "?"))
        print(i, blocks[-1], {k: round(100 * v, 1) for k, v in r[0]["energy"]["mis"].items()},
              {k: round(100 * v, 1) for k, v in r[0]["zlobe"]["mis"].items()}, flush=True)
    for r, b in zip(recs, blocks):
        r["block"] = b
    (out / "split_diagnostics.json").write_text(json.dumps(recs, indent=2, default=float))
    summary(out / "split_summary.png", recs, blocks)
    # example panels: a baseline repeat, and the median-misassignment controller-region beam
    base = next(r for r in recs if r["block"] == "baseline")
    ctl = sorted((r for r in recs if r["block"] == "controller"), key=lambda r: r["energy"]["mis"]["midpoint"])
    for tag, r in (("baseline", base), ("controller_median", ctl[len(ctl) // 2])):
        rec, d = analyse(r["i"], segment, e_ref, l_tdc)
        panel(out / f"split_{tag}.png", f"{tag.replace('_', ' ')} (sample {r['i']})", rec, d)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
