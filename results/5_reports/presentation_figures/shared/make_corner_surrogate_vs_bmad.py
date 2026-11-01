#!/usr/bin/env python
"""Dark-theme corner plot: NF-surrogate PREDICTION vs real Bmad TRUTH at the hand-tuned
two-bunch baseline (the 2024-10-14 golden working point).

Demo counterpart to make_corner_dark.py (which compares two *physical* set-points, both from
Bmad). Here the contrast is model-vs-truth at a single set-point:

  truth      -> results/rl/baseline/openloop/clouds.npz  (bmad_drive / bmad_witness)
                Bmad/Tao L0AFEND->PENT at the golden config, 100k macroparticles, CSR +
                transverse wakes ON (the campaign convention the surrogate was trained on),
                subsampled to 2048 via surrogate.preprocess._subsample.
  prediction -> trained/twobunch_combined_ft (the latest 26-knob conditional RealNVP/RQS flow),
                sampled at the golden 26-D knob vector normalized in the model's own
                union ("tightbox+expanded") frame.

The 26 knobs are set to their *golden* baselines (sweep_params `expanded` set), which is exactly
the point the cached Bmad run sits at: that run overrode only the 8 longitudinal knobs on top of
setLattice_configs/2024-10-14_twoBunch_baseline.yml, leaving the 18 transverse knobs (FF quads,
FF kickers, BC20 sextupole movers) at their golden tuned values.

Both clouds live in the training frame: COORD_KEYS = (x,y,z,px,py,pz), positions [m],
momenta [eV/c]; displayed as mm / MeV/c / GeV/c.

Colour encodes SOURCE only -- cool azure = Bmad truth, warm coral = surrogate -- with both
bunches of a source sharing its colour (the same cool/warm contrast make_corner_dark.py uses for
its two set-points). The drive/witness split needs no colour: the bunches separate visibly in
z, px and py. The per-metric surrogate-vs-truth table is written to JSON, not drawn on the axes.

Run (GPU 0 is the free A100; flow sampling is ~ms):
  cd twobunch-s2e-rl && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/presentation_figures/make_corner_surrogate_vs_bmad.py

Outputs (next to this script):
  corner_surrogate_vs_bmad_baseline_dark.png
  corner_surrogate_vs_bmad_baseline_metrics.json   (full metric table + knob vector)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch

HERE = Path(__file__).resolve().parent
# repo root by marker, not by index -- survives moving this script between figure subfolders
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
BMAD_NPZ = ROOT / "results/rl/baseline/openloop/clouds.npz"
SANITY_JSON = ROOT / "results/rl/baseline/openloop/eval_sanity.json"
NORM_JSON = ROOT / "processed/twobunch_combined_norm.json"
CKPT = ROOT / "trained/twobunch_combined_ft/checkpoints/best-*.ckpt"
OUT_PNG = HERE / "corner_surrogate_vs_bmad_baseline_dark.png"
OUT_JSON = HERE / "corner_surrogate_vs_bmad_baseline_metrics.json"

# (col, label, unit, display-scale) on (x,y,z,px,py,pz)
COORDS = [(0, "x", "mm", 1e3), (1, "y", "mm", 1e3), (2, "z", "mm", 1e3),
          (3, "px", "MeV/c", 1e-6), (4, "py", "MeV/c", 1e-6), (5, "pz", "GeV/c", 1e-9)]

# ---- presentation theme (matches make_corner_dark.py / nf_surrogate_architecture.tex) ----
BG, FG = "#000000", "#E8EAEE"
BMAD_C = "#5B8FF9"    # Bmad truth      -- azure     (cool)
SURR_C = "#FF5D62"    # surrogate model -- coral-red (warm)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG,
})


def resolve(pattern: str) -> str:
    hits = sorted(glob.glob(str(pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def baseline_knobs_norm(flow) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Golden 26-D knob vector, normalized in the flow's own union frame.

    Keys/bounds come from the preprocess norm.json used to train the checkpoint and are
    cross-checked against the buffers stored in the checkpoint itself.
    """
    norm = json.load(open(NORM_JSON))
    keys = list(norm["knob_keys"])
    lo = np.asarray(norm["knob_low"], np.float64)
    hi = np.asarray(norm["knob_high"], np.float64)

    ck_lo = flow.knob_low.detach().cpu().numpy().astype(np.float64)
    ck_hi = flow.knob_high.detach().cpu().numpy().astype(np.float64)
    assert flow.hparams.condition_dim == len(keys), \
        f"checkpoint condition_dim {flow.hparams.condition_dim} != {len(keys)} knob_keys"
    np.testing.assert_allclose(lo, ck_lo, rtol=1e-6, err_msg="knob_low frame mismatch")
    np.testing.assert_allclose(hi, ck_hi, rtol=1e-6, err_msg="knob_high frame mismatch")

    # `expanded` baselines == the golden 2024-10-14 working point for all 26 knobs (the
    # `tightbox` set re-centres FF quads / sextupole strengths on the 15 cm matched point, so
    # resolve_sweep_set("tightbox+expanded") would hand back the WRONG baseline here).
    exp_keys, _, _, base = resolve_sweep_set("expanded")
    assert exp_keys == keys, "sweep-set key order differs from the training frame"
    phys = np.array([base[k] for k in keys], np.float64)
    knorm = (phys - lo) / (hi - lo)
    assert np.all((knorm >= -1e-9) & (knorm <= 1 + 1e-9)), \
        f"golden baseline outside the training box: {dict(zip(keys, knorm.round(3)))}"
    return phys, knorm.astype(np.float32), keys


def cloud_metrics(drive: torch.Tensor, witness: torch.Tensor) -> dict[str, float]:
    """Same estimators the surrogate's `observables` (and the RL reward) use, applied to a
    (1,N,6) drive/witness pair -- so surrogate and Bmad numbers are apples-to-apples.
    Note these are FULL-projected normalized emittances (properties.norm_emits), not the
    getBeamSpecs 90%-core values.
    """
    out = {}
    for name, parts in (("drive", drive), ("witness", witness)):
        for k, v in per_bunch(parts).items():
            out[f"{name}_{k}"] = float(v.squeeze())
    for k, v in inter_bunch(drive, witness).items():
        out[k] = float(v.squeeze())
    return out


def shared_lims(series):
    lims = {}
    for ci, _, _, sc in COORDS:
        vals = np.concatenate([a[:, ci] * sc for _, _, a in series])
        lo, hi = np.percentile(vals, [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-12)
        lims[ci] = (lo - pad, hi + pad)
    return lims


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", default=str(CKPT))
    ap.add_argument("--n-particles", type=int, default=2048,
                    help="surrogate cloud size (2048 = the cached Bmad subsample size)")
    ap.add_argument("--n-draws", type=int, default=8,
                    help="extra flow draws used only for the metric sampling scatter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ck = resolve(args.flow_ckpt)
    flow = TwoBunchFlow.load_from_checkpoint(ck, map_location=device).to(device).eval()
    for p in flow.parameters():
        p.requires_grad_(False)

    phys, knorm, keys = baseline_knobs_norm(flow)
    kt = torch.tensor(knorm, device=device).unsqueeze(0)

    # ---- Bmad truth at the same golden point (cached; 100k macro, CSR + wakes) ----
    bl = np.load(BMAD_NPZ)
    b_drive, b_witness = bl["bmad_drive"], bl["bmad_witness"]
    bmad_m = cloud_metrics(torch.tensor(b_drive, device=device).unsqueeze(0),
                           torch.tensor(b_witness, device=device).unsqueeze(0))

    # ---- surrogate prediction: the plotted draw, plus n_draws for sampling scatter ----
    torch.manual_seed(args.seed)
    with torch.no_grad():
        s_drive_t = flow.sample_bunch(kt, 0, args.n_particles)
        s_witness_t = flow.sample_bunch(kt, 1, args.n_particles)
        surr_m = cloud_metrics(s_drive_t, s_witness_t)
        reps = [cloud_metrics(flow.sample_bunch(kt, 0, args.n_particles),
                              flow.sample_bunch(kt, 1, args.n_particles))
                for _ in range(args.n_draws)]
        p_surv, t_d, t_w = (float(v) for v in flow.feasibility(kt))
    draw_std = {k: float(np.std([r[k] for r in reps])) for k in surr_m}
    s_drive = s_drive_t.squeeze(0).cpu().numpy()
    s_witness = s_witness_t.squeeze(0).cpu().numpy()

    # one colour per SOURCE: both bunches of a source share it (the drive/witness split is
    # already unmistakable in the clouds themselves -- they separate in z, px, py).
    series = [("Bmad (truth)", BMAD_C, np.vstack([b_drive, b_witness])),
              ("surrogate", SURR_C, np.vstack([s_drive, s_witness]))]

    # ---- report ----
    print(f"checkpoint : {ck}")
    print(f"knobs      : 26-D golden baseline (sweep_params `expanded` baselines)")
    print(f"Bmad truth : {BMAD_NPZ.relative_to(ROOT)} "
          f"(n_drive={len(b_drive)}, n_witness={len(b_witness)})")
    print(f"\n{'metric':26s}{'surrogate':>14s}{'Bmad':>14s}{'err %':>9s}")
    for key in sorted(set(surr_m) & set(bmad_m)):
        s, b = surr_m[key], bmad_m[key]
        err = 100.0 * (s - b) / b if b != 0 else float("nan")
        print(f"{key:26s}{s:14.6g}{b:14.6g}{err:+9.1f}")
    sanity = json.load(open(SANITY_JSON))["gap"]
    print("\nfeasibility heads (surrogate) vs Bmad charge fractions "
          "(100k-particle truth, eval_sanity.json):")
    print(f"  T_drive   {t_d:.4f}  vs  {sanity['T_drive']['bmad_med']:.4f}")
    print(f"  T_witness {t_w:.4f}  vs  {sanity['T_witness']['bmad_med']:.4f}"
          f"   (p_surv={p_surv:.4f})")

    json.dump({"checkpoint": str(Path(ck).relative_to(ROOT)),
               "bmad_source": str(BMAD_NPZ.relative_to(ROOT)),
               "knobs_phys": dict(zip(keys, phys.tolist())),
               "knobs_norm": dict(zip(keys, knorm.astype(float).tolist())),
               "n_particles": args.n_particles,
               "surrogate": surr_m, "surrogate_draw_std": draw_std, "bmad": bmad_m,
               "surrogate_feasibility": {"p_surv": p_surv, "T_drive": t_d, "T_witness": t_w},
               "bmad_transmission_100k": {"T_drive": sanity["T_drive"]["bmad_med"],
                                          "T_witness": sanity["T_witness"]["bmad_med"]}},
              open(OUT_JSON, "w"), indent=2)

    # ---- figure ----
    nc = len(COORDS)
    fig, axes = plt.subplots(nc, nc, figsize=(16.5, 16.5))
    lims = shared_lims(series)

    for r in range(nc):
        ci_r, lab_r, u_r, sc_r = COORDS[r]
        for c in range(nc):
            ci_c, lab_c, u_c, sc_c = COORDS[c]
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if c == r:                                   # diagonal: 1-D step density
                bins = np.linspace(*lims[ci_r], 46)
                for _, col, a in series:
                    ax.hist(a[:, ci_r] * sc_r, bins=bins, color=col, histtype="step",
                            lw=2.2, density=True)
                ax.set_xlim(*lims[ci_r])
                ax.set_yticks([])
            else:                                        # lower triangle: col_c (x) vs col_r (y)
                for _, col, a in series:
                    ax.scatter(a[:, ci_c] * sc_c, a[:, ci_r] * sc_r, s=3, alpha=0.35,
                               color=col, edgecolors="none")
                ax.set_xlim(*lims[ci_c])
                ax.set_ylim(*lims[ci_r])
                ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            for sp in ax.spines.values():
                sp.set_color(FG)
            ax.tick_params(labelsize=15, length=4)
            if r == nc - 1:
                ax.set_xlabel(f"{lab_c} [{u_c}]", fontsize=22, labelpad=8)
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(f"{lab_r} [{u_r}]", fontsize=22, labelpad=8)
            elif c != 0:
                ax.set_yticklabels([])

    handles = [plt.Line2D([], [], marker="o", ls="", ms=15, color=col, label=nm)
               for nm, col, _ in series]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.94),
                     fontsize=24, framealpha=0.0, handletextpad=0.4, borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(FG)

    fig.suptitle("Full 6D phase space at PENT, hand-tuned two-bunch baseline\n"
                 "Bmad truth  vs  normalizing-flow surrogate prediction",
                 fontsize=27, fontweight="bold", color="#FFFFFF", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.055,
                        wspace=0.08, hspace=0.08)
    fig.savefig(OUT_PNG, dpi=100)
    print(f"\nwrote {OUT_PNG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
