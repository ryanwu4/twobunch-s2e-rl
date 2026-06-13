"""Diagnostic plots for a trained two-bunch flow (true vs flow, per bunch).

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.diagnostics \
      --ckpt "trained/twobunch_flow/checkpoints/best-*.ckpt"

Writes to artifacts/surrogate/diagnostics/:
  corner_drive.png / corner_witness.png  full 6D lower-triangle phase space (one sample)
  slices.png                             x-y, x-px, y-py, z-pz for a couple samples x both bunches
  beam_matrix.png                        6x6 correlation matrices (true / flow / diff), per sample x bunch
  knob_response.png                      learned differentiable surface: observables vs each knob
  feasibility.png                        viability reliability + transmission calibration
  dispersion_ratio.png                   flow/true per-coord sigma ratio (over- / under-dispersion)
"""
from __future__ import annotations

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..datagen.paths import repo_root
from ..datagen.sweep_params import PARAM_KEYS
from .dataset import TwoBunchFlowDataset
from .model import TwoBunchFlow

# per-coordinate display scaling (cols: x,y,z [m]; px,py,pz [eV/c])
SCALE = np.array([1e6, 1e6, 1e6, 1e-6, 1e-6, 1e-6])
LABELS = ["x [µm]", "y [µm]", "z [µm]", "px [MeV/c]", "py [MeV/c]", "Δpz [MeV/c]"]
SHORT = ["x", "y", "z", "px", "py", "pz"]
PAIRS = [(0, 1, "x–y"), (0, 3, "x–px"), (1, 4, "y–py"), (2, 5, "z–pz")]
C_T, C_F = "#1f77b4", "#d62728"  # true, flow


def _disp(arr, pz_ref):
    a = arr.copy()
    a[:, 5] = a[:, 5] - pz_ref           # center pz by the true bunch mean -> show spread/offset
    return a * SCALE


def _sub(n, k, seed):
    return np.random.default_rng(seed).choice(n, min(k, n), replace=False)


class Bundle:
    """Holds the loaded model + dataset and produces physical clouds per row/bunch."""

    def __init__(self, ckpt, processed):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = TwoBunchFlow.load_from_checkpoint(ckpt, map_location=self.dev).eval().to(self.dev)
        self.ds = TwoBunchFlowDataset(processed)
        nm = self.ds.norm
        self.mean = {0: np.array(nm["drive_mean"]), 1: np.array(nm["witness_mean"])}
        self.std = {0: np.array(nm["drive_std"]), 1: np.array(nm["witness_std"])}

    def true(self, row, k):
        x = (self.ds.drive if k == 0 else self.ds.witness)[row].numpy()
        return x * self.std[k] + self.mean[k]

    @torch.no_grad()
    def flow(self, row, k, n=2048):
        kn = self.ds.knobs[row:row + 1].to(self.dev)
        return self.m.sample_bunch(kn, k, n)[0].cpu().numpy()

    @torch.no_grad()
    def observables(self, knobs_norm, n=512):
        return self.m.observables(torch.as_tensor(knobs_norm, dtype=torch.float32, device=self.dev), n=n)


# ---- corner (full lower triangle) -------------------------------------------
def corner(true, flow, title, path):
    pz = true[:, 5].mean()
    T, F = _disp(true, pz), _disp(flow, pz)
    ti, fi = _sub(len(T), 1500, 0), _sub(len(F), 1500, 1)
    fig, ax = plt.subplots(6, 6, figsize=(15, 15))
    for i in range(6):
        for j in range(6):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                lo = min(T[:, i].min(), F[:, i].min()); hi = max(T[:, i].max(), F[:, i].max())
                b = np.linspace(lo, hi, 45)
                a.hist(T[:, i], bins=b, histtype="step", color=C_T, density=True, lw=1.2)
                a.hist(F[:, i], bins=b, histtype="step", color=C_F, density=True, lw=1.2)
            else:
                a.scatter(T[ti, j], T[ti, i], s=2, alpha=0.22, color=C_T, edgecolors="none")
                a.scatter(F[fi, j], F[fi, i], s=2, alpha=0.22, color=C_F, edgecolors="none")
            if i == 5:
                a.set_xlabel(LABELS[j], fontsize=8)
            if j == 0 and i != 0:
                a.set_ylabel(LABELS[i], fontsize=8)
            a.tick_params(labelsize=6)
    fig.legend(handles=[plt.Line2D([], [], color=C_T, label="true"),
                        plt.Line2D([], [], color=C_F, label="flow")],
               loc="upper right", fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


# ---- important slices for a couple samples ----------------------------------
def slices(items, path):
    nr = len(items)
    fig, ax = plt.subplots(nr, 4, figsize=(16, 3.6 * nr))
    ax = np.atleast_2d(ax)
    for r, (true, flow, lab) in enumerate(items):
        pz = true[:, 5].mean(); T, F = _disp(true, pz), _disp(flow, pz)
        ti, fi = _sub(len(T), 1200, 0), _sub(len(F), 1200, 1)
        for c, (i, j, nm) in enumerate(PAIRS):
            a = ax[r, c]
            a.scatter(T[ti, i], T[ti, j], s=3, alpha=0.25, color=C_T, edgecolors="none")
            a.scatter(F[fi, i], F[fi, j], s=3, alpha=0.25, color=C_F, edgecolors="none")
            a.set_xlabel(LABELS[i], fontsize=8); a.tick_params(labelsize=7)
            a.set_ylabel((lab + "\n" if c == 0 else "") + LABELS[j], fontsize=8)
            if r == 0:
                a.set_title(nm, fontsize=11)
    fig.legend(handles=[plt.Line2D([], [], color=C_T, label="true"),
                        plt.Line2D([], [], color=C_F, label="flow")],
               loc="upper right", fontsize=11)
    fig.suptitle("Phase-space slices — true vs flow (rows: sample × bunch)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


# ---- beam (correlation) matrices --------------------------------------------
def beam_matrix(items, path):
    nr = len(items)
    fig, ax = plt.subplots(nr, 3, figsize=(13, 4 * nr))
    ax = np.atleast_2d(ax)
    for r, (true, flow, lab) in enumerate(items):
        Ct, Cf = np.corrcoef(true.T), np.corrcoef(flow.T)
        for c, (M, t, cmap, vlim) in enumerate([
            (Ct, "true corr", "coolwarm", 1), (Cf, "flow corr", "coolwarm", 1),
            (Cf - Ct, "flow − true", "PuOr", 0.3)]):
            a = ax[r, c]
            im = a.imshow(M, vmin=-vlim, vmax=vlim, cmap=cmap)
            a.set_xticks(range(6)); a.set_yticks(range(6))
            a.set_xticklabels(SHORT, fontsize=7); a.set_yticklabels(SHORT, fontsize=7)
            a.set_title(f"{lab} — {t}", fontsize=10)
            fig.colorbar(im, ax=a, fraction=0.046, shrink=0.8)
    fig.suptitle("Phase-space correlation matrices (true / flow / difference)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


# ---- knob -> observable response (the learned differentiable surface) -------
def knob_response(bundle, path, ngrid=40):
    med = bundle.ds.knobs.median(0).values.numpy()
    grid = np.linspace(0.02, 0.98, ngrid)
    obs_defs = [("witness_norm_emit_x", 1e6, "witness ε_n,x [µm]"),
                ("bunch_spacing", 1e6, "bunch spacing [µm]"),
                ("transverse_offset", 1e6, "transv offset [µm]"),
                ("p_surv", 1.0, "witness p_surv")]
    # build all (knob, gridpoint) conditions in one batch
    K = np.tile(med, (len(PARAM_KEYS) * ngrid, 1))
    for jk in range(len(PARAM_KEYS)):
        K[jk * ngrid:(jk + 1) * ngrid, jk] = grid
    out = bundle.observables(K, n=512)
    fig, ax = plt.subplots(len(obs_defs), len(PARAM_KEYS), figsize=(2.1 * len(PARAM_KEYS), 9), sharex=True)
    for oi, (okey, sc, olab) in enumerate(obs_defs):
        y = out[okey].cpu().numpy() * sc
        for jk, kk in enumerate(PARAM_KEYS):
            a = ax[oi, jk]
            seg = y[jk * ngrid:(jk + 1) * ngrid]
            a.plot(grid, seg, color="#333", lw=1.4)
            a.axvline(med[jk], color="#dd8452", ls=":", lw=1)
            a.tick_params(labelsize=6)
            if oi == 0:
                a.set_title(kk, fontsize=7, rotation=20)
            if jk == 0:
                a.set_ylabel(olab, fontsize=8)
            if oi == len(obs_defs) - 1:
                a.set_xlabel("knob (norm)", fontsize=7)
    fig.suptitle("Surrogate response: observables vs each knob (others at median; dotted = median)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


# ---- feasibility calibration ------------------------------------------------
@torch.no_grad()
def feasibility(bundle, path):
    ds = bundle.ds
    knobs = ds.knobs.to(bundle.dev)
    p_surv, t_d, t_w = bundle.m.feasibility(knobs)
    p_surv = p_surv.cpu().numpy(); t_d = t_d.cpu().numpy(); t_w = t_w.cpu().numpy()
    viable = ds.witness_viable.numpy().astype(bool)
    dp = ds.drive_present.numpy().astype(bool); wv = viable
    dfrac = ds.drive_frac.numpy(); wfrac = ds.witness_frac.numpy()

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # reliability
    bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(p_surv, bins) - 1, 0, 9)
    obs = np.array([viable[bi == b].mean() if (bi == b).any() else np.nan for b in range(10)])
    cen = 0.5 * (bins[:-1] + bins[1:])
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 0].plot(cen, obs, "o-", color="#55a868")
    ax[0, 0].set_xlabel("predicted p_surv"); ax[0, 0].set_ylabel("observed viable fraction")
    ax[0, 0].set_title("Witness-viability reliability")
    # separation hist
    ax[0, 1].hist(p_surv[viable], bins=30, alpha=0.7, color="#55a868", label="viable")
    ax[0, 1].hist(p_surv[~viable], bins=30, alpha=0.7, color="#c44e52", label="destroyed")
    ax[0, 1].set_xlabel("predicted p_surv"); ax[0, 1].set_ylabel("count")
    ax[0, 1].set_title("p_surv by true class"); ax[0, 1].legend(fontsize=9)
    # transmission scatters
    ax[1, 0].scatter(dfrac[dp], t_d[dp], s=4, alpha=0.3, color=C_T)
    ax[1, 0].plot([0, 1.05], [0, 1.05], "k--", lw=1)
    ax[1, 0].set_xlabel("true drive frac"); ax[1, 0].set_ylabel("predicted T_drive")
    ax[1, 0].set_title("Drive transmission")
    ax[1, 1].scatter(wfrac[wv], t_w[wv], s=4, alpha=0.3, color=C_F)
    ax[1, 1].plot([0, 1.05], [0, 1.05], "k--", lw=1)
    ax[1, 1].set_xlabel("true witness frac"); ax[1, 1].set_ylabel("predicted T_witness")
    ax[1, 1].set_title("Witness transmission (viable)")
    fig.suptitle("Feasibility heads — calibration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


# ---- per-coord dispersion ratio (over-/under-dispersion diagnosis) ----------
def dispersion_ratio(bundle, path, n_rows=300):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for bi, (k, name) in enumerate([(0, "drive"), (1, "witness")]):
        mask = (bundle.ds.drive_density if k == 0 else bundle.ds.witness_density).numpy().astype(bool)
        rows = np.where(mask)[0]
        rows = rng.choice(rows, min(n_rows, len(rows)), replace=False)
        ratios = []
        for r in rows:
            t, f = bundle.true(r, k), bundle.flow(r, k, n=2048)
            ratios.append(f.std(0) / np.maximum(t.std(0), 1e-30))
        ratios = np.array(ratios)  # (R,6)
        ax[bi].boxplot(ratios, tick_labels=SHORT, showfliers=False)
        ax[bi].axhline(1.0, color="#dd8452", ls="--", lw=1)
        ax[bi].set_title(f"{name}: flow/true σ per coord (n={len(rows)})")
        ax[bi].set_ylabel("σ_flow / σ_true"); ax[bi].tick_params(labelsize=9)
    fig.suptitle("Per-coordinate dispersion ratio (1.0 = matched; <1 under-disperses)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--processed", default=str(repo_root() / "processed" / "twobunch_flow.h5"))
    ap.add_argument("--out", default="artifacts/surrogate/diagnostics")
    ap.add_argument("--samples", type=int, nargs="+", default=[693, 4778],
                    help="processed-row indices (default: an intact + a scraped witness)")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    b = Bundle(ckpt, args.processed)

    rep = args.samples[0]
    corner(b.true(rep, 0), b.flow(rep, 0), f"Drive phase space — sample row {rep}", str(out / "corner_drive.png"))
    corner(b.true(rep, 1), b.flow(rep, 1), f"Witness phase space — sample row {rep}", str(out / "corner_witness.png"))

    items = []
    for row in args.samples:
        wf = float(b.ds.witness_frac[row])
        items.append((b.true(row, 0), b.flow(row, 0), f"row {row} drive"))
        items.append((b.true(row, 1), b.flow(row, 1), f"row {row} witness (frac {wf:.2f})"))
    slices(items, str(out / "slices.png"))
    beam_matrix(items, str(out / "beam_matrix.png"))

    knob_response(b, str(out / "knob_response.png"))
    feasibility(b, str(out / "feasibility.png"))
    dispersion_ratio(b, str(out / "dispersion_ratio.png"))
    print("done:", out)


if __name__ == "__main__":
    main()
