"""Validation of the NEW sigma_z (bunch-length) matching objective, across the 100-300 um sweep.

For each target spacing this compares, per bunch (witness | drive):
  - surrogate sigma_z : the controller's predicted bunch length at the exported setpoint
                        (setpoints_goal*um.json -> surrogate_metrics, from transfer_setpoints)
  - Bmad sigma_z      : std(z) of the tracked PENT beam (this run's saved sample_*_PENT.h5)
  - golden reference  : golden two-bunch baseline sigma_z @ PENT (data/tightbox_v2_full/sample_06000)
  - config target     : the sigma_z target the reward pulled toward (from logs/<run>/cfg.yaml)
  - tightbox (no sigma_z obj) : the PRE-objective policy's Bmad sigma_z -- shows the drift this term fixes

sigma_z is the full RMS of the z column (meters, shown in um) -- the quantity the surrogate reward
normalizes (log10). Uses the SAME surrogate value the validation bars use, so the two figures agree.
Writes sigma_z_vs_goal.png.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/rl/bptt_gc_sigmaz/sweep_analysis/plot_sigma_z.py
"""
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup

from twobunch_s2e_rl.datagen.paths import repo_root

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent                                   # results/rl/bptt_gc_sigmaz
VAL = RUN_DIR / "bmad_validation"
SETP = RUN_DIR / "setpoints"
LOGDIR = repo_root() / "logs" / "bptt_gc_sigmaz"
GOLDEN_H5 = repo_root() / "data/tightbox_v2_full/sample_06000_PENT.h5"
TIGHTBOX_VAL = repo_root() / "results/rl/bptt_gc_tightbox/bmad_validation"   # pre-sigma_z policy (optional)
GOALS = [100, 150, 200, 250, 300]
BLUE, ORANGE, GREEN, GREY, RED = "#4c72b0", "#dd8452", "#55a868", "#8c8c8c", "#e24a4a"


def _bmad_sigz(h5):
    """(drive_sigma_z, witness_sigma_z) in meters from a PENT beam; drive=max weight, witness=min."""
    P = ParticleGroup(str(h5)); w = np.unique(P.weight)
    drive = float(P[P.weight == w[-1]].z.std())
    witness = float(P[P.weight == w[0]].z.std()) if len(w) >= 2 else np.nan
    return drive, witness


def _surr_sigz_um(g):
    """(drive, witness) surrogate sigma_z [um] from the exported setpoint's surrogate_metrics."""
    m = json.load(open(SETP / f"setpoints_goal{g}um.json"))["surrogate_metrics"]
    return m["drive_sigma_z_um"], m["witness_sigma_z_um"]


def _targets():
    """{'drive_sigma_z': m, 'witness_sigma_z': m} from the trained cfg objectives (empty if absent)."""
    cfg = yaml.safe_load(open(LOGDIR / "cfg.yaml"))
    out = {}
    for o in cfg["params"]["diff_env"].get("objectives", []):
        if o.get("key") in ("drive_sigma_z", "witness_sigma_z") and "target" in o:
            out[o["key"]] = float(o["target"])
    return out


def main():
    surr = {"drive": [], "witness": []}
    bmad = {"drive": [], "witness": []}
    tb = {"drive": [], "witness": []}                    # pre-sigma_z tightbox policy (may be missing)
    for g in GOALS:
        sd, sw = _surr_sigz_um(g); surr["drive"].append(sd * 1e-6); surr["witness"].append(sw * 1e-6)
        bd, bw = _bmad_sigz(VAL / f"sample_{g:05d}_PENT.h5"); bmad["drive"].append(bd); bmad["witness"].append(bw)
        tbf = TIGHTBOX_VAL / f"sample_{g:05d}_PENT.h5"
        if tbf.exists():
            td, tw = _bmad_sigz(tbf); tb["drive"].append(td); tb["witness"].append(tw)
        else:
            tb["drive"].append(np.nan); tb["witness"].append(np.nan)

    gold_d, gold_w = _bmad_sigz(GOLDEN_H5)
    golden = {"drive": gold_d, "witness": gold_w}
    tgt = _targets()
    tgt_key = {"drive": "drive_sigma_z", "witness": "witness_sigma_z"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, bunch in zip(axes, ("witness", "drive")):        # witness first (beam-critical, tighter target)
        um = 1e6
        ax.plot(GOALS, np.array(surr[bunch]) * um, "-o", color=BLUE, label="surrogate")
        ax.plot(GOALS, np.array(bmad[bunch]) * um, "-o", color=ORANGE, label="Bmad (truth)")
        if np.isfinite(tb[bunch]).any():
            ax.plot(GOALS, np.array(tb[bunch]) * um, "--o", color=GREY, ms=4, lw=1.2, alpha=0.8,
                    label="Bmad, no σ_z obj (tightbox)")
        ax.axhline(golden[bunch] * um, color=GREEN, ls="--", lw=1.6,
                   label=f"golden {golden[bunch]*um:.1f} µm")
        if tgt_key[bunch] in tgt:
            ax.axhline(tgt[tgt_key[bunch]] * um, color=RED, ls=":", lw=1.8,
                       label=f"target {tgt[tgt_key[bunch]]*um:.1f} µm")
        ax.set_xlabel("target spacing [µm]"); ax.set_ylabel(f"{bunch} σ_z [µm]")
        ax.set_title(f"{bunch} bunch length", fontsize=12); ax.legend(fontsize=8.5)
        ax.set_ylim(bottom=0)
    fig.suptitle("Bunch-length (σ_z) matching: surrogate vs Bmad vs golden/target across the sweep",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "sigma_z_vs_goal.png", dpi=130); plt.close(fig)
    print(f"wrote sigma_z_vs_goal.png to {HERE}")
    for bunch in ("witness", "drive"):
        print(f"  {bunch:8s} golden={golden[bunch]*1e6:5.1f}  target="
              f"{tgt.get(tgt_key[bunch], float('nan'))*1e6:5.1f}  "
              f"Bmad={[round(v*1e6,1) for v in bmad[bunch]]}  "
              f"surr={[round(v*1e6,1) for v in surr[bunch]]} µm")


if __name__ == "__main__":
    main()
