"""Beamtime control loop (direct L1 + BLEN obs, after the ops notes) for the 2026-10-17 notebook entry.
Writes lab-notebook/images/2026-10-17_control_loop_beamtime.png; reuses the 10-15 drawing helpers.

  /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/notebook_figs_2026-10-17_diagram.py
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("nb15", Path(__file__).parent / "notebook_figs_2026-10-15.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)
plt, box, arrow, canvas, INK2, OUT = nb.plt, nb.box, nb.arrow, nb.canvas, nb.INK2, nb.OUT


def control_loop():
    fig, ax = canvas(13, 5.2, 13, 5.2)
    box(ax, 0.2, 1.7, 2.3, 1.9, "Policy (BPTT)", "obs: XTCAV features, errors,\ntrend, PACT dL1 readback,\nBC11 + BC14 BLEN log-ratios,\n8-step history", fc="#eef3fb")
    box(ax, 3.2, 3.0, 2.8, 1.6, "L1: KLYS:LI11:{11,21}:PREQ", "BC11 BLEN loop OFF\nsame delta to both (absolute PV)\nbeam gain g1 ~ U(0.7, 1.3), hidden\nramp drift U(+-0.05) deg/step", new=True)
    box(ax, 3.2, 0.95, 2.8, 1.6, "L2: SCP multiknob l2_phase", "BC14 loop OFF, delta writes\n(AIDA-PVA, slc_mkb.set_mkb)\ng2 ~ U(0.35, 1.1), hidden\nmoves LI12-14 sub-boosters 1:1", new=True)
    box(ax, 6.7, 1.7, 2.4, 1.9, "Beam", "surrogate (B2-cov) for training\nBmad for closed-loop check\nreal scans for DAQ replay\nhidden injector context", fc="#f3f2ee")
    box(ax, 9.8, 2.75, 3.0, 1.55, "XTCAV +-V screen", "M = 30 images/step, deployed\nsplit + validity; scale U(0.8, 1.2)", new=True)
    box(ax, 9.8, 0.85, 3.0, 1.55, "BLEN pyrometers", "BC11: witness length (free with\nloop off); BC14: realised L2\nhidden slope scale, 2.5-3.5% noise", new=True)
    arrow(ax, (2.5, 3.0), (3.2, 3.6)); arrow(ax, (2.5, 2.3), (3.2, 1.75))
    arrow(ax, (6.0, 3.6), (6.7, 3.05), "dL1", off=(0.15, 0.12)); arrow(ax, (6.0, 1.75), (6.7, 2.25), "dL2", off=(0.15, -0.3))
    arrow(ax, (9.1, 3.0), (9.8, 3.45)); arrow(ax, (9.1, 2.3), (9.8, 1.7))
    ax.plot([11.3, 11.3, 1.35], [0.85, 0.3, 0.3], color=INK2, lw=1.2)
    ax.plot([12.8, 12.98, 12.98, 11.3], [3.5, 3.5, 0.3, 0.3], color=INK2, lw=1.2)
    arrow(ax, (1.35, 0.3), (1.35, 1.7))
    ax.text(6.3, 0.38, "measured spacing, drive sz, charge fraction, validity; BC11 / BC14 BLEN log-ratio to the episode start",
            ha="center", fontsize=7.8, color=INK2)
    ax.plot([4.6, 4.6, 1.35], [4.6, 4.85, 4.85], color=INK2, lw=1.2)
    arrow(ax, (1.35, 4.85), (1.35, 3.6))
    ax.text(3.0, 4.9, "FAST_PACT (klystron phase) readback", ha="center", fontsize=7.8, color=INK2)
    ax.text(0.2, 5.05, "Beamtime control loop: tune L1/L2 directly with the bunch-length loops off, then re-lock them at the final BLEN readings",
            fontsize=10.5, weight="bold", va="bottom")
    fig.savefig(OUT / "2026-10-17_control_loop_beamtime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    control_loop()
