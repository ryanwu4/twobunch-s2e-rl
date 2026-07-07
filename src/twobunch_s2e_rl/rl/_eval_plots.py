"""Shared phase-space sweep plots (corner + combined LPS) for the eval scripts.

Generic over (N,6) clouds in (x,y,z,px,py,pz) [m, eV/c] -- works on surrogate-sampled OR Bmad-
tracked bunches. Renders per-goal static PNGs and, across a goal sweep, animated GIFs (PIL; no
imageio). A `frame` is a dict: {slug, title, drive: (N,6)|None, witness: (N,6)|None}.

No torch / no Bmad imports here, so both eval.py (surrogate) and eval_bmad.py (ground truth) can
share it without dragging in each other's heavy deps.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 6D phase-space coords on (x,y,z,px,py,pz): (col, label, unit, scale-to-display)
_COORDS = [(0, "x", "mm", 1e3), (1, "y", "mm", 1e3), (2, "z", "mm", 1e3),
           (3, "px", "MeV/c", 1e-6), (4, "py", "MeV/c", 1e-6), (5, "pz", "GeV/c", 1e-9)]
_DRIVE_C, _WITNESS_C = "#1f77b4", "#d62728"


def _fig_to_rgba(fig):
    """Render an Agg figure to a uint8 (H,W,4) array. Frames keep a fixed pixel size (figsize*dpi),
    so a stack of them can be written straight to a GIF (no bbox_inches='tight', which would vary)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4).copy()


def _save_gif(frames, path, fps=2.0):
    from PIL import Image
    imgs = [Image.fromarray(f).convert("RGB") for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000.0 / max(fps, 1e-3)), loop=0)


def _coord_lims(frames):
    """Per-coordinate (display-unit) limits pooled over BOTH bunches across ALL frames, so every
    frame shares axes -- the witness then visibly slides relative to the drive as the goal sweeps."""
    lims = {}
    for ci, _, _, sc in _COORDS:
        vals = [fr[b][:, ci] * sc for fr in frames for b in ("drive", "witness")
                if fr.get(b) is not None and len(fr[b]) > 1]
        if vals:
            lo, hi = np.percentile(np.concatenate(vals), [0.5, 99.5])
            pad = 0.05 * (hi - lo + 1e-12)
            lims[ci] = (lo - pad, hi + pad)
        else:
            lims[ci] = (-1.0, 1.0)
    return lims


def _bunches(fr):
    return (("drive", fr.get("drive"), _DRIVE_C), ("witness", fr.get("witness"), _WITNESS_C))


def _corner_figure(fr, lims):
    """Full 6x6 corner of BOTH bunches overlaid (drive blue, witness red): lower-triangle scatter,
    diagonal 1-D density. Shared limits across frames (passed in)."""
    nc = len(_COORDS)
    fig, axes = plt.subplots(nc, nc, figsize=(13, 13))
    for r in range(nc):
        ci_r, lab_r, u_r, sc_r = _COORDS[r]
        for c in range(nc):
            ci_c, lab_c, u_c, sc_c = _COORDS[c]
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if c == r:                                   # diagonal: 1-D density of this coord
                for _, a, col in _bunches(fr):
                    if a is not None and len(a) > 1:
                        ax.hist(a[:, ci_r] * sc_r, bins=40, color=col, alpha=0.5, density=True)
                ax.set_xlim(*lims[ci_r])
                ax.set_yticks([])
            else:                                        # lower triangle: col_c (x) vs col_r (y)
                for _, a, col in _bunches(fr):
                    if a is not None and len(a) > 1:
                        ax.scatter(a[:, ci_c] * sc_c, a[:, ci_r] * sc_r, s=2, alpha=0.25,
                                   color=col, edgecolors="none")
                ax.set_xlim(*lims[ci_c])
                ax.set_ylim(*lims[ci_r])
            ax.tick_params(labelsize=7)
            if r == nc - 1:
                ax.set_xlabel(f"{lab_c} [{u_c}]", fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(f"{lab_r} [{u_r}]", fontsize=9)
            elif c != 0:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for n, _, c in _bunches(fr)]
    fig.legend(handles=handles, loc="upper right", fontsize=12, markerscale=1.5)
    fig.suptitle(fr["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def _lps_figure(fr, zlim, pzlim):
    """Longitudinal phase space (z vs pz) with BOTH bunches on the same axes; dashed lines mark the
    per-bunch z-centroids so their separation == the (signed) bunch spacing. A top marginal (sharing
    the z-axis) shows the longitudinal particle-density profile -- two peaks split by the spacing."""
    fig, (axh, ax) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": (1, 5), "hspace": 0.05})
    zbins = np.linspace(zlim[0], zlim[1], 80)            # shared bins so both bunches are comparable
    for nm, a, col in _bunches(fr):
        if a is not None and len(a) > 1:
            z = a[:, 2] * 1e3
            ax.scatter(z, a[:, 5] * 1e-9, s=3, alpha=0.3, color=col,
                       edgecolors="none", label=f"{nm} (n={len(a)})")
            ax.axvline(float(np.mean(z)), color=col, ls="--", lw=1.2, alpha=0.8)
            axh.hist(z, bins=zbins, color=col, alpha=0.5, density=True)
    axh.set_ylabel("density", fontsize=9)
    axh.set_yticks([])
    axh.tick_params(labelbottom=False)
    axh.set_title(fr["title"], fontsize=12)
    ax.set(xlabel="z [mm]", ylabel="pz [GeV/c]", xlim=zlim, ylim=pzlim)
    ax.legend(fontsize=9, markerscale=2, loc="upper left")
    # explicit margins (not tight_layout) -> identical geometry every frame, no shared-axis warning
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.09)
    return fig


_TRAJ_C = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def plot_closedloop_trajectory(trajectories, out_png, spacing_target_um=200.0, title=None):
    """Closed-loop Bmad trajectories vs step: spacing, per-bunch survival, reward. `trajectories`
    maps a label -> dict of equal-length lists {step, spacing_um, T_drive, T_witness, reward};
    multiple labels overlay (e.g. bptt_dr vs shac_dr). Returns the written path."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for i, (label, tr) in enumerate(trajectories.items()):
        c = _TRAJ_C[i % len(_TRAJ_C)]
        s = tr["step"]
        axes[0].plot(s, tr["spacing_um"], "-o", ms=3, color=c, label=label)
        axes[1].plot(s, tr["T_drive"], "-o", ms=3, color=c, label=f"{label} drive")
        axes[1].plot(s, tr["T_witness"], "--s", ms=3, color=c, label=f"{label} witness")
        axes[2].plot(s, tr["reward"], "-o", ms=3, color=c, label=label)
    if spacing_target_um is not None:
        axes[0].axhline(spacing_target_um, color="k", ls=":", lw=1,
                        label=f"target {spacing_target_um:.0f} um")
    axes[1].axhline(0.9, color="gray", ls=":", lw=0.8, label="T=0.9")
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("bunch spacing [um]")
    axes[1].set_ylabel("survival fraction T")
    axes[2].set_ylabel("reward")
    axes[2].set_xlabel("closed-loop step")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98) if title else (0, 0, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_sweep_plots(frames, artdir, name, fps=2.0):
    """Per-goal static corner + combined-LPS PNGs, and (for >=2 goals) GIFs animating over the sweep.
    Returns the GIF paths written (empty if <2 frames)."""
    frames = [fr for fr in frames if fr.get("drive") is not None or fr.get("witness") is not None]
    if not frames:
        return []
    lims = _coord_lims(frames)
    corner_rgba, lps_rgba = [], []
    for fr in frames:
        cf = _corner_figure(fr, lims)
        corner_rgba.append(_fig_to_rgba(cf))
        cf.savefig(artdir / f"corner_{fr['slug']}.png", dpi=cf.dpi)
        plt.close(cf)
        lf = _lps_figure(fr, lims[2], lims[5])
        lps_rgba.append(_fig_to_rgba(lf))
        lf.savefig(artdir / f"lps_{fr['slug']}.png", dpi=lf.dpi)
        plt.close(lf)
    written = []
    if len(frames) >= 2:                                 # a GIF needs >=2 sweep points
        for kind, stack in (("corner", corner_rgba), ("lps", lps_rgba)):
            gif = artdir / (f"{kind}_{name}_sweep.gif" if name else f"{kind}_sweep.gif")
            _save_gif(stack, gif, fps=fps)
            written.append(gif)
            print(f"  wrote {gif}")
    return written
