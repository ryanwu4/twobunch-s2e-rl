"""Gradient-check figure (2026-10-19): surrogate vs Bmad d/dL1 and d/dL2 at the 12 points of
scripts/gradient_check_2026-10-18.py, for the beam (cloud) and the screen readings. Spacing in um/deg; sigma_z as
d ln sigma_z / d deg. Surrogate = spacing-head flow, autograd (B2-cov is the same for sigma_z, within 3% for spacing).
Output results/2_surrogate_screen/gradient_check_2026-10-18/gradient_parity.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = "results/2_surrogate_screen/gradient_check_2026-10-18"
B, S = np.load(f"{D}/bmad.npz"), np.load(f"{D}/surr.npz")
lab = B["lab"]
c = lab[:, 2] < 0
INJ = ["golden", "held-out DAQ-like (lhsloc102)", "held-out full-box (lhs123)"]
COL = ["#2a6fdb", "#e8590c", "#2b8a3e"]
fig, axs = plt.subplots(2, 2, figsize=(11, 10))
panels = [("bunch_spacing_cloud", "beam spacing", "µm/deg"), ("drive_sigma_z_cloud", "beam drive σz", "d ln σz / deg"),
          ("bunch_spacing", "screen spacing", "µm/deg"), ("drive_sigma_z", "screen drive σz", "d ln σz / deg")]
for ax, (k, title, unit) in zip(axs.flat, panels):
    b = S[f"bmad|{k}"][..., 0]                                   # (3 inj, 4 lin, 2 ctrl), Bmad FD at h = 0.3 deg
    s = S[f"spacing head|ag|{k}"].reshape(3, 4, 2)
    if "sigma_z" in k:
        c0 = B[k][c].reshape(3, 4)[..., None]
        b, s = b / c0, s / c0
    else:
        b, s = b * 1e6, s * 1e6
    lim = np.nanmax(np.abs(np.r_[b.ravel(), s.ravel()])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    for i in range(3):
        ax.scatter(b[i, :, 0], s[i, :, 0], marker="o", s=55, color=COL[i], label=f"{INJ[i]}, d/dL1" if ax is axs[0, 0] else None)
        ax.scatter(b[i, :, 1], s[i, :, 1], marker="^", s=60, facecolor="none", edgecolor=COL[i], lw=1.5,
                   label=f"{INJ[i]}, d/dL2" if ax is axs[0, 0] else None)
    r = np.median(s / b, axis=(0, 1))
    ax.set_title(f"{title}   (median ratio L1 {r[0]:.2f}, L2 {r[1]:.2f})", fontsize=11)
    ax.set_xlabel(f"Bmad finite difference [{unit}]")
    ax.set_ylabel(f"surrogate gradient [{unit}]")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{D}/gradient_parity.png", dpi=130)
print("ok")
