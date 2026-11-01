"""Fig 1: per-shot time series across step boundaries (E300_15673 steps 9-14; E331_16024 steps 20-30)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AUD, load_mat, load_reext, tcav_class


def roll(x, w=9):
    out = np.full(len(x), np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def panel(r, k0, k1, axs, with_images):
    D = load_mat(r)
    t, st, pv = D["t"], D["step"], D["pv"]
    m = (st >= k0) & (st <= k1)
    t0 = t[m].min()
    tt = t - t0
    A, P = pv["TCAV_LI20_2400_A"], pv["TCAV_LI20_2400_P"]
    c, _ = tcav_class(A, P)
    rows = [("LI12_SBST_1_PDES", "L2 SBST PDES / PHAS [deg]", "LI12_SBST_1_PHAS"),
            ("KLYS_LI11_11_FB_FAST_PACT", "L1 klys 11-1 fast PACT [deg]", None),
            ("BLEN_LI11_359_BRAW", "BC11 BLEN [raw]", None),
            ("BLEN_LI20_3014_BRAW", "BC20 BLEN [raw]", None),
            ("BPMS_LI14_801_X", "BC14 BPM 801 X [mm]", None),
            ("TORO_LI20_2452_TMIT", "TORO LI20 2452 [1e9 e]", None)]
    i = 0
    for name, lab, name2 in rows:
        ax = axs[i]
        i += 1
        if name not in pv:
            continue
        a = pv[name].copy()
        if "PACT" in name:
            a[np.abs(a - np.nanmedian(a)) > 5] = np.nan
        if "TMIT" in name:
            a = a / 1e9
        ax.plot(tt[m], a[m], ".", ms=1.5, color="0.6")
        if "SBST" not in name:
            ax.plot(tt[m], roll(a, 9)[m], "-", lw=1.0, color="C0")
        if name2:
            ax.plot(tt[m], pv[name2][m], "-", lw=1.0, color="C3", label="PHAS readback")
            ax.plot(tt[m], pv[name][m], "-", lw=1.0, color="C0", label="PDES")
            ax.legend(fontsize=6, loc="upper left")
        ax.set_ylabel(lab, fontsize=7)
        for k in range(k0, k1 + 1):
            ax.axvline(tt[st == k].min(), color="k", lw=0.5, ls=":")
    ax = axs[i]
    i += 1
    cols = {"+": "C2", "-": "C3", "x": "C1", "a": "C4", "o": "k"}
    for ch, y in zip("+-xao", [4, 3, 2, 1, 0]):
        sel = m & (c == ch)
        ax.plot(tt[sel], np.full(sel.sum(), y), "|", color=cols[ch], ms=6)
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels(["+90 ok", "-90 ok", "phase settling", "amp settling", "TCAV off"], fontsize=6)
    if with_images:
        R = load_reext(r)
        ci = D["common_index"]
        tsh = t[ci[R["ind"].astype(int)] - 1] - t0
        sc = (np.abs(R["phase"] - 90) < 1) | (np.abs(R["phase"] + 90) < 1)
        sep = R["z_lo"] - R["z_hi"]
        ax = axs[i]
        for sgn, col in ((1, "C2"), (-1, "C3")):
            s = sc & (np.sign(R["phase"]) == sgn) & (R["step"] >= k0) & (R["step"] <= k1) & (R["ok"] == 1)
            ax.plot(tsh[s], sep[s], ".", ms=2.5, color=col, label="+90" if sgn > 0 else "-90")
        ax.set_ylabel("signed band sep\nz_d - z_w [um]", fontsize=7)
        ax.legend(fontsize=6)
        for k in range(k0, k1 + 1):
            ax.axvline(tt[st == k].min(), color="k", lw=0.5, ls=":")
    axs[-1].set_xlabel(f"{r}: wall-clock time [s] (dotted = first recorded shot of each step)")


fig, axs = plt.subplots(8, 2, figsize=(14, 13), dpi=120, sharex="col")
panel("E300_15673", 9, 14, axs[:, 0], True)
panel("E331_16024", 20, 30, axs[:7, 1], False)
axs[7, 1].axis("off")
axs[0, 0].set_title("E300_15673 (BC11 BLEN fb 4000, +/-90 TCAV, 150 shots/step)", fontsize=9)
axs[0, 1].set_title("E331_16024 (+90 TCAV only, 20 shots/step)", fontsize=9)
fig.tight_layout()
fig.savefig(AUD / "figs" / "fig1_timeseries_step_boundaries.png")
