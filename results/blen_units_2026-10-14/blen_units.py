"""How do the BC11 / BC14 / BC20 BLEN (bunch-length monitor) count units behave in the real DAQ data?
Per-step medians vs the L2 scan, across the 3 E300 (2026-04-15) and 14 E331 (2026-05-01) runs, and
against the XTCAV spacing (E300, energy-gated re-extraction from the harness audit). Read-only.

  /home/rwu4/miniconda3/envs/slac-rl/bin/python results/blen_units_2026-10-14/blen_units.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = Path("/home/rwu4/photoinjector-rl/twobunch-s2e-rl/results/harness_audit_2026-10-14")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUD / "scripts"))
import common as c  # noqa: E402

PV = {"BC11": "BLEN_LI11_359_BRAW", "BC14": "BLEN_LI14_888_BRAW", "BC20": "BLEN_LI20_3014_BRAW",
      "L1": "KLYS_LI11_11_FB_FAST_PACT"}
res = {}
for r in c.E300 + c.E331:
    try:
        d = c.load_mat(r)
    except Exception as e:  # noqa: BLE001
        print(r, "skip", e); continue
    sv = np.atleast_1d(d["params"]["scanVals"]).ravel().astype(float)
    steps = np.unique(d["step"])
    rec = {"scan": [], "n": []}
    for k in PV:
        rec[k], rec[k + "_rms"] = [], []
    for s in steps:
        m = d["step"] == s
        rec["scan"].append(float(sv[s - 1]) if 0 < s <= len(sv) else np.nan)
        rec["n"].append(int(m.sum()))
        for k, pv in PV.items():
            a = d["pv"].get(pv, np.full(len(m), np.nan))[m]
            rec[k].append(float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan)
            rec[k + "_rms"].append(float(c.robust_std(a)))
    if r in c.E300:
        R = c.load_reext(r)
        sp = []
        for s in steps:
            m = (R["step"] == s) & R["ok"].astype(bool)
            sep = np.abs(R["z_hi"][m] - R["z_lo"][m])
            sp.append(float(np.nanmedian(sep)) if m.any() else np.nan)
        rec["xtcav_sep_um"] = sp
    res[r] = rec
    print(r, "BC11 %.0f  BC14 %.0f..%.0f  L1 %.2f..%.2f" % (np.nanmedian(rec["BC11"]), np.nanmin(rec["BC14"]),
          np.nanmax(rec["BC14"]), np.nanmin(rec["L1"]), np.nanmax(rec["L1"])), flush=True)
(OUT / "blen_steps.json").write_text(json.dumps(res, indent=1))

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for r, rec in res.items():
    col = "C0" if r in c.E300 else "C1"
    lw = 1.6 if r in c.E300 else 0.8
    ax[0, 0].plot(rec["scan"], rec["BC14"], "-", color=col, lw=lw, alpha=0.8)
    ax[0, 1].plot(rec["scan"], np.array(rec["BC14"]) / np.nanmax(rec["BC14"]), "-", color=col, lw=lw, alpha=0.8)
    ax[1, 0].plot(np.nanmedian(rec["L1"]), np.nanmedian(rec["BC11"]), "o" if r in c.E300 else "s", color=col)
    if "xtcav_sep_um" in rec:
        ax[1, 1].plot(rec["BC14"], rec["xtcav_sep_um"], "o-", ms=3, label=r)
ax[0, 0].set(xlabel="L2_PHASE.MKB scan value [deg]", ylabel="BC14 BLEN [raw counts], step median",
             title="BC14 BLEN vs L2 scan (blue E300 04-15, orange E331 05-01)")
ax[0, 1].set(xlabel="L2 scan [deg]", ylabel="BC14 BLEN / run max", title="Same, normalized per run (shape)")
ax[1, 0].set(xlabel="L1 FAST_PACT run median [deg]", ylabel="BC11 BLEN run median [counts]",
             title="BC11 BLEN setpoint vs the L1 phase the feedback chose")
ax[1, 1].set(xlabel="BC14 BLEN [counts]", ylabel="XTCAV energy-gated separation [um]", title="E300: spacing vs BC14 BLEN")
ax[1, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "blen_units.png", dpi=120)
print("wrote", OUT)
