"""Structure of the injector effect on bunch spacing away from the golden linac point (truth = Bmad,
injector_full): is the pair effect (same linac knobs, injector point vs golden) the golden-linac response of
the same injector point, scaled by the linac's compression? Fits Delta_pair ~ a * Delta_cross and
Delta_pair ~ a * Delta_cross * (s_ref / s_golden), and a linear model in the injector knobs with linac-dependent
coefficients (inj x linac interactions), to say what kind of model / data the interaction needs.

  PYTHONPATH=src python scripts/injector_effect_structure.py --out results/injector_flow_analysis_2026-10-15
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_injector_flow import INJ, ORIG, PROC, r2  # noqa: E402
from surrogate_gates import truth  # noqa: E402

from twobunch_s2e_rl.artifacts import resolve_ckpt  # noqa: E402
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt("trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt")),
                                          map_location="cpu")
    scal = {k: getattr(m, k).double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    dm = TwoBunchFlowDataModule(processed_h5=PROC, split="group", orig_split=orig_split_keys(*ORIG),
                                scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    with h5py.File(PROC) as h:
        camp, idx, key, ref = h["campaign"][...], h["idx"][...], h["pair_key"][...], h["pair_ref"][...]
    ok = ds.drive_density.numpy() & ds.witness_density.numpy()
    rows = np.where((camp == 2) & ok)[0]
    meta = {int(json.loads(Path(f).read_text())["idx"]): json.loads(Path(f).read_text())
            for f in glob.glob("data/injector_full/sample_*.json")}
    md = {r: meta[int(idx[r])] for r in rows}
    sp = dict(zip(rows, 1e6 * truth(ds, list(rows), scal)["bunch_spacing"]))
    cross = {md[r]["injector_point"]: r for r in rows if md[r]["block"] == "crossed"}
    g = cross["golden"]
    by = {}
    for r in rows:
        if key[r] >= 0:
            by.setdefault(int(key[r]), {})["ref" if ref[r] else "pt"] = r
    P = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2 and md[d["pt"]]["injector_point"] in cross]
    dpair = np.array([sp[p] - sp[q] for p, q in P])
    dcross = np.array([sp[cross[md[p]["injector_point"]]] - sp[g] for p, _ in P])
    ratio = np.array([sp[q] / sp[g] for _, q in P])
    lin = lambda X, y: np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0]
    pred = lambda X, c: np.c_[X, np.ones(len(X))] @ c
    res = {"n_pairs": len(P), "golden_spacing_um": float(sp[g]), "pair_signal_rms_um": float(dpair.std())}
    c1 = lin(dcross[:, None], dpair); res["pair_vs_cross"] = {"slope": float(c1[0]), "R2": r2(pred(dcross[:, None], c1), dpair)}
    xs = dcross * ratio
    c2 = lin(xs[:, None], dpair); res["pair_vs_cross_x_compression"] = {"slope": float(c2[0]), "R2": r2(pred(xs[:, None], c2), dpair)}
    # interaction model: Delta = sum_k (b_k + c_k . linac) * dinj_k on the 5 dominant injector knobs
    L = ["L1PhaseSet", "L2PhaseSet", "L1EnergyOffset", "L2EnergyOffset", "L3EnergyOffset"]
    gold = np.array([md[g]["knobs"][k] for k in INJ[:5]])
    dI = np.array([[md[p]["knobs"][k] for k in INJ[:5]] for p, _ in P]) - gold
    dI = dI / np.maximum(np.abs(dI).max(0), 1e-30)
    lin_k = np.array([[md[q]["knobs"][k] for k in L] for _, q in P])
    lin_k = (lin_k - lin_k.mean(0)) / np.maximum(lin_k.std(0), 1e-30)
    X = np.concatenate([dI] + [dI * lin_k[:, [j]] for j in range(len(L))], 1)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    cvp = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), X, dpair, cv=10)
    res["interaction_ridge_cv"] = {"R2": r2(cvp, dpair), "features": "5 injector knobs x (1 + 5 linac knobs)"}
    cv0 = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), dI, dpair, cv=10)
    res["injector_only_ridge_cv"] = {"R2": r2(cv0, dpair)}
    (out / "injector_effect_structure.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].scatter(dcross, dpair, s=10, c=ratio, cmap="coolwarm", vmin=0, vmax=2)
    axes[0].set(xlabel="effect at the golden linac point (cross) [um]", ylabel="effect at the pair's linac point [um]",
                title=f"pair vs cross effect, same injector point: R2 {res['pair_vs_cross']['R2']:.2f}")
    axes[1].scatter(xs, dpair, s=10, c=ratio, cmap="coolwarm", vmin=0, vmax=2)
    axes[1].set(xlabel="cross effect x (s_ref / s_golden) [um]", ylabel="pair effect [um]",
                title=f"compression-scaled: R2 {res['pair_vs_cross_x_compression']['R2']:.2f}")
    sc = axes[2].scatter(cvp, dpair, s=10, c=ratio, cmap="coolwarm", vmin=0, vmax=2)
    axes[2].set(xlabel="10-fold CV prediction [um]", ylabel="pair effect [um]",
                title=f"ridge: 5 injector knobs x (1 + 5 linac knobs), CV R2 {res['interaction_ridge_cv']['R2']:.2f}")
    fig.colorbar(sc, ax=axes[2], label="s_ref / s_golden (compression state)")
    for ax in axes:
        lim = np.array(ax.get_xlim()); ax.plot(lim, lim, "k--", lw=0.8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_effect_structure.png", dpi=120)


if __name__ == "__main__":
    main()
