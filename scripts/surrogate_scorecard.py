"""One scorecard for the injector-aware flow candidates on the same held-out rows (processed/xtcav23e_inj_2026-10-15.h5,
group split with the old flow's split roles):

  original eval set (surrogate/eval.py PARITY_KEYS R2, viability AUC, T MAE) on (a) the OLD flow's val rows and
  (b) the held-out injector rows (val + test injector points);
  gates (surrogate_gates.py): old-val spacing MAE / sigma_z rel, paired-delta R2 / slope / p90 (test pairs), T calibration;
  injector effect (analyze_injector_flow.py): cross (golden linac) and pairs R2 / slope on held-out points, + train pairs.
CRN (one frozen base sample) for every prediction. Writes <out>/scorecard.json and scorecard.md.

  PYTHONPATH=src:scripts python scripts/surrogate_scorecard.py --out results/surrogate_scorecard_2026-10-15
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_injector_flow import ORIG, PROC  # noqa: E402

from twobunch_s2e_rl.artifacts import resolve_ckpt  # noqa: E402
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys  # noqa: E402
from twobunch_s2e_rl.surrogate.eval import PARITY_KEYS  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch  # noqa: E402

MODELS = {"old flow (widened)": "trained/xtcav23e_inj12_widened_2026-10-15/widened.ckpt",
          "ft_s0": "trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt",
          "ft_s1": "trained/xtcav23e_inj_ft_s1_2026-10-15/checkpoints/best-*.ckpt",
          "ft_s2": "trained/xtcav23e_inj_ft_s2_2026-10-15/checkpoints/best-*.ckpt",
          "ftc_s0 best (ep 46)": "trained/xtcav23e_inj_ftc_s0_2026-10-15/checkpoints/best-*.ckpt",
          "ftc_s0 final (ep 79)": "trained/xtcav23e_inj_ftc_s0_2026-10-15/checkpoints/final.ckpt"}


def r2(p, t):
    return float(1 - np.sum((p - t) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30))


@torch.no_grad()
def observables(model, knobs, n, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    zd, zw = torch.randn(n, model.latent_dim, generator=g).to(dev), torch.randn(n, model.latent_dim, generator=g).to(dev)
    out = {}
    for b in range(0, len(knobs), 64):
        o = model.observables(knobs[b:b + 64].to(dev), n=n, z_drive=zd, z_witness=zw)
        for k, v in o.items():
            if torch.is_tensor(v) and v.dim() == 1:
                out.setdefault(k, []).append(v.double().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(MODELS["ft_s0"])), map_location="cpu")
    scal = {k: getattr(ref, k).double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    orig = orig_split_keys(*ORIG)
    dm = TwoBunchFlowDataModule(processed_h5=PROC, split="group", orig_split=orig, scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    role = np.full(len(ds), "train", dtype=object)
    role[np.asarray(dm.val.indices)] = "val"
    role[np.asarray(dm.test.indices)] = "test"
    with h5py.File(PROC) as h:
        camp, idx, key, pref = h["campaign"][...], h["idx"][...], h["pair_key"][...], h["pair_ref"][...]
    dd, wd = ds.drive_density.numpy(), ds.witness_density.numpy()
    both = dd & wd
    oldval = np.array([i for i in dm.val.indices if orig.get(ds.row_keys[i]) == "val"])
    inj_ho = np.where((camp == 2) & (role != "train"))[0]
    inj_all = np.where((camp == 2) & both)[0]
    rows = np.unique(np.r_[oldval, inj_ho, inj_all])
    pos = {r: i for i, r in enumerate(rows)}
    # truth
    d = ds.drive[rows].double() * scal["drive_std"] + scal["drive_mean"]
    w = ds.witness[rows].double() * scal["witness_std"] + scal["witness_mean"]
    T = {}
    for nm, parts in (("drive", d), ("witness", w)):
        for kk, vv in per_bunch(parts).items():
            T[f"{nm}_{kk}"] = vv.numpy()
    T.update({k: v.numpy() for k, v in inter_bunch(d, w).items()})
    T["T_drive"], T["T_witness"] = ds.drive_frac[rows].numpy(), ds.witness_frac[rows].numpy()
    viable = ds.witness_viable[rows].numpy() if hasattr(ds, "witness_viable") else None
    # pairs / cross
    meta = {int(json.loads(Path(f).read_text())["idx"]): json.loads(Path(f).read_text())
            for f in glob.glob("data/injector_full/sample_*.json")}
    blk = {r: meta[int(idx[r])] for r in inj_all}
    gcross = [r for r in inj_all if blk[r]["injector_point"] == "golden" and blk[r]["block"] == "crossed"][0]
    cross = [(r, gcross) for r in inj_all if blk[r]["block"] == "crossed" and blk[r]["injector_point"] != "golden"]
    by = {}
    for r in inj_all:
        if key[r] >= 0:
            by.setdefault(int(key[r]), {})["ref" if pref[r] else "pt"] = r
    pairs = [(x["pt"], x["ref"]) for x in by.values() if len(x) == 2]

    def parity(P, sel):
        res = {}
        for k, _, logp in PARITY_KEYS:
            bunch = k.split("_")[0]
            m = np.isin(rows, sel) & (wd[rows] if bunch == "witness" else dd[rows])
            if any(s in k for s in ("spacing", "difference", "offset")):
                m = np.isin(rows, sel) & both[rows]
            p, t = P[k][m], T[k][m]
            if logp:
                p, t = np.log10(np.abs(p) + 1e-30), np.log10(np.abs(t) + 1e-30)
            res[k] = r2(p, t)
        m = np.isin(rows, sel)
        if viable is not None and viable[m].min() != viable[m].max():
            res["witness_viability_auc"] = float(roc_auc_score(viable[m], P["p_surv"][m]))
        md, mw = m & dd[rows], m & (viable.astype(bool) if viable is not None else wd[rows])
        res["T_drive_mae"] = float(np.abs(P["T_drive"][md] - T["T_drive"][md]).mean())
        res["T_witness_mae"] = float(np.abs(P["T_witness"][mw] - T["T_witness"][mw]).mean())
        return res

    def delta(P, S, k="bunch_spacing"):
        if not S:
            return None
        t = 1e6 * np.array([T[k][pos[x]] - T[k][pos[y]] for x, y in S])
        p = 1e6 * np.array([P[k][pos[x]] - P[k][pos[y]] for x, y in S])
        return {"n": len(S), "R2": r2(p, t), "slope": float(np.polyfit(t, p, 1)[0]),
                "p90_um": float(np.percentile(np.abs(p - t), 90))}

    card = {}
    for name, ck in MODELS.items():
        files = glob.glob(ck)
        if not files:
            print("missing", ck)
            continue
        m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(ck)), map_location=dev).eval()
        P = observables(m, ds.knobs[rows], a.n, dev)
        c = {"orig_eval_oldval": parity(P, oldval), "orig_eval_injector_heldout": parity(P, inj_ho)}
        ov = [pos[r] for r in oldval if both[r]]
        c["gate_oldval"] = {"spacing_mae_um": float(1e6 * np.abs(P["bunch_spacing"][ov] - T["bunch_spacing"][ov]).mean()),
                            "sigma_z_rel": float(np.abs(P["drive_sigma_z"][ov] / T["drive_sigma_z"][ov] - 1).mean())}
        c["paired_test"] = delta(P, [s for s in pairs if role[s[0]] == "test"])
        c["pairs_heldout"] = delta(P, [s for s in pairs if role[s[0]] != "train"])
        c["pairs_train"] = delta(P, [s for s in pairs if role[s[0]] == "train"])
        c["cross_heldout"] = delta(P, [s for s in cross if role[s[0]] != "train"])
        tt = [pos[r] for r in inj_ho]
        err = []
        for k in ("T_drive", "T_witness"):
            mm = (T[k][tt] >= 0.95) & (T[k][tt] <= 1.0)
            err += list(np.abs(P[k][tt][mm] - T[k][tt][mm]))
        c["T_calibration_mae"] = float(np.mean(err))
        c["inj_abs_spacing_mae_um"] = float(1e6 * np.abs(P["bunch_spacing"][tt] - T["bunch_spacing"][tt]).mean())
        card[name] = c
        print(name, json.dumps({k: v for k, v in c.items() if not k.startswith("orig")}), flush=True)
        del m
        torch.cuda.empty_cache()
    ref_old = json.loads(Path("results/surrogate/xtcav23/metrics.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps({"reference_xtcav23_metrics_json": ref_old, "models": card}, indent=1))
    # markdown table
    names = list(card)
    L = ["| metric | xtcav23 ref (own val) | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for k, _, _ in PARITY_KEYS:
        L.append(f"| R² {k} (old val) | {ref_old.get(k, {}).get('r2', float('nan')):.3f} | "
                 + " | ".join(f"{card[n]['orig_eval_oldval'][k]:.3f}" for n in names) + " |")
    for k in ("witness_viability_auc", "T_drive_mae", "T_witness_mae"):
        L.append(f"| {k} (old val) | {ref_old.get(k, float('nan')) or float('nan'):.3f} | "
                 + " | ".join(f"{card[n]['orig_eval_oldval'].get(k, float('nan')):.3f}" for n in names) + " |")
    for k, _, _ in PARITY_KEYS:
        L.append(f"| R² {k} (injector held-out) | | " + " | ".join(f"{card[n]['orig_eval_injector_heldout'][k]:.3f}" for n in names) + " |")
    L.append("| gate: old-val spacing MAE [µm] | | " + " | ".join(f"{card[n]['gate_oldval']['spacing_mae_um']:.2f}" for n in names) + " |")
    L.append("| gate: old-val σz rel | | " + " | ".join(f"{card[n]['gate_oldval']['sigma_z_rel']:.3f}" for n in names) + " |")
    for s in ("paired_test", "pairs_heldout", "pairs_train", "cross_heldout"):
        L.append(f"| {s} R² / slope / p90 | | " + " | ".join(
            f"{card[n][s]['R2']:.2f} / {card[n][s]['slope']:.2f} / {card[n][s]['p90_um']:.1f}" for n in names) + " |")
    L.append("| T calibration MAE (gate ≤ 0.02) | | " + " | ".join(f"{card[n]['T_calibration_mae']:.4f}" for n in names) + " |")
    L.append("| injector held-out spacing MAE [µm] | | " + " | ".join(f"{card[n]['inj_abs_spacing_mae_um']:.2f}" for n in names) + " |")
    (out / "scorecard.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
