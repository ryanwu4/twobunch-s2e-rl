"""Deterministic regressors on bunch spacing (truth = Bmad, all processed rows, flow's group split): does the
data support better spacing / injector-effect accuracy than the flow? GBM, poly-2/3 ridge, MLP ensemble.
Prints absolute spacing MAE per campaign / role and the held-out pair R^2 / slope (same pairs as the gates).

  PYTHONPATH=src:scripts python scripts/spacing_regressor_baselines.py
"""
import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from analyze_injector_flow import ORIG, PROC
from surrogate_gates import truth
from twobunch_s2e_rl.artifacts import resolve_ckpt
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow


def main():
    m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt("trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt")),
                                          map_location="cpu")
    scal = {k: getattr(m, k).double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    dm = TwoBunchFlowDataModule(processed_h5=PROC, split="group", orig_split=orig_split_keys(*ORIG),
                                scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    role = np.full(len(ds), "train", dtype=object)
    role[np.asarray(dm.val.indices)] = "val"
    role[np.asarray(dm.test.indices)] = "test"
    ok = ds.drive_density.numpy() & ds.witness_density.numpy()
    rows = np.where(ok)[0]
    with h5py.File(PROC) as h:
        camp, key, ref = h["campaign"][...], h["pair_key"][...], h["pair_ref"][...]
    y = 1e6 * truth(ds, list(rows), scal)["bunch_spacing"]
    X = ds.knobs.numpy()[rows].astype(np.float32)
    R, C = role[rows], camp[rows]
    tr = R == "train"
    pos = {r: i for i, r in enumerate(rows)}
    by = {}
    for r in rows:
        if key[r] >= 0 and camp[r] == 2:
            by.setdefault(int(key[r]), {})["ref" if ref[r] else "pt"] = r
    P = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2]

    def report(name, p):
        out = [f"{name}:"]
        for c, nm in ((0, "old"), (2, "inj")):
            for r_ in ("train", "val", "test"):
                s = (C == c) & (R == r_)
                if s.any():
                    out.append(f"{nm}/{r_} {np.abs(p[s] - y[s]).mean():.2f}")
        S = [(a, b) for a, b in P if role[a] != "train"]
        dt = np.array([y[pos[a]] - y[pos[b]] for a, b in S])
        dp = np.array([p[pos[a]] - p[pos[b]] for a, b in S])
        out.append(f"| held-out pairs R2 {1 - np.sum((dp - dt) ** 2) / np.sum((dt - dt.mean()) ** 2):.2f} "
                   f"slope {np.polyfit(dt, dp, 1)[0]:.2f}")
        print(" ".join(out), flush=True)

    report("GBM", HistGradientBoostingRegressor(max_iter=3000, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=5,
                                                l2_regularization=1e-3, early_stopping=False, random_state=0)
           .fit(X[tr], y[tr]).predict(X))
    for deg in (2, 3):
        report(f"poly-{deg} ridge", make_pipeline(PolynomialFeatures(deg), StandardScaler(),
                                                  RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(X[tr], y[tr]).predict(X))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor((y - y[tr].mean()) / y[tr].std(), device=dev, dtype=torch.float32)
    preds = []
    for seed in range(3):
        torch.manual_seed(seed)
        idx = np.where(tr)[0]
        np.random.default_rng(seed).shuffle(idx)
        vi, ti = idx[: len(idx) // 10], torch.as_tensor(idx[len(idx) // 10:], device=dev)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.SiLU(), torch.nn.Linear(512, 512),
                                  torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(), torch.nn.Linear(512, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
        best, best_state = 1e9, None
        for ep in range(3000):
            perm = torch.randperm(len(ti), device=dev)
            for b in range(0, len(ti), 256):
                j = ti[perm[b:b + 256]]
                loss = torch.nn.functional.smooth_l1_loss(net(Xt[j]).squeeze(-1), yt[j], beta=0.05)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            if ep % 25 == 0:
                with torch.no_grad():
                    v = (net(Xt[vi]).squeeze(-1) - yt[vi]).abs().mean().item()
                if v < best:
                    best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        with torch.no_grad():
            preds.append(net(Xt).squeeze(-1).cpu().numpy() * y[tr].std() + y[tr].mean())
        report(f"MLP seed {seed}", preds[-1])
    report("MLP ensemble(3)", np.mean(preds, 0))


if __name__ == "__main__":
    main()
