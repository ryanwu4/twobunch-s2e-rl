"""B2-cov + a deterministic z-centroid head (centroid_dims=[2]), no flow retraining: the head is fitted on the TRAIN rows
(z centroids + spacing + injector-pair differences, Huber) with early stopping on the VAL pairs' paired-delta rms,
everything else (transverse / momentum centres, shape, T heads) stays B2-cov's. Writes a Lightning-loadable checkpoint.

  PYTHONPATH=src python scripts/build_zcentroid_flow_2026-10-18.py --out trained/xtcav23e_inj_covB2zc_2026-10-18
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS, build_pair_source

SRC = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
TABLE = "processed/xtcav23e_inj_cov_2026-10-17.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--w-pair", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--param", choices=("per_bunch", "spacing"), default="per_bunch")
    ap.add_argument("--act", choices=("relu", "silu"), default="relu")
    ap.add_argument("--abs-mae-max", type=float, default=4.5,
                    help="selection: best val paired-delta rms among steps whose val ABSOLUTE spacing MAE [um] is <= this")
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    hp.update(centroid_head=True, centroid_dims=[2], w_centroid=1.0, centroid_param=a.param, centroid_act=a.act)
    model = TwoBunchFlow(**hp)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and all(k.startswith("centroid_heads.") or k == "_z_center" for k in missing), (missing, unexpected)
    roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    # the data in the CHECKPOINT's standardized frame (as train.py does for a warm start; B2-cov carries its lineage's
    # scaler, not the cov table's -- the first build used the table's own and fitted the head in the wrong frame)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    dm = TwoBunchFlowDataModule(processed_h5=TABLE, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    model.pair_source = build_pair_source(dm, model, 0.3)
    ds = dm.full
    (md, sd), (mw, sw) = model._scaler(0), model._scaler(1)
    zc = lambda parts, m, s: parts.mean(1)[:, 2] * s[2] + m[2]        # noqa: E731
    both = (ds.drive_density & ds.witness_density).numpy().astype(bool)
    sp_all = (zc(ds.drive, md, sd) - zc(ds.witness, mw, sw)).numpy()
    tr = np.asarray(dm.train.indices)
    va = set(np.asarray(dm.val.indices).tolist())
    # val pairs (point, reference) by pair key, as the scorecard
    import h5py
    with h5py.File(TABLE) as h:
        key, pref = h["pair_key"][...], h["pair_ref"][...]
    by = {}
    for i in np.flatnonzero((key >= 0) & both):
        by.setdefault(int(key[i]), {})["ref" if pref[i] else "pt"] = i
    vpairs = np.array([(v["pt"], v["ref"]) for v in by.values() if len(v) == 2 and v["pt"] in va])
    knobs = ds.knobs
    data = []
    for k, (parts, dens) in enumerate(((ds.drive, ds.drive_density), (ds.witness, ds.witness_density))):
        rows = tr[dens[tr].numpy().astype(bool)]
        data.append((knobs[rows], parts[rows].mean(1)[:, 2]))
    tb = tr[both[tr]]
    kb, spb = knobs[tb], torch.tensor(sp_all[tb])
    opt = torch.optim.AdamW(model.centroid_heads.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(0)
    scale = 1e-5

    vrows = np.array(sorted(i for i in va if both[i]))

    def val_rms():
        with torch.no_grad():
            kv = knobs[vrows]
            spv = (model.centroid_phys(kv, 0)[:, 2] - model.centroid_phys(kv, 1)[:, 2]).numpy()
        mae = float(np.mean(np.abs(spv - sp_all[vrows])) * 1e6)
        with torch.no_grad():
            kk = knobs[vpairs.ravel()]
            sp = (model.centroid_phys(kk, 0)[:, 2] - model.centroid_phys(kk, 1)[:, 2]).numpy().reshape(-1, 2)
        dpred = sp[:, 0] - sp[:, 1]
        dtrue = sp_all[vpairs[:, 0]] - sp_all[vpairs[:, 1]]
        e = dpred - dtrue
        return float(np.sqrt(np.mean(e ** 2)) * 1e6), float(np.polyfit(dtrue, dpred, 1)[0]), float(1 - np.var(e) / np.var(dtrue)), mae

    best = (np.inf, None, -1)
    for it in range(a.steps):
        loss = 0.0
        for k, (kn, tz) in enumerate(data):
            if k == 1 and a.param == "spacing":
                continue                      # the witness z is drive z - spacing: fitted through the spacing terms
            i = torch.randint(len(kn), (512,), generator=g)
            loss = loss + F.smooth_l1_loss(model.centroid_std(kn[i], k)[:, 2], tz[i], beta=0.05)
        i = torch.randint(len(kb), (512,), generator=g)
        sp = model.centroid_phys(kb[i], 0)[:, 2] - model.centroid_phys(kb[i], 1)[:, 2]
        loss = loss + F.smooth_l1_loss(sp / scale, spb[i] / scale, beta=0.5) + a.w_pair * model._pair_loss()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (it + 1) % a.eval_every == 0:
            r, sl, r2, mae = val_rms()
            if mae <= a.abs_mae_max and r < best[0]:
                best = (r, {k: v.clone() for k, v in model.centroid_heads.state_dict().items()}, it + 1)
            print(f"step {it + 1}: val paired-delta rms {r:.2f} um, slope {sl:.2f}, R2 {r2:.2f}, abs spacing MAE {mae:.2f} um "
                  f"(best {best[0]:.2f} @ {best[2]})", flush=True)
    if best[1] is None:
        raise SystemExit(f"no step with val absolute spacing MAE <= {a.abs_mae_max} um")
    model.centroid_heads.load_state_dict(best[1])
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    ck2 = {"state_dict": model.state_dict(), "hyper_parameters": hp,
           "pytorch-lightning_version": ck.get("pytorch-lightning_version"), "epoch": 0, "global_step": best[2]}
    name = f"best-step={best[2]:05d}-val_paired_delta_rms_um={best[0]:.4f}.ckpt"
    torch.save(ck2, out / "checkpoints" / name)
    src_dir = Path(SRC).parents[1]
    for f in ("norm.json",):
        if (src_dir / f).exists():
            shutil.copy(src_dir / f, out / f)
    (out / "BUILD.json").write_text(json.dumps({"from": SRC, "table": TABLE, "args": vars(a), "best_step": best[2],
                                                "val_paired_delta_rms_um": best[0]}, indent=1))
    print(out / "checkpoints" / name)


if __name__ == "__main__":
    main()
