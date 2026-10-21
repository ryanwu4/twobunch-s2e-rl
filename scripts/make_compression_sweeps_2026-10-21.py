"""Dense L1 sweeps around full compression (2026-10-21, reviewer item 5): Bmad-only campaign on cached injector beams.

Writes data/compress_sweeps_2026-10-21/ (new dir; injector_cov_2026-10-16 is only read):
  manifest.json  rows (block "jacobian", point_campaign = injector_cov_2026-10-16 so each row joins its injector point's
                 split group): n_train train-role + n_val val-role coverage-campaign injector points + the golden injector,
                 x 2 L2 values (uniform in the controller box), x 15 L1 values on [L1* - 2, L1* + 2] clipped to the
                 controller box, where L1* = argmin over L1 of the B2-cov surrogate's drive sigma_z at that (injector, L2);
                 every other linac knob at its baseline. Unique linac_id per row (5_000_000 + j).
  beams/, l0af_reference.json, l0af_autophase/   COPIED from injector_cov_2026-10-16 (cached Impact-T beams)
  launch.yaml    the campaign config (injector_cov_2026-10-16 fidelity / physics, this output dir)
Run (slac-rl env, repo root): PYTHONPATH=src python scripts/make_compression_sweeps_2026-10-21.py
then (bmad-qpad-dev): python -m twobunch_s2e_rl.datagen.injector_campaign data/compress_sweeps_2026-10-21/launch.yaml
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from twobunch_s2e_rl.datagen.injector_campaign import atomic_write  # noqa: E402
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

B2 = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/injector_cov_2026-10-16")
    ap.add_argument("--src-cfg", default="logs/injector_cov_2026-10-16/launch_config.yaml")
    ap.add_argument("--out", default="data/compress_sweeps_2026-10-21")
    ap.add_argument("--points", default="/tmp/claude-1011/-home-rwu4-photoinjector-rl/39cf6bee-ba8a-4919-92a7-359b01edba48/scratchpad/cov_points_by_split.json")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--n-l1", type=int, default=15)
    ap.add_argument("--n-l2", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    src, out = REPO / a.src, REPO / a.out
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    cfg = yaml.safe_load((REPO / a.src_cfg).read_text())
    rows0 = json.loads((src / "manifest.json").read_text())
    pts = {}
    for r in rows0:
        pts.setdefault(r["injector_point"], r["injector"])
    roles = json.loads(Path(a.points).read_text())
    rng = np.random.default_rng(a.seed)
    tr = [t for t in roles["train"] if t != "golden"]
    va = [t for t in roles["val"] if t != "golden"]
    chosen = ([("golden", "golden")] + [(t, "train") for t in rng.choice(tr, a.n_train, replace=False)]
              + [(t, "val") for t in rng.choice(va, a.n_val, replace=False)])
    lkeys, llo, lhi, lbase = resolve_sweep_set(cfg["linac_sweep_set"])
    box = cfg["design"]["linac"]["controller_box"]
    flow = TwoBunchFlow.load_from_checkpoint(str(REPO / B2), map_location="cpu").eval()
    fk = list(flow.hparams.knob_keys)
    lo, hi = flow.knob_low.numpy(), flow.knob_high.numpy()
    i1, i2 = fk.index("L1PhaseSet"), fk.index("L2PhaseSet")
    l1_grid = np.linspace(box["L1PhaseSet"][0], box["L1PhaseSet"][1], 81)
    rows = []
    j = 0
    for tag, role in chosen:
        inj = pts[tag]
        for l2 in rng.uniform(*box["L2PhaseSet"], a.n_l2):
            base = {**lbase, **inj}
            X = np.array([[base[k] for k in fk]] * len(l1_grid), float)
            X[:, i1], X[:, i2] = l1_grid, l2
            kn = torch.as_tensor((X - lo) / (hi - lo), dtype=torch.float32)
            with torch.no_grad():
                sz = np.array([flow.observables(kn[b:b + 27], n=2048)["drive_sigma_z"].numpy() for b in range(0, 81, 27)]).ravel()
            l1s = l1_grid[int(np.argmin(sz))]
            l1v = np.clip(np.linspace(l1s - 2.0, l1s + 2.0, a.n_l1), *box["L1PhaseSet"])
            for l1 in np.unique(np.round(l1v, 4)):
                knobs = {k: float(lbase[k]) for k in lkeys}
                knobs["L1PhaseSet"], knobs["L2PhaseSet"] = float(l1), float(l2)
                row = {"idx": len(rows), "block": "jacobian", "beam": "impact", "injector_point": tag,
                       "injector": dict(inj), "knobs": knobs, "linac_id": 5_000_000 + j, "linac_region": "controller",
                       "jacobian_delta": f"sweep_L1*={l1s:.2f}", "holdout": False, "sweep_role": role,
                       "injector_region": "golden" if tag == "golden" else ("local" if tag.startswith("lhsloc") else "full")}
                if tag != "golden":
                    row["point_campaign"] = cfg["name"]
                rows.append(row)
                j += 1
    cfg2 = dict(cfg)
    cfg2["name"] = "compress_sweeps_2026-10-21"
    cfg2["output_dir"] = str(out)
    cfg2["n_workers"] = 110
    cfg2["design"] = {**cfg["design"], "sweeps_of": cfg["name"], "sweeps_seed": a.seed}
    out.mkdir(parents=True)
    atomic_write(out / "manifest.json", json.dumps(rows, indent=1))
    shutil.copytree(src / "beams", out / "beams")
    shutil.copy2(src / "l0af_reference.json", out / "l0af_reference.json")
    if (src / "l0af_autophase").exists():
        shutil.copytree(src / "l0af_autophase", out / "l0af_autophase")
    (out / "launch.yaml").write_text(f"# GENERATED by scripts/make_compression_sweeps_2026-10-21.py ({len(rows)} rows)\n"
                                     + yaml.safe_dump(cfg2, sort_keys=False))
    by = {}
    for r in rows:
        by[r["sweep_role"]] = by.get(r["sweep_role"], 0) + 1
    print(f"{out}: {len(rows)} rows by role {by}")


if __name__ == "__main__":
    main()
