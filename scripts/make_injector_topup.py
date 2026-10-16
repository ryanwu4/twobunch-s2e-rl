"""Bmad-only top-up of injector_full (overnight 2026-10-16): more exact injector-effect pairs, no new Impact-T.

Writes a NEW campaign directory (default data/injector_topup) with
  manifest.json   pre-built rows (the driver reuses a manifest it finds):
    (a) n_pairs golden-injector re-runs of existing UNPAIRED injector_full nested rows (controller region
        weighted), same knobs + linac_id + design seed as the original -> same Bmad seed -> exact pairs;
        pair_campaign = injector_full links them to the original nested row in preprocess
    (b) one new nested linac draw per existing LHS injector point (controller_frac in the controller region),
        each + its golden-injector pair (linac_id 3_000_000 + j); point_campaign = injector_full puts the row in
        its injector point's split group
  beams/, l0af_reference.json, l0af_autophase/   COPIED from injector_full (the cached Impact-T beams and the
        pinned L0AF reference; injector_full itself is only read)
  topup_launch.yaml   the campaign config (injector_full_launch.yaml fidelity + this output dir)

  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/make_injector_topup.py
  then (bmad-qpad-dev, repo root): python -m twobunch_s2e_rl.datagen.injector_campaign <out>/topup_launch.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from twobunch_s2e_rl.datagen.injector_campaign import _controller_draws, _lhs, atomic_write  # noqa: E402
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/injector_full")
    ap.add_argument("--src-cfg", default="configs/datagen/injector_full_launch.yaml")
    ap.add_argument("--out", default="data/injector_topup")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--controller-pair-frac", type=float, default=0.7)
    ap.add_argument("--controller-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    src, out = REPO / a.src, REPO / a.out
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    cfg = yaml.safe_load((REPO / a.src_cfg).read_text())
    rows0 = json.loads((src / "manifest.json").read_text())
    src_name = cfg["name"]
    golden = next(r["injector"] for r in rows0 if r["injector_point"] == "golden")
    test_tags = {r["injector_point"] for r in rows0 if r.get("holdout") and r["injector_point"] != "golden"}
    paired_ids = {r["linac_id"] for r in rows0 if r["block"] == "paired_golden"}
    nested = [r for r in rows0 if r["block"] == "nested" and r["linac_id"] not in paired_ids]
    rng = np.random.default_rng(a.seed)
    ctrl = [r for r in nested if r["linac_region"] == "controller"]
    full = [r for r in nested if r["linac_region"] != "controller"]
    n_c = min(len(ctrl), round(a.controller_pair_frac * a.n_pairs))
    pick = [ctrl[i] for i in rng.choice(len(ctrl), n_c, replace=False)] + \
           [full[i] for i in rng.choice(len(full), a.n_pairs - n_c, replace=False)]
    rows = []

    def add(**kw):
        rows.append({"idx": len(rows), **kw})

    for r in sorted(pick, key=lambda r: r["linac_id"]):           # (a) exact pairs of existing rows
        add(block="paired_golden", beam="impact", injector_point="golden", injector=dict(golden),
            knobs=dict(r["knobs"]), linac_id=r["linac_id"], linac_region=r["linac_region"],
            pair_of=r["injector_point"], pair_campaign=src_name, holdout=r["injector_point"] in test_tags)
    # (b) one new nested draw per LHS injector point + its golden pair
    lkeys, llo, lhi, lbase = resolve_sweep_set(cfg["linac_sweep_set"])
    pts = {}
    for r in rows0:
        if r["injector_point"].startswith("lhs"):
            pts.setdefault(r["injector_point"], r["injector"])
    tags = sorted(pts, key=lambda t: int(t[3:]))
    n_ctrl = round(a.controller_frac * len(tags))
    des = cfg["design"]["linac"]
    draws_c = _controller_draws(n_ctrl, lkeys, llo, lhi, lbase, des["controller_box"], a.seed + 1,
                                des.get("controller_l1_energy"))
    draws_f = _lhs(len(tags) - n_ctrl, llo, lhi, a.seed + 2)
    order = rng.permutation(len(tags))
    for j, ti in enumerate(order):
        tag = tags[ti]
        is_c = j < n_ctrl
        knobs = dict(zip(lkeys, map(float, (draws_c[j] if is_c else draws_f[j - n_ctrl]))))
        lid = 3_000_000 + j
        reg = "controller" if is_c else "full"
        add(block="nested", beam="impact", injector_point=tag, injector=dict(pts[tag]), knobs=knobs,
            linac_id=lid, linac_region=reg, point_campaign=src_name, holdout=tag in test_tags)
        add(block="paired_golden", beam="impact", injector_point="golden", injector=dict(golden), knobs=dict(knobs),
            linac_id=lid, linac_region=reg, pair_of=tag, holdout=tag in test_tags)
    # campaign config: same fidelity / physics / design seed (Bmad seed = seed * 10007 + linac_id + 1)
    cfg2 = dict(cfg)
    cfg2["name"] = "injector_topup"
    cfg2["output_dir"] = str(out)
    cfg2["design"] = {**cfg["design"], "topup_of": src_name, "topup_seed": a.seed, "topup_n_pairs": a.n_pairs}
    out.mkdir(parents=True)
    atomic_write(out / "manifest.json", json.dumps(rows, indent=1))
    shutil.copytree(src / "beams", out / "beams")
    shutil.copy2(src / "l0af_reference.json", out / "l0af_reference.json")
    if (src / "l0af_autophase").exists():
        shutil.copytree(src / "l0af_autophase", out / "l0af_autophase")
    (out / "topup_launch.yaml").write_text(
        f"# GENERATED by scripts/make_injector_topup.py: Bmad-only top-up of {src_name} "
        f"({len(rows)} rows: {len(pick)} pairs of existing nested rows + {len(tags)} new nested + {len(tags)} pairs)\n"
        + yaml.safe_dump(cfg2, sort_keys=False))
    blocks = {}
    for r in rows:
        blocks[(r["block"], r["linac_region"])] = blocks.get((r["block"], r["linac_region"]), 0) + 1
    print(f"{out}: {len(rows)} rows {blocks}; holdout {sum(r['holdout'] for r in rows)}")


if __name__ == "__main__":
    main()
