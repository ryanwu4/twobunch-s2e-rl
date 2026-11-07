"""Check a calibration campaign before launch and write datasets/<name>/design_check.json.

- physics: every data-defining key of campaign.yaml equals the reference campaign's (the campaign whose
  simulator the model was trained on), except the design, the saved planes and the bookkeeping keys;
- box: the swept knobs are the model bundle's knobs, in its order, with its bounds (to float32);
- bundle: the shipped checkpoint matches the sha256 its bundle.yaml records;
- design: the manifest the generator will write (rows, one beam per row, rows inside the bundle box) and its sha256.

From the repo root (slac-rl env, twobunch_cnf installed):
  PYTHONPATH=$PWD/src python conformal/scripts/check_design.py conformal/datasets/<name> \
      --reference configs/datagen/injector_cov_2026-10-16.yaml --bundle c_b3_ema
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

NOT_PHYSICS = {"name", "output_dir", "purpose", "n_workers", "design", "save_beams_at"}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="datasets/<name> (holds campaign.yaml)")
    ap.add_argument("--reference", required=True, help="campaign config the model's training data used")
    ap.add_argument("--bundle", required=True, help="shipped twobunch_cnf bundle name")
    a = ap.parse_args()

    import twobunch_cnf
    from twobunch_s2e_rl.artifacts import git_state
    from twobunch_s2e_rl.datagen.injector_campaign import injector_points, load_cfg, make_rows
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set

    ds = Path(a.dataset)
    cfg = load_cfg(ds / "campaign.yaml")
    ref = yaml.safe_load(Path(a.reference).read_text())
    keys = (set(cfg) | set(ref)) - NOT_PHYSICS
    physics_diff = {k: [cfg.get(k), ref.get(k)] for k in sorted(keys) if cfg.get(k) != ref.get(k)}

    bdir = Path(twobunch_cnf.__file__).parent / "resources" / a.bundle
    bundle = yaml.safe_load((bdir / "bundle.yaml").read_text())
    ckpt_sha = sha256(bdir / bundle["checkpoint"]["file"])
    bknobs = bundle["knobs"]
    skeys, slo, shi, _ = resolve_sweep_set(cfg["sweep_set"])
    blo, bhi = np.array([k["low"] for k in bknobs]), np.array([k["high"] for k in bknobs])
    span = bhi - blo
    # the bundle stores the bounds as float32 values
    bounds_ok = (skeys == [k["name"] for k in bknobs]
                 and np.array_equal(np.float32(slo), np.float32(blo))
                 and np.array_equal(np.float32(shi), np.float32(bhi)))

    rows = make_rows(cfg)
    x = np.array([[{**r["knobs"], **r["injector"]}[k] for k in skeys] for r in rows])
    u = (x - blo) / span                        # the model's normalized inputs
    text = json.dumps(rows, indent=1)           # exactly what build_manifest writes
    out = {
        "dataset": cfg["name"],
        "campaign_yaml_sha256": sha256(ds / "campaign.yaml"),
        "generator": git_state(),
        "reference_config": str(a.reference), "reference_config_sha256": sha256(a.reference),
        "physics_matches_reference": not physics_diff, "physics_diff": physics_diff,
        "bundle": a.bundle, "bundle_checkpoint_sha256": ckpt_sha,
        "bundle_checkpoint_sha256_ok": ckpt_sha == bundle["checkpoint"]["sha256"],
        "twobunch_cnf_version": twobunch_cnf.__version__,
        "box_matches_bundle": bool(bounds_ok), "n_knobs": len(skeys),
        "rows": len(rows), "blocks": dict(Counter(r["block"] for r in rows)),
        "impact_beams": len(injector_points(rows)), "linac_ids_unique": len({r["linac_id"] for r in rows}) == len(rows),
        "normalized_inputs": {"min": float(u.min()), "max": float(u.max()),
                              "mean_per_knob_range": [float(u.mean(0).min()), float(u.mean(0).max())]},
        "rows_outside_bundle_box": int(np.sum(np.any((u < 0) | (u > 1), axis=1))),
        "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    (ds / "design_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "generator"}, indent=1))
    ok = (out["physics_matches_reference"] and out["bundle_checkpoint_sha256_ok"] and out["box_matches_bundle"]
          and out["rows_outside_bundle_box"] == 0 and not out["generator"]["git_dirty"])
    raise SystemExit(0 if ok else "design check FAILED")


if __name__ == "__main__":
    main()
