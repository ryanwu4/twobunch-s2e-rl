"""Backfill a campaign.json record for a campaign that predates campaign.json — written OUTSIDE its
data dir (data/ is never modified), and passed to preprocess with --campaign-json SUBDIR=PATH.

The record is rebuilt from the campaign's config yaml (what run_sweep ran with) and its manifest
(the knobs actually varied, e.g. xtcav23_valgate also varied the downstream Q2/Q1/Q0FF), exactly in
run_sweep.write_campaign_json's schema, plus `injector_settings` = golden (these campaigns tracked
the golden 2024-10-14 beam = injector at golden) and a `backfill` provenance block. Caveat: the
FACET2-S2E / git hashes are TODAY's, not the generation-time ones (flagged in `backfill`); the
baseline-config / defaults / input-beam shas are of today's files, which is what a merge check
compares anyway.

Usage (bmad-qpad-dev env: needs FACET2_S2E.loadConfig):
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/backfill_campaign_json.py configs/datagen/xtcav23_full.yaml [more.yaml ...] \
      --out-dir results/injector_pilot_2026-10-14/campaign_backfill
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from twobunch_s2e_rl.artifacts import facet2_s2e_hash, git_state
from twobunch_s2e_rl.datagen.paths import facet2_root
from twobunch_s2e_rl.datagen.run_sweep import GOLDEN_BEAM_SUFFIX, TREATY_POINTS, load_cfg
from twobunch_s2e_rl.datagen.sweep_params import injector_golden, resolve_sweep_set


def _sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def backfill(cfg_path) -> dict:
    import FACET2_S2E as qs
    cfg = load_cfg(cfg_path)
    root = facet2_root()
    baseline = qs.loadConfig("/" + cfg["baseline_config"], str(root))
    out_dir = Path(cfg["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text())
    set_name = cfg.get("sweep_set", "original8")
    keys, low, high, _ = resolve_sweep_set(set_name)
    varied = sorted({k for r in manifest for k in r["knobs"]})
    swept = keys + [k for k in varied if k not in keys]      # e.g. valgate's downstream FF quads
    if baseline.get("inputBeamFilePathSuffix") != GOLDEN_BEAM_SUFFIX:
        raise ValueError(f"{cfg['name']}: input beam is not the golden 2024-10-14 beam")
    return {
        "name": cfg["name"], "config": cfg,
        "swept": {"sweep_set": set_name, "knob_keys": swept, "knob_low": low, "knob_high": high},
        "fixed": {
            "baseline_config": cfg["baseline_config"],
            "baseline_config_sha256": _sha(root / cfg["baseline_config"]),
            "defaults_yml_sha256": _sha(root / "setLattice_configs" / "defaults.yml"),
            "baseline_settings": baseline,
            "fixed_overrides": cfg.get("fixed_overrides", {}),
            "input_beam": baseline.get("inputBeamFilePathSuffix"),
            "input_beam_sha256": _sha(str(root) + str(baseline.get("inputBeamFilePathSuffix", ""))),
            "csrTF": cfg["csrTF"], "transverseWakes": cfg.get("transverseWakes", False),
            "num_macro_particles": int(cfg["num_macro_particles"]),
            "treaty_points": cfg.get("treaty_points", TREATY_POINTS),
            "injector_settings": injector_golden(),
        },
        "facet2_s2e_hash": facet2_s2e_hash(), **git_state(),
        "backfill": {"date": datetime.date.today().isoformat(),
                     "by": "scripts/backfill_campaign_json.py",
                     "note": "rebuilt from config yaml + manifest; facet2/git hashes are of the "
                             "backfill time, not of generation",
                     "n_manifest_rows": len(manifest)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c in a.configs:
        rec = backfill(c)
        p = out / f"{Path(rec['config']['output_dir']).name}.json"
        if p.exists():
            print(f"exists, kept: {p}")
            continue
        p.write_text(json.dumps(rec, indent=2, default=float))
        print(f"wrote {p} ({len(rec['swept']['knob_keys'])} swept keys)")


if __name__ == "__main__":
    main()
