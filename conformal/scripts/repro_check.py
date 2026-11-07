"""Reproduction check: re-run rows of a campaign the model was trained on, with the generator that makes the
calibration data, and require identical outputs (FACET2-S2E here is deterministic given settings and seeds).
Passing shows the calibration data come from the same simulator as the training data.

  prepare: copy the chosen source rows (same knobs, injector settings and linac_id, so the same Tao seed)
           into the check's data dir as its manifest.json; the check campaign then runs them
  compare: compare every re-run row with its source row (seed, injector beam record, getBeamSpecs at every
           treaty point, every dataset in the saved beams) and write <check dir>/result.json

  python conformal/scripts/repro_check.py prepare conformal/datasets/repro_check_2026-11-06 --source data/injector_cov_2026-10-16
  python conformal/scripts/repro_check.py compare conformal/datasets/repro_check_2026-11-06 --source data/injector_cov_2026-10-16
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import yaml

# one row of each kind the source campaign has: golden injector, full-box injector, injector-context
# (local) injector, and the controller and full linac regions
KINDS = [("paired_golden", "controller", "full"), ("nested", "full", "full"),
         ("nested", "controller", "local"), ("crossed", "shared", "local")]


def pick_rows(source: Path) -> list[dict]:
    rows = json.loads((source / "manifest.json").read_text())
    out = []
    for block, lreg, ireg in KINDS:
        r = next(r for r in rows if (r["block"], r.get("linac_region"), r.get("injector_region")) == (block, lreg, ireg)
                 and json.loads((source / f"sample_{r['idx']:05d}.json").read_text()).get("success"))
        out.append(r)
    return out


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def h5_datasets(path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, o: out.__setitem__(name, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def compare_row(gate_dir: Path, source: Path, g: dict, s_idx: int, planes) -> dict:
    gj = json.loads((gate_dir / f"sample_{g['idx']:05d}.json").read_text())
    sj = json.loads((source / f"sample_{s_idx:05d}.json").read_text())
    res = {"gate_idx": g["idx"], "source_idx": s_idx, "kind": g["block"], "gate_success": gj.get("success"),
           "seed_equal": gj.get("bmad_random_seed") == sj.get("bmad_random_seed"),
           "knobs_equal": same(gj["knobs"], sj["knobs"]),
           "injector_record_equal": same({k: v for k, v in gj["injector"].items() if k != "impact_wall_s"},
                                         {k: v for k, v in sj["injector"].items() if k != "impact_wall_s"}),
           "specs_equal": {ele: same(gj["specs"].get(ele), sj["specs"].get(ele)) for ele in sj["specs"]}}
    for ele in planes:
        a = h5_datasets(gate_dir / f"sample_{g['idx']:05d}_{ele}.h5")
        b = h5_datasets(source / f"sample_{s_idx:05d}_{ele}.h5")
        res[f"beam_{ele}_identical"] = a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    res["pass"] = bool(res["gate_success"] and res["seed_equal"] and res["knobs_equal"]
                       and res["injector_record_equal"] and all(res["specs_equal"].values())
                       and all(v for k, v in res.items() if k.startswith("beam_")))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "compare"])
    ap.add_argument("gate", help="datasets/<check name> (holds campaign.yaml)")
    ap.add_argument("--source", required=True, help="data dir of the source campaign")
    a = ap.parse_args()
    gate, source = Path(a.gate), Path(a.source)
    cfg = yaml.safe_load((gate / "campaign.yaml").read_text())
    out_dir = Path(cfg["output_dir"])
    if a.action == "prepare":
        rows = pick_rows(source)
        gate_rows = [{**r, "idx": i, "source_idx": r["idx"]} for i, r in enumerate(rows)]
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "manifest.json").write_text(json.dumps(gate_rows, indent=1))
        print(f"{len(gate_rows)} rows -> {out_dir}/manifest.json:",
              [(r["idx"], r["source_idx"], r["block"], r["linac_region"], r["injector_region"]) for r in gate_rows])
        return
    rows = json.loads((out_dir / "manifest.json").read_text())
    # planes saved by both campaigns
    planes = [p for p in cfg["save_beams_at"]
              if (source / f"sample_{rows[0]['source_idx']:05d}_{p}.h5").exists()]
    results = [compare_row(out_dir, source, r, r["source_idx"], planes) for r in rows]
    verdict = {"pass": all(r["pass"] for r in results), "source": str(source), "planes_compared": planes,
               "rows": results}
    (gate / "result.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    raise SystemExit(0 if verdict["pass"] else "reproduction check FAILED")


if __name__ == "__main__":
    main()
