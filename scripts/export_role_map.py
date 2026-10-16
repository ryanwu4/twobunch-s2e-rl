"""Export a flow's exact train / val / test row roles ({'campaign/idx': role}) so later tables (re-preprocessed,
or merged with a top-up campaign) reproduce the split: pass it as train.py --role-map. Rows not in the map
inherit their split group's role (TwoBunchFlowDataModule), so top-up rows take their partner's role.

  PYTHONPATH=src python scripts/export_role_map.py --processed processed/xtcav23e_inj_2026-10-15.h5 \\
      --orig-split processed/xtcav23_combined.h5=xtcav23_full,xtcav23_pilot --out processed/role_map_ftc_s0_2026-10-16.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--orig-split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    h5, camps = a.orig_split.split("=", 1)
    dm = TwoBunchFlowDataModule(processed_h5=a.processed, split="group", orig_split=orig_split_keys(h5, camps.split(",")))
    dm.setup()
    keys = dm.full.row_keys
    role = {}
    for name, sub in (("train", dm.train), ("val", dm.val), ("test", dm.test)):
        for i in sub.indices:
            role[keys[int(i)]] = name
    Path(a.out).write_text(json.dumps({"source": a.processed, "orig_split": a.orig_split, "roles": role}, indent=0))
    print(a.out, Counter(role.values()))


if __name__ == "__main__":
    main()
