"""Exploratory RL configs on an injector-aware flow (overnight 2026-10-16): S1 cloud + S2 v6 screen, blen_dlog (and
the direct-L1 v2 variants), with the 37-knob frame, the DAQ-derived injector context (configs/rl/eval/
injector_ctx_2026-10-16.yaml), the drive charge fraction observed in S1 too, the T hinge back at 0.98 (per-sample
labels), and their own reward-norm caches. Writes configs/rl/injflow_2026-10-16/<stem>.yaml.

  python scripts/make_injflow_rl_configs.py --flow trained/<run>/checkpoints/<ckpt> --tag pairA40
"""
import argparse
from pathlib import Path

import yaml

SRC = {"cloud_blen": "configs/rl/delta_xtcav23_cloud_v4_blen.yaml",
       "screen_v6_blen": "configs/rl/delta_xtcav23_screen_v6_blen.yaml",
       "cloud_l1direct_v2": "configs/rl/delta_xtcav23_cloud_v4_l1direct_v2.yaml",
       "screen_v6_l1direct_v2": "configs/rl/delta_xtcav23_screen_v6_l1direct_v2.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", default="configs/rl/eval/injector_ctx_2026-10-16.yaml")
    a = ap.parse_args()
    if not Path(a.flow).exists():
        raise SystemExit(f"{a.flow} missing")
    ctx = yaml.safe_load(Path(a.ctx).read_text())
    out = Path("configs/rl/injflow_2026-10-16")
    out.mkdir(parents=True, exist_ok=True)
    for stem, src in SRC.items():
        c = yaml.safe_load(Path(src).read_text())
        de = c["params"]["diff_env"]
        de["flow_ckpt"] = a.flow
        de.update(ctx)                                    # baseline_set + context_dist
        if "drive_charge_fraction" not in (de.get("observe") or []):
            de["observe"] = list(de.get("observe") or []) + ["drive_charge_fraction"]
            de.setdefault("measurement", {}).setdefault("shot_noise", {})["drive_charge_fraction"] = 0.07
            de["measurement"].setdefault("step_noise_floor", {})["drive_charge_fraction"] = 0.02
        for o in de["objectives"]:
            if o.get("kind") == "hinge" and o.get("key") in ("T_drive", "T_witness"):
                o["hinge_at"] = 0.98
        de["reward_norms_json"] = f"results/rl/_shared/norms/metric_norms_injflow_{a.tag}_{stem}.json"
        p = out / f"injflow_{a.tag}_{stem}.yaml"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_text(f"# EXPLORATORY (no flow has passed the gates): {Path(src).name} on {a.flow}\n"
                     f"# + injector context {a.ctx}, drive charge fraction observed, T hinge 0.98. Overnight 2026-10-16.\n"
                     + yaml.safe_dump(c, sort_keys=False))
        print(p)


if __name__ == "__main__":
    main()
