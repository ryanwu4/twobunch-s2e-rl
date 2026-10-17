"""Direct-L1 + BLEN-observation RL configs (2026-10-16, Ryan: the beamtime design after the ops notes): ops tune L1 by
hand with KLYS:LI11:{11,21}:PREQ (common mode) with the BC11 BLEN loop off and then re-lock the loop, so the controller
moves L1 directly (direct_rb, as direct-L1 v2) and OBSERVES the free BC11 BLEN plus the BC14 BLEN (readback of the
realised L2; BC14 loop off with the L2 multiknob). BLEN model v1 (results/blen_model_2026-10-16):
  BC11 (DAQ-band fit, golden-injector rows): ln S = -0.218 d + 0.0246 d^2 + 0.00233 d^3, d = L1 + 20.29 deg; hidden
       per-episode slope scale U(0.8, 1.3); noise 0.025 ln per log-ratio (shot rms ~5% / sqrt(30), 0.015 model floor,
       x sqrt 2 for the start reading)
  BC14 (lo band, controller box incl. varied injectors, R^2 0.95, residual 0.10 ln): ln S = 0.199 d2 + 0.005 d2^2
       - 0.114 d1 - 0.010 d1^2 + 0.014 d1 d2, d2 = L2 + 37, d1 = L1 + 22; slope 0.20/deg vs the DAQ 0.10-0.42 per
       multiknob degree (x the hidden L2 gain) -> slope scale U(0.7, 1.5); noise 0.035 ln (shot rms ~10% / sqrt(30))
Injector terms are constant within an episode and cancel in the log-ratio to the start reading.

  python scripts/make_l1blen_rl_configs.py --src configs/rl/injflow_2026-10-16/injflow_flowA_screen_v6_l1direct_v2.yaml ...
writes configs/rl/l1blen_2026-10-16/<src stem with l1direct_v2 -> l1blen>.yaml
"""
import argparse
from pathlib import Path

import yaml

BLEN_OBS = {
    "BLEN_LI11_359": {"centers": {"L1PhaseSet": -20.29},
                      "poly": {"L1PhaseSet": -0.218, "L1PhaseSet^2": 0.0246, "L1PhaseSet^3": 0.00233},
                      "slope_scale": [0.8, 1.3], "noise": 0.025},
    "BLEN_LI14_888": {"centers": {"L1PhaseSet": -22.0, "L2PhaseSet": -37.0},
                      "poly": {"L2PhaseSet": 0.199, "L2PhaseSet^2": 0.005, "L1PhaseSet": -0.114,
                               "L1PhaseSet^2": -0.010, "L1PhaseSet*L2PhaseSet": 0.014},
                      "slope_scale": [0.7, 1.5], "noise": 0.035},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default="configs/rl/l1blen_2026-10-16")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for src in a.src:
        c = yaml.safe_load(Path(src).read_text())
        de = c["params"]["diff_env"]
        if de["v4"]["actuators"]["L1PhaseSet"]["type"] != "direct_rb":
            raise SystemExit(f"{src}: L1 actuator is not direct_rb")
        de["v4"]["blen_obs"] = BLEN_OBS
        stem = Path(src).stem.replace("l1direct_v2", "l1blen")
        de["reward_norms_json"] = f"results/rl/_shared/norms/metric_norms_{stem}.json"
        p = out / f"{stem}.yaml"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_text(f"# Direct-L1 + BLEN obs (BC11 + BC14 log-ratios; scripts/make_l1blen_rl_configs.py) on {Path(src).name}\n"
                     + yaml.safe_dump(c, sort_keys=False))
        print(p)


if __name__ == "__main__":
    main()
