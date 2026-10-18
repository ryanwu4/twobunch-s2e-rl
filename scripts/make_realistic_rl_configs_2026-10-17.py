"""Realistic-operation RL configs (Ryan 10-17): two directions on an injector-aware flow, constrained by the DAQ
realism pass (results/realism_2026-10-17/REPORT.md; one controller step ~ 20 s):

  noblen   direction 1, NO BLEN anywhere: BC11 + BC14 loops off; L1 via PREQ (direct_rb: hidden beam gain
           U(0.7, 1.3), klystron readback g_rf U(0.95, 1.05), noise 0.02 deg), L2 via the multiknob; the policy sees
           the XTCAV features + the klystron readback only
  bc11loop direction 2, BLEN in a real loop: BC11 loop ON for L1 (blen_loop: the action changes ln(setpoint), the
           solved loop with the BLEN-model DAQ-band cubic and hidden slope scale U(0.8, 1.8) -> setpoint gain
           ~2.6-5.7 deg/ln at the operating point, state-dependent; beam-side drift cancelled, seen in the PACT
           readback (noise 0.10 deg), loop noise 0.05 deg, charge term), L2 via the multiknob with BC14 off; no
           BLEN in the observation

Both: L1 drift RW 0.035 deg/step + ramp U(+-0.04) (rms 0.023 = the DAQ ramp rms); L2 RW 0.05 + ramp U(+-0.04) + jumps (p 0.02/step, N(0, 0.45) deg);
absolute knob window = the controller box (no per-episode budgets on direct actuators); continuous operation:
128-step episodes, a new setpoint every U[16, 48] steps around the current state; 16-step history; action cost 0.05;
1200 iterations.

  python scripts/make_realistic_rl_configs_2026-10-17.py --tag covB2
"""
import argparse
import copy
from pathlib import Path

import yaml

SRC = {"noblen": "configs/rl/injflow_2026-10-16/injflow_{tag}_screen_v6_l1direct_v2.yaml",
       "bc11loop": "configs/rl/injflow_2026-10-16/injflow_{tag}_screen_v6_blen.yaml"}
WINDOW = {"L1PhaseSet": [-26.5, -18.5], "L2PhaseSet": [-41.0, -33.0]}
BC11 = {"center": -20.29, "poly": {"L1PhaseSet": -0.218, "L1PhaseSet^2": 0.0246, "L1PhaseSet^3": 0.00233}}


def build(kind, tag):
    c = yaml.safe_load(Path(SRC[kind].format(tag=tag)).read_text())
    de = c["params"]["diff_env"]
    v4 = de["v4"]
    de["episode_length"] = 128
    v4.update(history=16, action_cost=0.05, goal_change=[16, 48], window=copy.deepcopy(WINDOW),
              drift_per_step={"L1PhaseSet": 0.035, "L2PhaseSet": 0.05},
              drift_ramp={"L1PhaseSet": 0.04, "L2PhaseSet": 0.04},
              jumps={"L2PhaseSet": [0.02, 0.45]})
    v4.pop("blen_obs", None)
    l2 = {"type": "direct", "step": 0.3, "gain_range": [0.35, 1.1]}
    if kind == "noblen":
        l1 = {"type": "direct_rb", "step": 0.25, "gain_range": [0.7, 1.3], "readback_noise": 0.02,
              "readback_gain_range": [0.95, 1.05]}
    else:
        l1 = {"type": "blen_loop", "step": 0.1, "cum_limit": 1.0, "readback_noise": 0.10, **copy.deepcopy(BC11),
              "slope_scale": [0.8, 1.8], "settle": 1.0, "loop_noise": 0.05, "charge_rw": 0.003,
              "charge_sens": [0.4, 2.5]}
    v4["actuators"] = {"L1PhaseSet": l1, "L2PhaseSet": l2}
    c["params"]["config"]["max_epochs"] = 1200
    stem = f"real_{tag}_screen_{kind}"
    de["reward_norms_json"] = f"results/rl/_shared/norms/metric_norms_{stem}.json"
    return stem, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="configs/rl/realistic_2026-10-17")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in SRC:
        stem, c = build(kind, a.tag)
        p = out / f"{stem}.yaml"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_text(f"# {kind}: realistic operation (scripts/make_realistic_rl_configs_2026-10-17.py, results/realism_2026-10-17)\n"
                     + yaml.safe_dump(c, sort_keys=False))
        print(p)


if __name__ == "__main__":
    main()
