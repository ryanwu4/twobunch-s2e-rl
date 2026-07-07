"""Head-to-head SHAC vs BPTT comparison: deterministic eval of each run's best_policy on the
same held-out random starts, printed as a table + saved to results/rl/_shared/compare.json.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.rl.compare \
      --shac logs/shac --bptt logs/bptt \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt"
"""
from __future__ import annotations

import argparse
import json

from ..datagen.paths import rl_shared_dir
from .eval import evaluate


def _row(name, r):
    s, sv, em = r["spacing_um"], r["survival"], r["emittance_um_rad_median"]
    return (f"{name:6s} | spacing {s['median']:6.1f} um (|Δ200|={s['abs_err_to_200_median']:5.1f}) "
            f"| both>90% {sv['frac_both_over_90']*100:5.1f}% "
            f"(Td {sv['T_drive_median']:.2f}/Tw {sv['T_witness_median']:.2f}) "
            f"| εw_x/y {em['witness_norm_emit_x']:.1f}/{em['witness_norm_emit_y']:.1f} "
            f"| pin {r['knob_boundary_pin_frac']*100:.1f}% | R {r['reward_mean']:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shac", default="logs/shac")
    ap.add_argument("--bptt", default="logs/bptt")
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--num-rollouts", type=int, default=256)
    ap.add_argument("--rf-drift-std", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = {}
    for name, logdir in (("SHAC", args.shac), ("BPTT", args.bptt)):
        try:
            out[name] = evaluate(logdir, args.flow_ckpt, policy=args.policy,
                                 num_rollouts=args.num_rollouts, rf_drift_std=args.rf_drift_std,
                                 device=args.device)
        except FileNotFoundError as e:
            print(f"[skip {name}] {e}")

    print("\n==== SHAC vs BPTT (held-out random starts) ====")
    for name in out:
        print(_row(name, out[name]))
    p = rl_shared_dir() / "compare.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
