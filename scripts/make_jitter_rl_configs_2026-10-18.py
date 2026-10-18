"""Jitter-aware realistic RL configs: the realistic noblen / bc11loop v3 configs + beam-side per-step phase wander +
the Monte-Carlo jitter map (results/jitter_mc_2026-10-18), with the constant per-image spacing shot noise reduced to the
camera / analysis part (the map carries the jitter part).

  python scripts/make_jitter_rl_configs_2026-10-18.py --map results/jitter_mc_2026-10-18/<map>.npz --camera-sigma-um 8 \
      --wander-deg 0.1 --tag jit
"""
import argparse
from pathlib import Path

import yaml

SRC = {"noblen": "configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml",
       "bc11loop_v3": "configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop_v3.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--camera-sigma-um", type=float, required=True)
    ap.add_argument("--wander-deg", type=float, nargs=2, default=[0.1, 0.1], help="L1, L2 per-step beam-side wander")
    ap.add_argument("--tag", default="jit")
    ap.add_argument("--iters", type=int, default=900)
    a = ap.parse_args()
    out = Path("configs/rl/realistic_2026-10-17")
    for kind, src in SRC.items():
        c = yaml.safe_load(Path(src).read_text())
        de = c["params"]["diff_env"]
        v4 = de["v4"]
        v4["phase_wander"] = {"L1PhaseSet": a.wander_deg[0], "L2PhaseSet": a.wander_deg[1]}
        v4["jitter_map"] = a.map
        de["measurement"]["shot_noise"]["bunch_spacing"] = a.camera_sigma_um * 1e-6
        c["params"]["config"]["max_epochs"] = a.iters
        stem = f"real_covB2_screen_{kind}_{a.tag}"
        de["reward_norms_json"] = f"results/rl/_shared/norms/metric_norms_{stem}.json"
        p = out / f"{stem}.yaml"
        if p.exists():
            raise SystemExit(f"{p} exists")
        p.write_text(f"# {kind} + jitter ({a.tag}): map {a.map}, camera {a.camera_sigma_um} um/image, wander {a.wander_deg} deg/step\n"
                     + yaml.safe_dump(c, sort_keys=False))
        print(p)


if __name__ == "__main__":
    main()
