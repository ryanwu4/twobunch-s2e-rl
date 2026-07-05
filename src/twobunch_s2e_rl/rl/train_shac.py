"""SHAC training entry point for first-order MBRL over the two-bunch flow surrogate.

Reward = the composite TwoBunchRewardSpec (spacing target + per-bunch emittance + survival)
computed from the flow's differentiable observables, backpropagated to the 8 knobs. Reuses the
vendored `rl.diffrl.SHAC` unchanged.

Usage:
    python -m twobunch_s2e_rl.rl.train_shac \
        --cfg configs/rl/shac.yaml \
        --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt" \
        --logdir logs/shac --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .diffrl import SHAC
from ._train_utils import add_args, attach_csv_hook, build_env_fn, override


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    args = p.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg = override(cfg, args)
    Path(cfg["params"]["general"]["logdir"]).mkdir(parents=True, exist_ok=True)

    env_fn = build_env_fn(cfg, args.flow_ckpt)
    algo = SHAC(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        attach_csv_hook(algo, cfg["params"]["general"]["logdir"])
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
