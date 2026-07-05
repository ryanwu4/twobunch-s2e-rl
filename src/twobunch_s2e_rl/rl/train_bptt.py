"""BPTT training entry point for first-order MBRL over the two-bunch flow surrogate.

Same env/reward as SHAC; pure backprop through the (truncated) rollout, no critic. Reuses the
vendored `rl.diffrl.BPTT` unchanged. NOTE: `steps_num` is the truncated BPTT horizon -- with
flow sampling, full-episode (=64) backprop is memory-heavy, so the config uses a shorter window.

Usage:
    python -m twobunch_s2e_rl.rl.train_bptt \
        --cfg configs/rl/bptt.yaml \
        --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt" \
        --logdir logs/bptt --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .diffrl import BPTT
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
    algo = BPTT(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        attach_csv_hook(algo, cfg["params"]["general"]["logdir"])
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
