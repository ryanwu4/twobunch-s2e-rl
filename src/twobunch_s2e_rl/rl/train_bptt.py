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
from ..artifacts import write_run_json
from ._train_utils import add_args, attach_csv_hook, build_env_fn, override, resolve_ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    args = p.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg = override(cfg, args)
    logdir = cfg["params"]["general"]["logdir"]
    Path(logdir).mkdir(parents=True, exist_ok=True)
    flow_ckpt = args.flow_ckpt or cfg["params"]["diff_env"].get("flow_ckpt")
    if not flow_ckpt:
        raise SystemExit("pass --flow-ckpt or set diff_env.flow_ckpt in the config")

    env_fn = build_env_fn(cfg, flow_ckpt)
    algo = BPTT(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        de = cfg["params"]["diff_env"]
        write_run_json(logdir, "policy", cfg, parents=[resolve_ckpt(flow_ckpt), de["campaign_h5"]],
                       flow_ckpt=resolve_ckpt(flow_ckpt), algo="BPTT")
        attach_csv_hook(algo, logdir)
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
