"""Baseline sanity check for the goal-conditioned MBRL work.

Feed the EXACT hand-tuned 2024-10-14 baseline knobs (`sweep_params.BASELINE_KNOBS`, the golden
two-bunch working point) into BOTH the frozen flow surrogate and Bmad, and report the same
observables the MBRL eval logs (bunch_spacing, per-bunch T, per-bunch norm_emit_x/y) as a
surrogate-vs-Bmad gap. A reference point with a known-good beam: it checks (a) the surrogate's
fidelity at the baseline operating point and (b) that Bmad reproduces the golden working point
(spacing ~202 um) when handed back its own settings.

Same wiring as `eval_bmad.py --mode openloop`, but the knob vector is the fixed baseline instead
of a policy rollout. Surrogate prediction is averaged over `--n-draws` flow samples (matches the
openloop validator). Run from the repo root under the torch env (the bridge spawns its own Bmad
worker under bmad-qpad-dev):

  PYTHONPATH=$PWD/src python scripts/baseline_sanity.py \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt" --num-macro 100000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_s2e_rl.datagen.sweep_params import (
    BASELINE_KNOBS, BOUNDS_HIGH, BOUNDS_LOW, PARAM_KEYS)
from twobunch_s2e_rl.rl._train_utils import build_reward_spec
from twobunch_s2e_rl.rl.bmad_bridge import BmadBridge
from twobunch_s2e_rl.rl.bmad_env import BmadTwoBunchEnv
from twobunch_s2e_rl.rl.eval_bmad import _BASELINE_CFG, _full_charges, _gap_report
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow


def _resolve(p):
    return sorted(glob.glob(p))[-1] if "*" in p else p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--cfg", default="logs/bptt_gc/cfg.yaml",
                    help="run cfg for the reward spec + campaign/charge paths")
    ap.add_argument("--num-macro", type=int, default=100000)
    ap.add_argument("--n-particles", type=int, default=2048)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--no-csr", action="store_true")
    ap.add_argument("--no-wakes", action="store_true")
    ap.add_argument("--skip-bmad", action="store_true", help="surrogate-only quick check")
    ap.add_argument("--save-clouds", default=None,
                    help="npz path to save baseline surrogate + Bmad 6D clouds (for corner plots)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    de = yaml.safe_load(open(args.cfg))["params"]["diff_env"]
    spec = build_reward_spec(de)

    lo, hi = np.array(BOUNDS_LOW), np.array(BOUNDS_HIGH)
    phys = np.array([BASELINE_KNOBS[k] for k in PARAM_KEYS], dtype=np.float64)
    knorm = ((phys - lo) / (hi - lo)).astype(np.float32)
    print("baseline knobs (phys):", {k: round(float(v), 3) for k, v in zip(PARAM_KEYS, phys)})
    print("baseline knobs (norm):", {k: round(float(v), 3) for k, v in zip(PARAM_KEYS, knorm)})

    # ---- surrogate: average several draws at the baseline knobs ----
    flow = TwoBunchFlow.load_from_checkpoint(_resolve(args.flow_ckpt),
                                             map_location=device).to(device).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
    kt = torch.tensor(knorm, device=device).unsqueeze(0)            # (1, 8)
    with torch.no_grad():
        draws = [flow.observables(kt, n=args.n_particles) for _ in range(args.n_draws)]
    surr_ach = {k: torch.stack([d[k] for d in draws]).mean(0) for k in draws[0]}

    out = {"input": "2024-10-14_baseline", "num_macro": args.num_macro,
           "knobs_phys": {k: float(v) for k, v in zip(PARAM_KEYS, phys)}}

    if args.skip_bmad:
        print("\n=== surrogate-only at baseline knobs (um / um-rad) ===")
        for k in ("bunch_spacing", "T_drive", "T_witness",
                  "drive_norm_emit_x", "drive_norm_emit_y",
                  "witness_norm_emit_x", "witness_norm_emit_y"):
            v = float(surr_ach[k].cpu())
            sc = 1e6 if ("emit" in k or k == "bunch_spacing") else 1.0
            print(f"  {k:22s} {v * sc:10.3f}")
        return

    # ---- Bmad: track the same physical knobs once (same masking as eval_bmad openloop) ----
    drive_full, witness_full = _full_charges(de["campaign_h5"])
    bridge = BmadBridge(baseline_config=_BASELINE_CFG, drive_full_nc=drive_full,
                        witness_full_nc=witness_full, num_macro=args.num_macro,
                        csr=not args.no_csr, wakes=not args.no_wakes, P=args.n_particles)
    try:
        benv = BmadTwoBunchEnv(bridge, spec, num_envs=1, device=device, min_particles=64)
        MIN = benv.min_particles
        res = bridge.track(phys.astype(np.float32))
        if not res["ok"]:
            print("Bmad track FAILED:", res.get("error"))
        p = benv._per_env(res)
        nd, nw = int(res["n_drive"]), int(res["n_witness"])
        nan = lambda: torch.full((1,), float("nan"), device=device)
        if (not res["ok"]) or nd < MIN:
            for k in ("drive_norm_emit_x", "drive_norm_emit_y", "drive_norm_emit_4d"):
                if k in p: p[k] = nan()
        if (not res["ok"]) or nw < MIN:
            for k in ("witness_norm_emit_x", "witness_norm_emit_y", "witness_norm_emit_4d"):
                if k in p: p[k] = nan()
        if (not res["ok"]) or nd < MIN or nw < MIN:
            for k in ("bunch_spacing", "transverse_offset", "angular_misalignment"):
                if k in p: p[k] = nan()
        print(f"  tracked: n_drive={nd} n_witness={nw}")
        print("\n=== baseline sanity: surrogate vs Bmad truth at the hand-tuned baseline knobs ===")
        _gap_report(surr_ach, p, out)

        if args.save_clouds:
            sd = flow.sample_bunch(kt, 0, args.n_particles)[0].cpu().numpy()
            sw = flow.sample_bunch(kt, 1, args.n_particles)[0].cpu().numpy()
            Path(args.save_clouds).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_clouds, phys=phys.astype(np.float32),
                     surr_drive=sd, surr_witness=sw,
                     bmad_drive=np.asarray(res["drive"], np.float32),
                     bmad_witness=np.asarray(res["witness"], np.float32))
            print(f"  saved baseline clouds -> {args.save_clouds}")
    finally:
        bridge.close()

    Path("artifacts/rl/openloop").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("artifacts/rl/openloop/eval_baseline_sanity.json", "w"), indent=2)
    print("\nWrote artifacts/rl/openloop/eval_baseline_sanity.json")


if __name__ == "__main__":
    main()
