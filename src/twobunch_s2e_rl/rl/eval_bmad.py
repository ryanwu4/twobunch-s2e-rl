"""Evaluate a trained two-bunch policy on real FACET2-S2E (Bmad) physics.

Two modes:

  --mode openloop  (default):  roll the policy on the SURROGATE to its converged knobs, then
        track those knobs ONCE each on Bmad and compare surrogate-predicted vs Bmad-true
        observables AND beam phase space. A large gap = the policy exploited surrogate error.
        Cost = n_points Bmad tracks (~minutes each; ~20 min at 100k). Saves per-point
        phase-space overlay PNGs + a clouds npz (so plots can be restyled without re-tracking).

  --mode closedloop:  run the policy with Bmad IN THE LOOP for --episode-len steps; report the
        Bmad-true achieved trajectory. Cost = num_envs * episode_len tracks.

Bmad fidelity defaults match the campaign that trained the surrogate (CSR + transverse wakes,
2024-10-14 baseline). --num-macro defaults LOWER than the campaign's 100k for speed; raise to
100000 to match (emittance needs it; spacing/survival/destruction show at any fidelity).

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.rl.eval_bmad --logdir logs/bptt \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt" --mode openloop \
      --n-points 1 --num-macro 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..datagen.sweep_params import BOUNDS_HIGH, BOUNDS_LOW
from ._train_utils import build_reward_spec, resolve_ckpt
from .bmad_bridge import BmadBridge
from .bmad_env import BmadTwoBunchEnv
from .diff_env import N_KNOB, TwoBunchFlowEnv
from .eval import _load_policy
from .reward import EMIT_KEYS

REPORT = ("bunch_spacing", "T_drive", "T_witness") + EMIT_KEYS
_BASELINE_CFG = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
# phase-space planes: (col_i, col_j, xlabel, ylabel, x_scale, y_scale) on (x,y,z,px,py,pz)
_PLANES = [(2, 0, "z [mm]", "x [mm]", 1e3, 1e3),
           (0, 3, "x [mm]", "px [MeV/c]", 1e3, 1e-6),
           (2, 5, "z [mm]", "pz [GeV/c]", 1e3, 1e-9)]


def _full_charges(campaign_h5: str):
    norm = json.load(open(str(campaign_h5).replace(".h5", "_norm.json")))
    return norm["drive_full_charge_nC"], norm["witness_full_charge_nC"]


@torch.no_grad()
def _rollout(env, actor, rms):
    obs = env.reset()
    for _ in range(env.episode_length):
        o = rms.normalize(obs) if rms is not None else obs
        obs, _, _, info = env.step(torch.tanh(actor(o, deterministic=True)))
    # converged knobs are the PRE-reset state (env._knobs has been re-randomized by the
    # episode-end auto-reset); read obs_before_reset, consistent with info["achieved"].
    return info["achieved"], info["obs_before_reset"][:, :N_KNOB].detach()


def _med(ach, k, sc=1.0):
    # nan-median: invalid (failed/scraped-bunch) points are NaN'd in the openloop validator so
    # they are excluded here rather than fabricating agreement via neutral substitutes.
    a = ach[k].detach().cpu().numpy()
    return float(np.nanmedian(a)) * sc if np.isfinite(a).any() else float("nan")


def _ss(a, m=3000, seed=0):
    a = np.asarray(a)
    if len(a) > m:
        return a[np.random.default_rng(seed).choice(len(a), m, replace=False)]
    return a


def _plot_phase_space(sd, sw, bd, bw, sc, bc, title, out_png):
    """Overlay surrogate (filled) vs Bmad (x) clouds, rows=drive/witness, cols=phase planes."""
    sd, sw, bd, bw = (_ss(x) for x in (sd, sw, bd, bw))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # surrogate=blue, bmad=red for ALL bunches; rows are labeled drive/witness.
    SURR_C, BMAD_C = "#1f77b4", "#d62728"
    rows = [("drive", sd, bd), ("witness", sw, bw)]
    for r, (nm, s, b) in enumerate(rows):
        for c, (i, j, xl, yl, sx, sy) in enumerate(_PLANES):
            ax = axes[r, c]
            if len(s):
                ax.scatter(s[:, i] * sx, s[:, j] * sy, s=3, alpha=0.3, color=SURR_C,
                           label=f"surrogate (n={len(s)})")
            if len(b):
                ax.scatter(b[:, i] * sx, b[:, j] * sy, s=7, alpha=0.45, color=BMAD_C, marker="x",
                           label=f"bmad (n={len(b)})")
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{nm}\n{yl}" if c == 0 else yl)
        axes[r, 0].legend(fontsize=8, markerscale=2, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _gap_report(surr_ach, bmad_ach, out):
    n_valid = {k: int(np.isfinite(bmad_ach[k].detach().cpu().numpy()).sum()) for k in REPORT}
    print(f"\n{'metric':22s} {'surrogate':>12s} {'bmad':>12s}   gap   (bmad n_valid)")
    for k in REPORT:
        s, b = _med(surr_ach, k), _med(bmad_ach, k)
        u = "um-rad" if "emit" in k else ("um" if k == "bunch_spacing" else "")
        sc = 1e6 if ("emit" in k or k == "bunch_spacing") else 1.0
        print(f"  {k:22s} {s*sc:12.3f} {b*sc:12.3f}   Δ={(b-s)*sc:+.3f} {u}   (n={n_valid[k]})")
        out.setdefault("gap", {})[k] = {"surr_med": s, "bmad_med": b, "bmad_n_valid": n_valid[k]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--mode", default="openloop", choices=["openloop", "closedloop"])
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--n-points", type=int, default=1)
    ap.add_argument("--episode-len", type=int, default=None)
    ap.add_argument("--num-macro", type=int, default=20000)
    ap.add_argument("--n-particles", type=int, default=2048)
    ap.add_argument("--no-csr", action="store_true")
    ap.add_argument("--no-wakes", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    logdir = Path(args.logdir)
    cfg = yaml.safe_load(open(logdir / "cfg.yaml"))
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    flow_ckpt = resolve_ckpt(args.flow_ckpt)
    drive_full, witness_full = _full_charges(de["campaign_h5"])
    actor, rms = _load_policy(str(logdir / f"{args.policy}_policy.pt"), device)
    artdir = Path("artifacts/rl")
    artdir.mkdir(parents=True, exist_ok=True)

    bridge = BmadBridge(baseline_config=_BASELINE_CFG, drive_full_nc=drive_full,
                        witness_full_nc=witness_full, num_macro=args.num_macro,
                        csr=not args.no_csr, wakes=not args.no_wakes, P=args.n_particles)
    out = {"logdir": str(logdir), "mode": args.mode, "num_macro": args.num_macro}
    try:
        if args.mode == "openloop":
            senv = TwoBunchFlowEnv(args.n_points, device=device, seed=args.seed,
                                   episode_length=int(de["episode_length"]), stochastic_init=True,
                                   no_grad=True, flow_ckpt=flow_ckpt, reward_spec=spec,
                                   n_particles=args.n_particles, action_scale=float(de["action_scale"]))
            _, knobs = _rollout(senv, actor, rms)
            flow = senv._flow
            # stabilize the surrogate prediction: average over several flow draws at the converged
            # knobs (a single env draw can be high-variance at a borderline operating point).
            with torch.no_grad():
                draws = [flow.observables(knobs, n=args.n_particles) for _ in range(8)]
            surr_ach = {k: torch.stack([d[k] for d in draws]).mean(0) for k in draws[0]}
            sd = flow.sample_bunch(knobs, 0, args.n_particles).cpu().numpy()   # (npts, n, 6)
            sw = flow.sample_bunch(knobs, 1, args.n_particles).cpu().numpy()

            lo = torch.tensor(BOUNDS_LOW, device=device); hi = torch.tensor(BOUNDS_HIGH, device=device)
            phys = (lo + knobs * (hi - lo)).cpu().numpy()
            benv = BmadTwoBunchEnv(bridge, spec, num_envs=1, device=device, min_particles=64)
            MIN = benv.min_particles
            per, bd, bw = [], [], []
            for i in range(args.n_points):
                res = bridge.track(phys[i])
                if not res["ok"]:
                    print(f"  [point {i}] Bmad track FAILED: {res['error']}")
                p = benv._per_env(res)
                # VALIDATION integrity: a failed track or a bunch scraped below min_particles has
                # no meaningful emittance/spacing (_per_env substitutes neutral campaign-mean /
                # target values, which would fabricate agreement). NaN those out so the gap report
                # excludes them; keep the real charge-fraction T (valid even at 0 = destroyed).
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
                per.append(p)
                bd.append(np.asarray(res["drive"], np.float32))
                bw.append(np.asarray(res["witness"], np.float32))
                print(f"  [point {i}] tracked: n_drive={nd} n_witness={nw}"
                      f"{'  (witness scraped < min)' if nw < MIN else ''}")
            bmad_ach = {k: torch.cat([p[k] for p in per], dim=0) for k in per[0]}

            print("\n=== open-loop: surrogate prediction vs Bmad truth at the policy's knobs ===")
            _gap_report(surr_ach, bmad_ach, out)

            np.savez(artdir / f"clouds_{logdir.name}.npz", knobs=knobs.cpu().numpy(),
                     phys=phys, surr_drive=sd, surr_witness=sw,
                     **{f"bmad_drive_{i}": bd[i] for i in range(args.n_points)},
                     **{f"bmad_witness_{i}": bw[i] for i in range(args.n_points)})
            if not args.no_plot:
                for i in range(args.n_points):
                    g = lambda a, k: float(a[k].cpu().numpy()[i])
                    title = (f"{logdir.name.upper()} point {i}: surrogate vs Bmad @ converged knobs   "
                             f"|  spacing surr {g(surr_ach,'bunch_spacing')*1e6:.0f} / "
                             f"bmad {g(bmad_ach,'bunch_spacing')*1e6:.0f} um   "
                             f"|  T_witness surr {g(surr_ach,'T_witness'):.2f} / "
                             f"bmad {g(bmad_ach,'T_witness'):.2f}   "
                             f"T_drive surr {g(surr_ach,'T_drive'):.2f} / bmad {g(bmad_ach,'T_drive'):.2f}")
                    png = artdir / f"phasespace_{logdir.name}_pt{i}.png"
                    _plot_phase_space(sd[i], sw[i], bd[i], bw[i], surr_ach, bmad_ach, title, png)
                    print(f"  wrote {png}")
        else:
            ep = args.episode_len or int(de["episode_length"])
            benv = BmadTwoBunchEnv(bridge, spec, num_envs=1, device=device, seed=args.seed,
                                   episode_length=ep, action_scale=float(de["action_scale"]),
                                   stochastic_init=True, min_particles=64)
            obs = benv.reset()
            print(f"\n=== closed-loop on Bmad ({ep} steps) ===")
            for t in range(ep):
                o = rms.normalize(obs) if rms is not None else obs
                obs, r, d, info = benv.step(torch.tanh(actor(o, deterministic=True)))
                a = info["achieved"]
                print(f"step {t+1:2d}  spacing {_med(a,'bunch_spacing',1e6):6.1f}um  "
                      f"T_d/T_w {_med(a,'T_drive'):.2f}/{_med(a,'T_witness'):.2f}  R={float(r.mean()):+.2f}")
            out["final"] = {k: _med(info["achieved"], k) for k in REPORT}
    finally:
        bridge.close()

    json.dump(out, open(artdir / f"eval_bmad_{logdir.name}_{args.mode}.json", "w"), indent=2)
    print(f"\nWrote artifacts/rl/eval_bmad_{logdir.name}_{args.mode}.json")


if __name__ == "__main__":
    main()
