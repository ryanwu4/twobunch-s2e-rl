"""Step-0 go/no-go probe for goal-conditioned (dynamic) bunch spacing.

Goal-conditioning's premise is that the policy can hit an arbitrary target spacing WHILE keeping
the bunches alive and the emittance near floor. Surrogate accuracy is not the question (spacing
R^2=0.996 over the campaign, and 100-300 um is ~+-1 sigma of the campaign mean 204 um, i.e.
interpolation) -- the question is the JOINT feasibility frontier: across the target range, do knob
settings exist with high survival and low emittance?

This samples random knobs through the FROZEN flow, bins the achieved spacing, and reports per bin
the fraction of knobs with both bunches surviving (T>=thresh) and the best (min) witness vertical
emittance among survivors. If a goal sub-range has no high-survival knobs, narrow the range before
training. Read-only; no training, no env.

  PYTHONPATH=$PWD/src python scripts/spacing_feasibility_probe.py \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt" --n 20000
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from twobunch_s2e_rl.surrogate.model import TwoBunchFlow


def _resolve(path: str) -> str:
    return sorted(glob.glob(path))[-1] if "*" in path else path


@torch.no_grad()
def probe(flow_ckpt, n=20000, n_particles=512, t_thresh=0.95, device="cuda",
          goal_lo_um=100.0, goal_hi_um=300.0, batch=2048, seed=0, out=None):
    device = device if torch.cuda.is_available() else "cpu"
    flow = TwoBunchFlow.load_from_checkpoint(_resolve(flow_ckpt), map_location=device).to(device).eval()
    g = torch.Generator(device=device).manual_seed(seed)

    keys = ("bunch_spacing", "T_drive", "T_witness", "witness_norm_emit_y", "drive_norm_emit_x")
    acc = {k: [] for k in keys}
    for i in range(0, n, batch):
        knobs = torch.rand(min(batch, n - i), 8, generator=g, device=device)
        o = flow.observables(knobs, n=n_particles)
        for k in keys:
            acc[k].append(o[k].cpu().numpy())
    d = {k: np.concatenate(v) for k, v in acc.items()}

    spacing_um = d["bunch_spacing"] * 1e6
    alive = (d["T_drive"] >= t_thresh) & (d["T_witness"] >= t_thresh)
    edges = np.arange(50, 401, 25.0)
    print(f"\nProbed {n} random knobs through {Path(_resolve(flow_ckpt)).name}")
    print(f"survival threshold: T_drive & T_witness >= {t_thresh}\n")
    print(f"{'spacing bin (um)':>18s} {'n':>6s} {'n_alive':>8s} {'frac_alive':>11s} "
          f"{'min wit_emit_y(um)':>19s}")
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (spacing_um >= a) & (spacing_um < b)
        nbin = int(m.sum())
        ma = m & alive
        emit = (d["witness_norm_emit_y"][ma] * 1e6)
        min_emit = float(emit.min()) if emit.size else float("nan")
        frac = float(ma.sum() / nbin) if nbin else float("nan")
        flag = "  <-- target" if (b > goal_lo_um and a < goal_hi_um) else ""
        print(f"{f'[{a:.0f},{b:.0f})':>18s} {nbin:6d} {int(ma.sum()):8d} {frac:11.3f} "
              f"{min_emit:19.3f}{flag}")
        rows.append((0.5 * (a + b), nbin, int(ma.sum()), frac, min_emit))

    # verdict over the requested goal range
    in_range = (spacing_um >= goal_lo_um) & (spacing_um < goal_hi_um)
    feasible = in_range & alive
    print(f"\nGoal range [{goal_lo_um:.0f}, {goal_hi_um:.0f}] um: "
          f"{int(in_range.sum())} samples, {int(feasible.sum())} with both bunches alive "
          f"({100.0 * feasible.sum() / max(in_range.sum(), 1):.1f}%).")
    sub = [r for r in rows if goal_lo_um <= r[0] < goal_hi_um]
    dead = [r for r in sub if r[1] >= 20 and (np.isnan(r[3]) or r[3] < 0.02)]
    if dead:
        print("  WARNING: bins with ~no high-survival knobs (consider narrowing the range): "
              + ", ".join(f"~{int(r[0])}um" for r in dead))
    else:
        print("  OK: every populated target bin has high-survival knobs -> range is feasible.")

    out = Path(out) if out else Path("artifacts/rl/spacing_feasibility.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    s = spacing_um
    ax[0].scatter(s[~alive], d["T_witness"][~alive], s=3, alpha=0.2, color="#999", label="scraped")
    ax[0].scatter(s[alive], d["T_witness"][alive], s=3, alpha=0.3, color="#1f77b4", label="alive")
    ax[0].axvspan(goal_lo_um, goal_hi_um, color="orange", alpha=0.12, label="goal range")
    ax[0].axhline(t_thresh, color="r", ls="--", lw=1)
    ax[0].set(xlabel="bunch spacing [um]", ylabel="T_witness", xlim=(0, 400), title="survival vs spacing")
    ax[0].legend(fontsize=8, markerscale=3)
    ax[1].scatter(s[alive], d["witness_norm_emit_y"][alive] * 1e6, s=3, alpha=0.3, color="#2ca02c")
    ax[1].axvspan(goal_lo_um, goal_hi_um, color="orange", alpha=0.12)
    ax[1].set(xlabel="bunch spacing [um]", ylabel="witness norm_emit_y [um-rad]", xlim=(0, 400),
              title="witness emittance vs spacing (survivors)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-particles", type=int, default=512)
    ap.add_argument("--t-thresh", type=float, default=0.95)
    ap.add_argument("--goal-lo-um", type=float, default=100.0)
    ap.add_argument("--goal-hi-um", type=float, default=300.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    probe(args.flow_ckpt, n=args.n, n_particles=args.n_particles, t_thresh=args.t_thresh,
          device=args.device, goal_lo_um=args.goal_lo_um, goal_hi_um=args.goal_hi_um,
          seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
