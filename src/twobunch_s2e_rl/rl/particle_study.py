"""Particle-count convergence study for the MBRL reward metrics.

Samples the flow at fixed knobs with N in {256..8192} and measures, per reward metric:
- value relative std across independent draws (the per-step reward MC noise the agent sees),
- value relative bias vs a large-N reference,
- GRADIENT noise: the coefficient-of-variation of ||d(metric)/d(knobs)|| across draws -- the arm
  that actually limits first-order MBRL (a reparameterized-gradient estimate can stay noisy even
  when the value has converged).

Recommends the smallest N whose value-std and gradient-CoV fall below tolerances. The feasibility
heads (T_drive/T_witness) are N-independent (no sampling) and excluded.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.rl.particle_study \
      --ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt"

Writes results/rl/particle_study.{json,png}.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..datagen.paths import repo_root
from ..datagen.sweep_params import BASELINE_KNOBS, BOUNDS_HIGH, BOUNDS_LOW, PARAM_KEYS
from ..surrogate.model import TwoBunchFlow

# reward-relevant, sampling-dependent metrics (T_drive/T_witness are heads -> N-independent)
METRICS = ("bunch_spacing", "transverse_offset", "angular_misalignment", "energy_difference",
           "drive_norm_emit_x", "drive_norm_emit_y", "witness_norm_emit_x", "witness_norm_emit_y")
N_GRID = (256, 512, 1024, 2048, 4096, 8192)


def _baseline_norm() -> np.ndarray:
    lo, hi = np.array(BOUNDS_LOW), np.array(BOUNDS_HIGH)
    base = np.array([BASELINE_KNOBS[k] for k in PARAM_KEYS])
    return ((base - lo) / np.maximum(hi - lo, 1e-12)).astype(np.float32)


def _stats_at(flow, knob_pt, N, n_draws, device, grad_draws):
    """Return per-metric (value_mean, value_std) and ||grad||-list across draws at one (pt, N)."""
    vals = {m: [] for m in METRICS}
    grads = {m: [] for m in METRICS}
    for d in range(n_draws):
        knobs = torch.tensor(knob_pt, device=device).unsqueeze(0).requires_grad_(True)  # (1,8)
        obs = flow.observables(knobs, n=N)
        for m in METRICS:
            vals[m].append(float(obs[m].detach().item()))
        if d < grad_draws:
            for m in METRICS:
                g = torch.autograd.grad(obs[m][0], knobs, retain_graph=True)[0]
                grads[m].append(float(g.norm().item()))
    return vals, grads


@torch.no_grad()
def _reference(flow, knob_pt, ref_N, device, reps=4):
    ref = {m: [] for m in METRICS}
    for _ in range(reps):
        knobs = torch.tensor(knob_pt, device=device).unsqueeze(0)
        obs = flow.observables(knobs, n=ref_N)
        for m in METRICS:
            ref[m].append(float(obs[m].item()))
    return {m: float(np.mean(v)) for m, v in ref.items()}


def study(ckpt, device="cuda", n_draws=32, grad_draws=16, ref_N=16384,
          n_points=3, out_dir="results/rl/surrogate", seed=0):
    device = device if torch.cuda.is_available() else "cpu"
    flow = TwoBunchFlow.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    for p in flow.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(seed)
    points = [("baseline", _baseline_norm())]
    for i in range(n_points - 1):
        points.append((f"rand{i}", rng.random(8).astype(np.float32)))

    results = {}      # results[point][N][metric] = {value_mean, value_relstd, value_relbias, grad_mean, grad_cov}
    for name, pt in points:
        ref = _reference(flow, pt, ref_N, device)
        results[name] = {}
        for N in N_GRID:
            vals, grads = _stats_at(flow, pt, N, n_draws, device, grad_draws)
            entry = {}
            for m in METRICS:
                v = np.array(vals[m]); g = np.array(grads[m])
                denom = max(abs(np.mean(v)), 1e-30)
                gm = max(abs(g.mean()), 1e-30)
                entry[m] = {
                    "value_mean": float(v.mean()),
                    "value_relstd": float(v.std() / denom),
                    "value_relbias": float(abs(v.mean() - ref[m]) / max(abs(ref[m]), 1e-30)),
                    "grad_mean": float(g.mean()),
                    "grad_cov": float(g.std() / gm),
                }
            results[name][str(N)] = entry
            med_relstd = float(np.median([entry[m]["value_relstd"] for m in METRICS]))
            med_gcov = float(np.median([entry[m]["grad_cov"] for m in METRICS]))
            print(f"[{name}] N={N:5d}  median value-relstd={med_relstd:.3f}  median grad-CoV={med_gcov:.3f}")

    # recommend: smallest N with median value-relstd<0.02 and median grad-CoV<0.5 (across points)
    rec = None
    for N in N_GRID:
        rs = np.median([results[nm][str(N)][m]["value_relstd"] for nm in results for m in METRICS])
        gc = np.median([results[nm][str(N)][m]["grad_cov"] for nm in results for m in METRICS])
        if rs < 0.02 and gc < 0.5:
            rec = N
            break
    rec = rec or N_GRID[-1]
    print(f"\nRECOMMENDED n_particles = {rec}")

    out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"recommended_n": rec, "n_draws": n_draws, "grad_draws": grad_draws,
               "ref_N": ref_N, "metrics": list(METRICS), "results": results}
    with open(out_dir / "particle_study.json", "w") as f:
        json.dump(payload, f, indent=2)

    # plot: value-relstd and grad-CoV vs N (baseline point), per metric
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    base = results["baseline"]
    for m in METRICS:
        axes[0].plot(N_GRID, [base[str(N)][m]["value_relstd"] for N in N_GRID], marker="o", label=m)
        axes[1].plot(N_GRID, [base[str(N)][m]["grad_cov"] for N in N_GRID], marker="o", label=m)
    for ax, ttl, thr in ((axes[0], "value rel-std", 0.02), (axes[1], "gradient CoV", 0.5)):
        ax.axhline(thr, color="k", ls="--", lw=1, alpha=0.6)
        ax.axvline(rec, color="r", ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("n_particles"); ax.set_title(f"{ttl} (baseline)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Particle-count study — recommended n = {rec}")
    fig.tight_layout()
    fig.savefig(out_dir / "particle_study.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir}/particle_study.json, particle_study.png")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="TwoBunchFlow checkpoint (glob ok)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--grad-draws", type=int, default=16)
    ap.add_argument("--ref-n", type=int, default=16384)
    ap.add_argument("--n-points", type=int, default=3)
    ap.add_argument("--out", default="results/rl/surrogate")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    study(ckpt, device=args.device, n_draws=args.n_draws, grad_draws=args.grad_draws,
          ref_N=args.ref_n, n_points=args.n_points, out_dir=args.out)


if __name__ == "__main__":
    main()
