"""Derive final-focus (FF) quad sweep ranges from a PENT beta-matching solve.

Motivation
----------
FACET-II feedback (2026-06-19) advised deriving the FF quad ranges for the next
two-bunch campaign empirically, by their recipe: set the PENT beta function to a
target (e.g. 5 cm) and read off the magnet values to understand each quad's range.
This also resolves a target-beta discrepancy: the code's golden PENT target is
beta = 0.5 m (UTILITY_quickstart.py getBeamSpecs), a legacy comment says 25 cm, and
Yiheng quoted ~5 cm slice / <=15 cm matched.

What this does
--------------
Matches the design Twiss (beta_x = beta_y at PENT, alpha = 0) by varying the 6 FF
telescope quads Q5FF..Q0FF, across target beta ~2 -> 50 cm, and reads off the solved
strengths -> derived per-quad ranges. Also scans the waist longitudinal position
(s_offset) about PENT at a feasible target beta (the "drive waist near PENT" authority
FACET raised).

The match is a single-particle DESIGN OPTICS calculation (beam-independent) -- exactly
what "set beta, read the magnets" means for bounding magnet ranges. It does NOT verify
the real chirped witness slice-beta (the heavier tracking study, deliberately deferred).

Optimizer note
--------------
The repo's `finalFocusSolver` uses scipy Nelder-Mead, which we found under-converges for
aggressive targets (beta < ~20 cm): it stalls without pinning any quad at a bound (see
results_*_neldermead.csv). This study instead uses a bounded least-squares (TRF) match on
the Twiss residual vector with continuation (warm-start from the previous beta), which is
the appropriate tool for a 4-residual / 6-parameter Twiss match. Same 6 quads, same EPICS
bounds, same `twiss_at_s` evaluation as the repo solver.

Run
---
    conda run -n bmad-qpad-dev python analysis/2026-06-19_ff-quad-range/run_ff_quad_range.py
"""

import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import FACET2_S2E as qs
from twobunch_s2e_rl.datagen.paths import facet2_root

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
REPO = str(facet2_root())   # FACET2-S2E checkout ($FACET2_S2E_ROOT or the installed FACET2_S2E)
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# The 6 FF telescope quads, with EPICS bounds mirroring finalFocusSolver
# (UTILITY_finalFocusSolver.py:129) / bounds.yml.
FF_QUADS = ["Q5FF", "Q4FF", "Q3FF", "Q2FF", "Q1FF", "Q0FF"]
EPICS_BOUNDS = {
    "Q5FF": (-256.0, 0.0),
    "Q4FF": (-446.0, 0.0),
    "Q3FF": (0.0, 457.0),
    "Q2FF": (0.0, 167.0),
    "Q1FF": (-257.0, 0.0),
    "Q0FF": (0.0, 167.0),
}
BOUND_LO = np.array([EPICS_BOUNDS[q][0] for q in FF_QUADS])
BOUND_HI = np.array([EPICS_BOUNDS[q][1] for q in FF_QUADS])

# Beta sweep grid (m): ~2 cm -> 50 cm, log-spaced (dense, for continuation).
BETA_GRID = np.geomspace(0.02, 0.50, 25)
# Waist scan: longitudinal offset range about PENT (m); target beta chosen adaptively.
WAIST_OFFSETS = np.round(np.arange(-0.5, 0.55, 0.1), 3)
# Targets for the beta-function-vs-s figure (filled with feasible values at runtime).
BETAS_FOR_PROFILE = [0.05, 0.15, 0.50]

# Diagnostics tolerances.
BETA_REL_TOL = 0.02      # achieved beta within 2% of target in both planes -> "feasible"
ALPHA_ABS_TOL = 0.02     # achieved |alpha| below this -> matched
BOUND_FRAC_TOL = 0.02    # solved value within 2% of bound span of a rail -> "pinned"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def read_ff(tao, quads=FF_QUADS):
    """Read current FF quad integrated strengths (kG.m)."""
    return {q: qs.getQuadkG(tao, q) for q in quads}


def set_quads(tao, x):
    """Set the 6 FF quads from a vector (kG.m), order = FF_QUADS."""
    tao.cmd("set global lattice_calc_on = F")
    for q, v in zip(FF_QUADS, x):
        qs.setQuadkG(tao, q, float(v))
    tao.cmd("set global lattice_calc_on = T")


def twiss_at(tao, ele="PENT", s_offset=0.0):
    t = tao.twiss_at_s(ele=ele, s_offset=s_offset)
    return dict(beta_x=t["beta_a"], alpha_x=t["alpha_a"],
                beta_y=t["beta_b"], alpha_y=t["alpha_b"])


def bound_pinned(quad, value):
    lo, hi = EPICS_BOUNDS[quad]
    span = hi - lo
    return min(value - lo, hi - value) < BOUND_FRAC_TOL * span


def _residuals(x, tao, ele, s_offset, tb):
    set_quads(tao, x)
    t = tao.twiss_at_s(ele=ele, s_offset=s_offset)
    # alpha residuals weighted up so a tight waist (alpha->0) is enforced alongside beta.
    return [t["beta_a"] - tb, t["beta_b"] - tb, 0.1 * t["alpha_a"], 0.1 * t["alpha_b"]]


def solve_match(tao, target_beta, s_offset, x0):
    """Bounded least-squares (TRF) Twiss match at PENT(+s_offset); warm-start from x0."""
    res = least_squares(
        _residuals, np.clip(x0, BOUND_LO, BOUND_HI),
        bounds=(BOUND_LO, BOUND_HI),
        args=(tao, "PENT", s_offset, target_beta),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200,  # bounded: infeasible pts fail fast
    )
    x = res.x
    set_quads(tao, x)                         # leave lattice at the solution
    ach = twiss_at(tao, "PENT", s_offset)
    rel_err = max(abs(ach["beta_x"] - target_beta),
                  abs(ach["beta_y"] - target_beta)) / target_beta
    feasible = (rel_err < BETA_REL_TOL
                and abs(ach["alpha_x"]) < ALPHA_ABS_TOL
                and abs(ach["alpha_y"]) < ALPHA_ABS_TOL)
    solved = {q: float(x[i]) for i, q in enumerate(FF_QUADS)}
    pinned = {q: bound_pinned(q, solved[q]) for q in FF_QUADS}
    return solved, ach, rel_err, feasible, pinned


def write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Initializing Tao (golden lattice, no beam needed for the optics match)...")
    tao = qs.initializeTao(filePath=REPO, runSetLatticeTF=True, runImpactTF=False)

    golden_named = read_ff(tao)
    golden_vec = np.array([golden_named[q] for q in FF_QUADS])
    base = twiss_at(tao, "PENT", 0.0)
    print("Golden FF quads (kG.m): " + ", ".join(f"{q}={golden_named[q]:.3f}" for q in FF_QUADS))
    print(f"Baseline PENT Twiss: beta_x={base['beta_x']:.4f} m, beta_y={base['beta_y']:.4f} m, "
          f"alpha_x={base['alpha_x']:.3f}, alpha_y={base['alpha_y']:.3f}")
    if not (0.4 < base["beta_x"] < 0.6 and 0.4 < base["beta_y"] < 0.6):
        print("  WARNING: baseline PENT beta is not ~0.5 m -- check the lattice config!")

    # ---- 1) Symmetric beta sweep (continuation from golden, high beta -> low beta) -----
    print("\n" + "=" * 80)
    print("Beta sweep (matching beta_x = beta_y at PENT, alpha = 0, bounded least-squares)...")
    targets = np.sort(BETA_GRID)[::-1]   # descending: start near golden 0.5 m
    beta_records, beta_rows = [], []
    prev = golden_vec.copy()
    consec_infeasible = 0
    for bt in targets:
        solved, ach, rel_err, feasible, pinned = solve_match(tao, bt, 0.0, prev)
        prev = np.array([solved[q] for q in FF_QUADS])   # ALWAYS continue along the branch
        flagged = [q for q in FF_QUADS if pinned[q]]
        print(f"  beta*={bt*100:5.1f} cm | achieved bx={ach['beta_x']*100:5.1f} "
              f"by={ach['beta_y']*100:5.1f} cm | relerr={rel_err:6.1%} | "
              f"feasible={feasible} | pinned={flagged}")
        beta_records.append(dict(target=bt, solved=solved, ach=ach,
                                 rel_err=rel_err, feasible=feasible, pinned=pinned))
        # beta descends monotonically; once we are clearly past the feasible floor, stop
        # (keeps 2 infeasible points to show the wall, avoids grinding deep-beta failures).
        consec_infeasible = consec_infeasible + 1 if not feasible else 0
        if consec_infeasible >= 2:
            print(f"  -> stopping descent: {consec_infeasible} consecutive infeasible "
                  f"(past the floor)")
            break
        beta_rows.append([f"{bt:.4f}"]
                         + [f"{solved[q]:.3f}" for q in FF_QUADS]
                         + [f"{ach['beta_x']:.4f}", f"{ach['beta_y']:.4f}",
                            f"{ach['alpha_x']:.3f}", f"{ach['alpha_y']:.3f}",
                            f"{rel_err:.4f}", str(feasible),
                            ";".join(flagged)])
    write_csv(HERE / "results_beta_sweep.csv",
              ["target_beta_m"] + [f"{q}_kGm" for q in FF_QUADS]
              + ["ach_beta_x_m", "ach_beta_y_m", "ach_alpha_x", "ach_alpha_y",
                 "rel_beta_err", "feasible", "pinned_quads"],
              beta_rows)

    feasible_betas = sorted(r["target"] for r in beta_records if r["feasible"])
    beta_floor = feasible_betas[0] if feasible_betas else float("nan")
    print(f"\nSmallest feasible PENT beta with the 6 FF quads: "
          f"{beta_floor*100:.1f} cm" if feasible_betas else "\nNo feasible target!")

    # ---- 2) Waist-location scan at a robustly-feasible target beta ----------------------
    # Use ~15 cm (operationally relevant and comfortably >= 2x the ~7.6 cm floor, so the
    # off-PENT match stays well-posed). Anchor every offset's warm-start at the s=0 solution
    # so the scan does not drift point-to-point.
    waist_beta = max(0.15, 2.0 * beta_floor) if feasible_betas else 0.15
    print("\n" + "=" * 80)
    print(f"Waist scan (target beta = {waist_beta*100:.1f} cm, scanning s_offset about PENT)...")
    waist_records, waist_rows = [], []
    anchor, _, _, _, _ = solve_match(tao, waist_beta, 0.0, golden_vec)
    anchor_vec = np.array([anchor[q] for q in FF_QUADS])
    for s in np.sort(WAIST_OFFSETS):
        solved, ach, rel_err, feasible, pinned = solve_match(tao, waist_beta, float(s), anchor_vec)
        print(f"  s_offset={s:+.2f} m | achieved bx={ach['beta_x']*100:5.1f} "
              f"by={ach['beta_y']*100:5.1f} cm | relerr={rel_err:6.1%} | feasible={feasible}")
        waist_records.append(dict(s=float(s), solved=solved, ach=ach, feasible=feasible))
        waist_rows.append([f"{s:+.2f}"]
                          + [f"{solved[q]:.3f}" for q in FF_QUADS]
                          + [f"{ach['beta_x']:.4f}", f"{ach['beta_y']:.4f}",
                             f"{rel_err:.4f}", str(feasible)])
    write_csv(HERE / "results_waist_scan.csv",
              ["s_offset_m"] + [f"{q}_kGm" for q in FF_QUADS]
              + ["ach_beta_x_m", "ach_beta_y_m", "rel_beta_err", "feasible"],
              waist_rows)

    # ---- 3) Aggregate per-quad derived ranges (feasible beta points only) --------------
    print("\n" + "=" * 80)
    print("Derived FF quad ranges (across feasible beta targets):")
    feas = [r for r in beta_records if r["feasible"]]
    summary_rows = []
    for q in FF_QUADS:
        vals = [r["solved"][q] for r in feas]
        lo, hi = (min(vals), max(vals)) if vals else (float("nan"), float("nan"))
        elo, ehi = EPICS_BOUNDS[q]
        hits = bool(vals and (min(vals) - elo < BOUND_FRAC_TOL * (ehi - elo)
                              or ehi - max(vals) < BOUND_FRAC_TOL * (ehi - elo)))
        gold = golden_named[q]
        # excursion relative to golden, expressed as +/- kG.m
        dn, up = (gold - lo, hi - gold) if vals else (float("nan"), float("nan"))
        print(f"  {q}: golden={gold:8.3f} | derived [{lo:8.3f}, {hi:8.3f}] "
              f"(-{dn:.1f}/+{up:.1f}) | EPICS [{elo:7.1f}, {ehi:6.1f}] | hits_bound={hits}")
        summary_rows.append([q, f"{gold:.3f}", f"{lo:.3f}", f"{hi:.3f}",
                             f"{dn:.3f}", f"{up:.3f}", f"{elo:.1f}", f"{ehi:.1f}", str(hits)])
    write_csv(HERE / "summary_ranges.csv",
              ["quad", "golden_kGm", "derived_min_kGm", "derived_max_kGm",
               "minus_from_golden_kGm", "plus_from_golden_kGm",
               "epics_min_kGm", "epics_max_kGm", "derived_hits_bound"],
              summary_rows)
    print(f"  (ranges span feasible PENT beta from {beta_floor*100:.0f} cm to 50 cm)")

    # ---- 4) Figures --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating figures...")
    bt_cm = np.array([r["target"] for r in beta_records]) * 100
    feas_mask = np.array([r["feasible"] for r in beta_records])

    # (a) solved quad strength vs target beta, with golden + EPICS bounds
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, q in zip(axes.ravel(), FF_QUADS):
        y = np.array([r["solved"][q] for r in beta_records])
        ax.plot(bt_cm, y, "-", color="0.7", zorder=1)
        ax.scatter(bt_cm[feas_mask], y[feas_mask], c="C0", s=28, label="feasible", zorder=3)
        ax.scatter(bt_cm[~feas_mask], y[~feas_mask], c="C3", marker="x", s=40,
                   label="infeasible", zorder=3)
        ax.axhline(golden_named[q], color="C2", ls="--", lw=1, label="golden")
        elo, ehi = EPICS_BOUNDS[q]
        ax.axhspan(elo, ehi, color="C1", alpha=0.08)
        ax.set_title(q); ax.set_ylabel("kG.m"); ax.set_xscale("log"); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("target beta* at PENT [cm]")
    fig.suptitle("Solved FF quad strength vs target PENT beta (orange band = EPICS bounds)")
    fig.tight_layout(); fig.savefig(FIGDIR / "ff_quads_vs_beta.png", dpi=130); plt.close(fig)

    # (b) achieved beta vs target beta (feasibility / where the solver cannot reach target)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(bt_cm, bt_cm, "k--", lw=1, label="ideal (achieved = target)")
    ax.plot(bt_cm, [r["ach"]["beta_x"] * 100 for r in beta_records], "o-", label="achieved beta_x")
    ax.plot(bt_cm, [r["ach"]["beta_y"] * 100 for r in beta_records], "s-", label="achieved beta_y")
    if feasible_betas:
        ax.axvline(beta_floor * 100, color="C3", ls=":", label=f"feasible floor ~{beta_floor*100:.0f} cm")
    ax.set_xlabel("target beta* [cm]"); ax.set_ylabel("achieved beta* [cm]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Match feasibility at PENT (6 FF quads)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGDIR / "achieved_vs_target_beta.png", dpi=130); plt.close(fig)

    # (c) solved quad strength vs waist s_offset
    s_arr = np.array([r["s"] for r in waist_records])
    fig, ax = plt.subplots(figsize=(9, 6))
    for q in FF_QUADS:
        ax.plot(s_arr, [r["solved"][q] for r in waist_records], "o-", label=q)
    ax.set_xlabel(f"waist offset from PENT [m]  (target beta = {waist_beta*100:.0f} cm)")
    ax.set_ylabel("solved FF quad [kG.m]")
    ax.set_title("FF quad strength vs waist location"); ax.grid(alpha=0.3); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(FIGDIR / "ff_quads_vs_waist.png", dpi=130); plt.close(fig)

    # (d) beta function vs s near PENT for a few matched targets
    profile_targets = [b for b in BETAS_FOR_PROFILE if (not feasible_betas) or b >= beta_floor]
    if feasible_betas and beta_floor not in profile_targets:
        profile_targets = sorted(set(profile_targets + [round(beta_floor, 3)]))
    fig, ax = plt.subplots(figsize=(9, 6))
    s_prof = np.arange(-2.0, 2.0, 0.05)
    for bt in profile_targets:
        rec = min(beta_records, key=lambda r: abs(r["target"] - bt))
        set_quads(tao, [rec["solved"][q] for q in FF_QUADS])
        bx = [tao.twiss_at_s(ele="PENT", s_offset=float(s))["beta_a"] for s in s_prof]
        by = [tao.twiss_at_s(ele="PENT", s_offset=float(s))["beta_b"] for s in s_prof]
        line, = ax.plot(s_prof, bx, "-", label=f"beta_x, target {rec['target']*100:.0f} cm")
        ax.plot(s_prof, by, "--", color=line.get_color(),
                label=f"beta_y, target {rec['target']*100:.0f} cm")
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("s - s(PENT) [m]"); ax.set_ylabel("beta [m]")
    ax.set_title("Beta function near PENT for matched solutions"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGDIR / "beta_profile_near_pent.png", dpi=130); plt.close(fig)
    print(f"  wrote 4 figures to {FIGDIR}")

    # ---- restore golden and finish -----------------------------------------------------
    set_quads(tao, golden_vec)
    print("\nDone. Lattice restored to golden FF settings.")


if __name__ == "__main__":
    main()
