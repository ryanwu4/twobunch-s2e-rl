"""Validate controller setpoints in Bmad: track each exported setpoint through the real FACET2-S2E
lattice and compare the tracked PENT metrics to the surrogate's predictions.

Reuses the datagen tracking path (setLattice(baseline + knobs) -> trackBeam -> getBeamSpecs at PENT),
which already handles all 26 knobs -- so NO 8-knob bmad_env is involved. The setpoint JSONs from
scripts/transfer_setpoints.py hold ABSOLUTE physical knob values keyed by the setLattice param names,
so they override the baseline directly. This is the honest surrogate-vs-truth open-loop check.

Runs in the FACET2-S2E env (Tao/PyTao): use `conda activate bmad-qpad-dev`.

Usage:
  conda activate bmad-qpad-dev && cd twobunch-s2e-rl
  PYTHONPATH=$PWD/src python scripts/validate_bmad.py \
    --setpoints-dir results/rl/bptt_gc_combined/setpoints \
    --norm-json processed/twobunch_combined_norm.json \
    --num-macro-particles 100000            # faithful (matches training); 20000 for a fast look
Outputs: results/rl/bptt_gc_combined/bmad_validation/validate_goal<g>um.json (surrogate vs Bmad, per metric),
         validation_summary.csv, and the tracked PENT beam sample_<g>_PENT.h5 (for later corner plots).
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from twobunch_s2e_rl.datagen import run_sweep as rs
from twobunch_s2e_rl.datagen.paths import repo_root


def _bmad_metrics(pent: dict, drive_full: float, witness_full: float) -> dict:
    """Map PENT getBeamSpecs -> the same metric names the surrogate predicts (display units)."""
    def g(k):
        v = pent.get(k)
        return float(v) if v is not None else float("nan")
    ang = np.hypot(g("PDrive_median_xp") - g("PWitness_median_xp"),
                   g("PDrive_median_yp") - g("PWitness_median_yp")) * 1e6
    clip = lambda c, full: float(np.clip(c / full, 0.0, 1.05)) if np.isfinite(c) else float("nan")
    return {
        "spacing_um": abs(g("bunchSpacing")) * 1e6,
        "T_drive": clip(g("PDrive_charge_nC"), drive_full),
        "T_witness": clip(g("PWitness_charge_nC"), witness_full),
        "drive_emit_x_um_rad": g("PDrive_norm_emit_x") * 1e6,
        "drive_emit_y_um_rad": g("PDrive_norm_emit_y") * 1e6,
        "witness_emit_x_um_rad": g("PWitness_norm_emit_x") * 1e6,
        "witness_emit_y_um_rad": g("PWitness_norm_emit_y") * 1e6,
        "transverse_offset_um": g("transverseCentroidOffset") * 1e6,
        "angular_misalignment_urad": ang,
        "transmission_total": g("transmission_total"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setpoints-dir", default="results/rl/bptt_gc_combined/setpoints")
    ap.add_argument("--norm-json", default="processed/twobunch_combined_norm.json",
                    help="for drive/witness full charge (T_drive/T_witness normalization)")
    ap.add_argument("--baseline-config", default="setLattice_configs/2024-10-14_twoBunch_baseline.yml")
    ap.add_argument("--num-macro-particles", type=int, default=100000)
    ap.add_argument("--no-csr", action="store_true", help="disable CSR (default: on, matches training)")
    ap.add_argument("--no-wakes", action="store_true", help="disable transverse wakes (default: on)")
    ap.add_argument("--out", default="results/rl/bptt_gc_combined/bmad_validation")
    args = ap.parse_args()

    with open(repo_root() / args.norm_json) as f:
        norm = json.load(f)
    drive_full = float(norm["drive_full_charge_nC"])
    witness_full = float(norm["witness_full_charge_nC"])

    files = sorted(glob.glob(str(repo_root() / args.setpoints_dir / "setpoints_goal*um.json")))
    if not files:
        raise SystemExit(f"no setpoints_goal*um.json under {args.setpoints_dir} "
                         f"(run scripts/transfer_setpoints.py first)")

    outdir = repo_root() / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "baseline_config": args.baseline_config,
        "csrTF": not args.no_csr,
        "transverseWakes": not args.no_wakes,
        "num_macro_particles": int(args.num_macro_particles),
        "output_dir": str(outdir),
        "save_beams_at": ["PENT"],
    }
    (outdir / "scratch").mkdir(exist_ok=True)
    rs._init_worker(cfg)                       # sets up Tao + baseline in this process
    import FACET2_S2E as qs                    # noqa: N806 (matches datagen convention)

    summary = []
    for fp in files:
        rec = json.load(open(fp))
        g_um = rec["target_um"]
        knobs = rec["knob_setpoints_physical"]
        print(f"\n=== tracking setpoint for goal {g_um:.0f} um "
              f"({args.num_macro_particles} particles, csr={cfg['csrTF']}, wakes={cfg['transverseWakes']}) ===")
        item = {"idx": int(round(g_um)), "knobs": knobs}
        tracked = rs._track_and_collect(qs, item)
        pent = tracked["specs"].get("PENT", {})
        bmad = _bmad_metrics(pent, drive_full, witness_full)
        surr = rec["surrogate_metrics"]

        cmp = {}
        for k in surr:
            s, b = float(surr[k]), bmad.get(k, float("nan"))
            cmp[k] = {"surrogate": s, "bmad": b, "abs_diff": b - s,
                      "pct_diff": (100.0 * (b - s) / s) if s not in (0.0,) and np.isfinite(b) else None}
        out = {"target_um": g_um, "wall_s": tracked.get("wall_s"),
               "n_particles": args.num_macro_particles, "csrTF": cfg["csrTF"],
               "transverseWakes": cfg["transverseWakes"],
               "transmission_total_bmad": bmad["transmission_total"],
               "comparison": cmp, "bmad_metrics": bmad, "surrogate_metrics": surr,
               "setpoint_source": str(fp)}
        with open(outdir / f"validate_goal{int(round(g_um))}um.json", "w") as f:
            json.dump(out, f, indent=2)
        summary.append(out)

        print(f"  {'metric':26s} {'surrogate':>12} {'bmad':>12} {'%diff':>8}")
        for k, c in cmp.items():
            pd = f"{c['pct_diff']:+.0f}%" if c["pct_diff"] is not None else "  n/a"
            print(f"  {k:26s} {c['surrogate']:12.4g} {c['bmad']:12.4g} {pd:>8}")
        print(f"  transmission_total (bmad): {bmad['transmission_total']:.3f}  | wall {tracked.get('wall_s',0):.0f}s")

    with open(outdir / "validation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_um", "metric", "surrogate", "bmad", "abs_diff", "pct_diff"])
        for out in summary:
            for k, c in out["comparison"].items():
                w.writerow([out["target_um"], k, c["surrogate"], c["bmad"], c["abs_diff"], c["pct_diff"]])
    print(f"\nwrote {len(summary)} validation JSONs + validation_summary.csv + PENT beams to {outdir}")


if __name__ == "__main__":
    main()
