"""Shared data-loading + derivation helpers for campaign analysis.

Extracted from the former analysis/{build_dataset,achievable_targets} so the report scripts
(now colocated in results/<study>/) and the reusable tools (analysis_tools/) can both import
these without depending on each other. Pure data plumbing — no plotting.
"""
import glob
import json
import warnings

import numpy as np
import pandas as pd

from .datagen.paths import repo_root
from .datagen.sweep_params import SWEEP_PARAMS_EXPANDED_EXTRA

POINTS = ["BEGBC20", "MFFF", "PENT"]
FF_QUADS = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]


def flatten_sample(d):
    """One per-sample JSON -> a flat row: meta + knobs + {POINT}__{metric} (lists -> _max/_mean)."""
    row = {}
    for k in ("idx", "success", "is_baseline_repeat", "num_macro_particles",
              "csrTF", "transverseWakes", "wall_s"):
        row[k] = d.get(k)
    for k, v in d["knobs"].items():
        row[k] = v
    for pt in POINTS:
        spec = d["specs"].get(pt, {})
        for mk, mv in spec.items():
            if isinstance(mv, list):
                arr = np.asarray(mv, dtype=float)
                with warnings.catch_warnings():  # all-NaN slice is expected (absent bunch)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row[f"{pt}__{mk}_max"] = np.nanmax(arr) if arr.size else np.nan
                    row[f"{pt}__{mk}_mean"] = np.nanmean(arr) if arr.size else np.nan
            else:
                row[f"{pt}__{mk}"] = mv
    return row


def load(subdir):
    """Load data/<subdir>/sample_*.json into a flat DataFrame sorted by idx."""
    files = sorted(glob.glob(str(repo_root() / "data" / subdir / "sample_*.json")))
    if not files:
        raise SystemExit(f"no sample_*.json under data/{subdir}")
    df = pd.DataFrame(flatten_sample(json.load(open(f))) for f in files)
    return df.sort_values("idx").reset_index(drop=True)


def P(df, metric):
    """PENT column as float array (NaN where absent, e.g. witness not resolved)."""
    col = f"PENT__{metric}"
    return df[col].to_numpy(dtype=float) if col in df else np.full(len(df), np.nan)


def derived(df):
    """Display-unit target quantities at PENT (per row; NaN where undefined)."""
    ang = np.sqrt((P(df, "PDrive_median_xp") - P(df, "PWitness_median_xp"))**2 +
                  (P(df, "PDrive_median_yp") - P(df, "PWitness_median_yp"))**2) * 1e6  # urad
    bmag_max = np.nanmax(np.vstack([P(df, f"P{b}_BMAG_{a}")
                                    for b in ("Drive", "Witness") for a in ("x", "y")]), axis=0)
    return {
        "transmission":      P(df, "transmission_total"),                       # fraction
        "spacing":           P(df, "bunchSpacing") * 1e6,                        # um (signed)
        "offset":            P(df, "transverseCentroidOffset") * 1e6,           # um
        "angle":             ang,                                                # urad
        "dE":               (P(df, "PDrive_median_energy") - P(df, "PWitness_median_energy")) * 1e-6,  # MeV
        "drive_emit_x":      P(df, "PDrive_norm_emit_x") * 1e6,                  # um-rad
        "drive_emit_y":      P(df, "PDrive_norm_emit_y") * 1e6,
        "witness_emit_x":    P(df, "PWitness_norm_emit_x") * 1e6,
        "witness_emit_y":    P(df, "PWitness_norm_emit_y") * 1e6,
        "drive_BMAG_x":      P(df, "PDrive_BMAG_x"),
        "drive_BMAG_y":      P(df, "PDrive_BMAG_y"),
        "witness_BMAG_x":    P(df, "PWitness_BMAG_x"),
        "witness_BMAG_y":    P(df, "PWitness_BMAG_y"),
        "BMAG_max":          bmag_max,                                           # worst of the 4
        "drive_sigz":        P(df, "PDrive_sigmaSI90_z") * 1e6,                  # um
        "witness_sigz":      P(df, "PWitness_sigmaSI90_z") * 1e6,
    }


def ff_offmanifold(df):
    """Per-row L2 distance of the 6 FF quads from golden, in units of each half-range."""
    z = np.zeros(len(df))
    for q in FF_QUADS:
        lo, hi, base = SWEEP_PARAMS_EXPANDED_EXTRA[q]
        half = (hi - lo) / 2.0
        z = z + ((df[q].to_numpy(dtype=float) - base) / half) ** 2
    return np.sqrt(z)
