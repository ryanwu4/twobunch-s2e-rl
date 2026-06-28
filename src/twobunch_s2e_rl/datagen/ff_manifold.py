"""Manifold-anchored FF-quad sampler for the expanded two-bunch sweep.

The deliverable region (matched + collinear + surviving) is a thin manifold; independent
box-LHS misses it (the 512-sample pilot got 0/512 near-matched). This module samples the 6
FF quads ON the matched beta*->FF-quad curve from the 2026-06-19 PENT beta-matching solve
plus bounded jitter, so the campaign concentrates where the beam is near-matched, with a
controlled mismatch spread. build_manifest consumes the per-set MANIFOLD_SPECS entry.

Rationale + figures: lab-notebook/claude/analyses/2026-06-21_manifold-anchoring-strategy.md
"""
import numpy as np

FF_KEYS = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]

# golden FF (kG.m) and EPICS clip bounds (mirror sweep_params / setLattice bounds.yml)
_GOLDEN_FF = np.array([-71.837, -81.251, 99.225, 126.350, -235.218, 126.353])
_EPICS_FF = np.array([(-256.0, 0.0), (-446.0, 0.0), (0.0, 457.0),
                      (0.0, 167.0), (-257.0, 0.0), (0.0, 167.0)])

# Matched beta*->FF-quad curve: the feasible rows of
# FACET2-S2E/analysis/2026-06-19_ff-quad-range/results_beta_sweep.csv
# columns: beta_m, Q5FF, Q4FF, Q3FF, Q2FF, Q1FF, Q0FF  (kG.m)
FF_MATCHED_CURVE = np.array([
    [0.5000, -72.043, -81.473, 99.353, 126.438, -234.626, 126.394],
    [0.4372, -72.548, -84.684, 98.248, 128.526, -235.795, 128.541],
    [0.3824, -72.617, -88.209, 96.284, 131.302, -236.795, 129.195],
    [0.3344, -73.238, -91.585, 94.812, 133.249, -237.694, 130.892],
    [0.2924, -74.075, -94.885, 93.177, 135.074, -238.483, 132.472],
    [0.2557, -76.494, -95.835, 90.898, 136.372, -239.123, 134.560],
    [0.2236, -78.094, -98.558, 88.777, 138.022, -239.714, 135.803],
    [0.1955, -79.715, -102.185, 87.256, 139.390, -240.266, 137.271],
    [0.1710, -82.150, -105.568, 86.237, 140.262, -240.761, 139.312],
    [0.1495, -84.528, -109.354, 84.770, 141.457, -241.165, 140.485],
    [0.1308, -87.490, -113.457, 83.789, 142.471, -241.442, 141.461],
    [0.1144, -90.856, -117.965, 83.020, 143.416, -241.677, 142.339],
    [0.1000, -94.770, -122.902, 82.551, 144.313, -241.826, 142.947],
    [0.0874, -98.923, -128.129, 81.850, 145.250, -242.014, 143.539],
    [0.0765, -103.287, -133.581, 81.146, 146.161, -242.185, 144.003],
])
_BETA_ASC = FF_MATCHED_CURVE[::-1, 0]          # ascending beta for np.interp
_QUAD_ASC = FF_MATCHED_CURVE[::-1, 1:]
_EXCURSION = np.abs(_GOLDEN_FF - FF_MATCHED_CURVE[-1, 1:])  # golden->floor span per quad

BETA_FLOOR_M = float(FF_MATCHED_CURVE[-1, 0])  # 0.0765 m -> ~7.6 cm feasibility floor
BETA_TOP_M = float(FF_MATCHED_CURVE[0, 0])     # 0.50 m  (golden)


def sample_anchored_ff(rng, n, jitter_frac=0.06, beta_lo=BETA_FLOOR_M, beta_hi=BETA_TOP_M):
    """Anchored FF-quad draws. Returns (beta (n,), ff (n,6)).

    beta* log-uniform in [beta_lo, beta_hi]; FF = interp(matched curve at beta*) + Gaussian
    jitter (sigma = jitter_frac * |golden-floor| per quad); clipped to EPICS bounds. `rng` is
    a numpy Generator (caller seeds it for determinism).
    """
    beta = np.exp(rng.uniform(np.log(beta_lo), np.log(beta_hi), n))
    ff = np.column_stack([np.interp(beta, _BETA_ASC, _QUAD_ASC[:, j]) for j in range(6)])
    ff = ff + rng.normal(0.0, 1.0, ff.shape) * (jitter_frac * _EXCURSION)
    ff = np.clip(ff, _EPICS_FF[:, 0], _EPICS_FF[:, 1])
    return beta, ff


# Per-sweep-set anchoring spec consumed by build_manifest. `wide_set` supplies the ranges for
# the stratification tail (full-width transverse box, for the feasibility head + recovery).
MANIFOLD_SPECS = {
    "expanded_anchored": {
        "ff_keys": FF_KEYS,
        "jitter_frac": 0.06,
        "beta_lo": BETA_FLOOR_M,
        "beta_hi": BETA_TOP_M,
        "stratify_tail_frac": 0.30,
        "wide_set": "expanded",
    },
}
