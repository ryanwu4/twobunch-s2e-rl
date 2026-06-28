"""Lightweight datagen tests — no Tao/FACET2_S2E needed.

Covers the authoritative knob tables (original 8-D + expanded 26-D) and the LHS manifest's
determinism + in-bounds guarantee (the resume-stability contract: same seed -> identical
indices/knobs), plus the back-compat default (no sweep_set -> original8).
"""
import numpy as np
import pytest

from twobunch_s2e_rl.datagen.sweep_params import (
    SWEEP_PARAMS, PARAM_KEYS, BOUNDS_LOW, BOUNDS_HIGH, BASELINE_KNOBS,
    SWEEP_PARAMS_EXPANDED_EXTRA, EXPANDED_PARAMS, EXPANDED_ANCHORED_PARAMS,
    SWEEP_SETS, resolve_sweep_set,
)
from twobunch_s2e_rl.datagen.ff_manifold import (
    FF_KEYS, FF_MATCHED_CURVE, MANIFOLD_SPECS, sample_anchored_ff,
)
from twobunch_s2e_rl.datagen.run_sweep import build_manifest


# --------------------------------------------------------------------------------------
# Original 8-D table
# --------------------------------------------------------------------------------------
def test_param_table_shape():
    assert len(SWEEP_PARAMS) == 8
    assert PARAM_KEYS == list(SWEEP_PARAMS)
    assert len(BOUNDS_LOW) == len(BOUNDS_HIGH) == 8


def test_bounds_ordering_and_baseline_within():
    for k in PARAM_KEYS:
        lo, hi, base = SWEEP_PARAMS[k]
        assert lo < hi, f"{k}: low !< high"
        assert lo <= base <= hi, f"{k}: baseline {base} outside [{lo}, {hi}]"


def test_derived_arrays_consistent():
    assert BOUNDS_LOW == [SWEEP_PARAMS[k][0] for k in PARAM_KEYS]
    assert BOUNDS_HIGH == [SWEEP_PARAMS[k][1] for k in PARAM_KEYS]
    assert BASELINE_KNOBS == {k: SWEEP_PARAMS[k][2] for k in PARAM_KEYS}


# --------------------------------------------------------------------------------------
# Expanded 26-D table + sweep-set resolver
# --------------------------------------------------------------------------------------
def test_expanded_table_shape():
    assert len(SWEEP_PARAMS_EXPANDED_EXTRA) == 18           # 6 FF quads + 4 kickers + 8 movers
    assert len(EXPANDED_PARAMS) == 26
    # expanded is a strict superset of the original 8, original keys first (order preserved)
    assert list(EXPANDED_PARAMS)[:8] == PARAM_KEYS
    assert set(EXPANDED_PARAMS) == set(PARAM_KEYS) | set(SWEEP_PARAMS_EXPANDED_EXTRA)


def test_expanded_bounds_ordering_and_baseline_within():
    for k, (lo, hi, base) in EXPANDED_PARAMS.items():
        assert lo < hi, f"{k}: low !< high"
        assert lo <= base <= hi, f"{k}: baseline {base} outside [{lo}, {hi}]"


def test_resolve_sweep_set():
    keys, low, high, baseline = resolve_sweep_set("original8")
    assert keys == PARAM_KEYS and len(low) == len(high) == 8

    keys, low, high, baseline = resolve_sweep_set("expanded")
    assert len(keys) == len(low) == len(high) == len(baseline) == 26
    assert keys == list(EXPANDED_PARAMS)
    assert low == [EXPANDED_PARAMS[k][0] for k in keys]
    assert baseline == {k: EXPANDED_PARAMS[k][2] for k in keys}

    assert resolve_sweep_set() == resolve_sweep_set("original8")  # default
    assert set(SWEEP_SETS) == {"original8", "expanded", "expanded_anchored"}
    with pytest.raises(KeyError):
        resolve_sweep_set("nope")


# --------------------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------------------
def _cfg(out_dir, n=64, seed=20260612, repeats=0, sweep_set=None):
    cfg = {"output_dir": str(out_dir), "n_samples": n, "seed": seed,
           "n_baseline_repeats": repeats}
    if sweep_set is not None:
        cfg["sweep_set"] = sweep_set
    return cfg


def test_manifest_deterministic(tmp_path):
    a = build_manifest(_cfg(tmp_path / "a"))
    b = build_manifest(_cfg(tmp_path / "b"))
    assert len(a) == len(b) == 64
    for ra, rb in zip(a, b):
        assert ra["idx"] == rb["idx"]
        assert ra["knobs"] == rb["knobs"]  # exact: same seed -> same LHS draw


def test_manifest_defaults_to_original8(tmp_path):
    # No sweep_set key -> original 8-D draw (back-compat with the existing campaign).
    man = build_manifest(_cfg(tmp_path / "compat", n=8))
    for row in man:
        assert list(row["knobs"]) == PARAM_KEYS


def test_manifest_within_bounds(tmp_path):
    man = build_manifest(_cfg(tmp_path / "c", n=128))
    for row in man:
        for k in PARAM_KEYS:
            lo, hi, _ = SWEEP_PARAMS[k]
            assert lo <= row["knobs"][k] <= hi, f"idx {row['idx']} knob {k} out of bounds"


def test_manifest_baseline_repeats(tmp_path):
    man = build_manifest(_cfg(tmp_path / "d", n=16, repeats=3))
    assert len(man) == 19
    reps = [r for r in man if r["is_baseline_repeat"]]
    assert len(reps) == 3
    for r in reps:
        assert r["knobs"] == BASELINE_KNOBS


def test_manifest_expanded(tmp_path):
    man = build_manifest(_cfg(tmp_path / "e", n=64, repeats=2, sweep_set="expanded"))
    assert len(man) == 66
    lhs = [r for r in man if not r["is_baseline_repeat"]]
    for row in lhs:
        assert list(row["knobs"]) == list(EXPANDED_PARAMS)   # 26 knobs, expanded order
        for k, (lo, hi, _) in EXPANDED_PARAMS.items():
            assert lo <= row["knobs"][k] <= hi, f"idx {row['idx']} knob {k} out of bounds"
    # baseline-repeats reproduce the golden (transversely-tuned) working point
    for r in man:
        if r["is_baseline_repeat"]:
            assert r["knobs"] == {k: EXPANDED_PARAMS[k][2] for k in EXPANDED_PARAMS}


# --------------------------------------------------------------------------------------
# Manifold-anchored set + FF sampler
# --------------------------------------------------------------------------------------
def test_anchored_set_shape_and_narrowing():
    assert set(SWEEP_SETS) >= {"expanded_anchored"}
    assert len(EXPANDED_ANCHORED_PARAMS) == 26
    assert list(EXPANDED_ANCHORED_PARAMS) == list(EXPANDED_PARAMS)   # same keys/order
    for k, (lo, hi, base) in EXPANDED_ANCHORED_PARAMS.items():
        assert lo < hi and lo <= base <= hi, f"{k} bounds/baseline"
    # transverse knobs (movers/kickers) are strictly narrower than `expanded`; baselines equal
    for k in ("S1EL_xOffset", "S2ER_yOffset", "XC1FFkG", "YC1FFkG"):
        a_lo, a_hi, a_b = EXPANDED_ANCHORED_PARAMS[k]
        e_lo, e_hi, e_b = EXPANDED_PARAMS[k]
        assert (a_hi - a_lo) < (e_hi - e_lo), f"{k} not narrowed"
        assert a_b == e_b, f"{k} baseline changed"
    # longitudinal knobs keep the expanded envelope
    for k in PARAM_KEYS:
        assert EXPANDED_ANCHORED_PARAMS[k] == EXPANDED_PARAMS[k]
    # FF quads: declared range widened to EPICS (>= expanded envelope), baseline = golden
    for k in FF_KEYS:
        a_lo, a_hi, a_b = EXPANDED_ANCHORED_PARAMS[k]
        e_lo, e_hi, e_b = EXPANDED_PARAMS[k]
        assert a_lo <= e_lo and a_hi >= e_hi and a_b == e_b, f"{k} FF envelope"


def test_sample_anchored_ff():
    spec = MANIFOLD_SPECS["expanded_anchored"]
    # jitter=0 recovers the matched curve exactly (np.interp at the drawn beta)
    rng = np.random.default_rng(7)
    beta, ff = sample_anchored_ff(rng, 200, jitter_frac=0.0)
    assert ff.shape == (200, 6)
    assert (beta >= spec["beta_lo"]).all() and (beta <= spec["beta_hi"]).all()
    b_asc, q_asc = FF_MATCHED_CURVE[::-1, 0], FF_MATCHED_CURVE[::-1, 1:]
    expect = np.column_stack([np.interp(beta, b_asc, q_asc[:, j]) for j in range(6)])
    assert np.allclose(ff, expect, atol=1e-9)
    # determinism (same seed -> same draw) and jitter actually perturbs off the curve
    f1 = sample_anchored_ff(np.random.default_rng(1), 50, 0.06)[1]
    f2 = sample_anchored_ff(np.random.default_rng(1), 50, 0.06)[1]
    assert np.array_equal(f1, f2)
    f3 = sample_anchored_ff(np.random.default_rng(2), 50, 0.06)[1]
    assert not np.allclose(f1, f3)


def test_manifest_anchored_stratified(tmp_path):
    n = 100
    man = build_manifest(_cfg(tmp_path / "anc", n=n, repeats=2, sweep_set="expanded_anchored"))
    lhs = [r for r in man if not r["is_baseline_repeat"]]
    assert len(lhs) == n
    anchor = [r for r in lhs if r["block"] == "anchor"]
    tail = [r for r in lhs if r["block"] == "tail"]
    assert len(tail) == round(0.30 * n) and len(anchor) == n - len(tail)

    spec = MANIFOLD_SPECS["expanded_anchored"]
    # anchor rows: FF on the curve neighbourhood, transverse knobs in the NARROW ranges,
    # and a recorded target beta
    for r in anchor:
        assert spec["beta_lo"] <= r["ff_target_beta_m"] <= spec["beta_hi"]
        for k, (lo, hi, _) in EXPANDED_ANCHORED_PARAMS.items():
            assert lo - 1e-9 <= r["knobs"][k] <= hi + 1e-9, f"anchor {k} out of narrow range"
    # tail rows: drawn from the WIDE (expanded) ranges; no target beta
    for r in tail:
        assert r["ff_target_beta_m"] is None
        for k, (lo, hi, _) in EXPANDED_PARAMS.items():
            assert lo - 1e-9 <= r["knobs"][k] <= hi + 1e-9, f"tail {k} out of wide range"
    # the tail genuinely explores wider movers than the anchor block
    assert max(r["knobs"]["S2ER_yOffset"] for r in tail) > \
        max(r["knobs"]["S2ER_yOffset"] for r in anchor)

    # deterministic
    man2 = build_manifest(_cfg(tmp_path / "anc2", n=n, sweep_set="expanded_anchored"))
    a2 = [r for r in man2 if not r["is_baseline_repeat"]]
    for r, s in zip(lhs, a2):
        assert r["knobs"] == s["knobs"] and r["block"] == s["block"]
