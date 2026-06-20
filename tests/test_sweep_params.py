"""Lightweight datagen tests — no Tao/FACET2_S2E needed.

Covers the authoritative knob tables (original 8-D + expanded 26-D) and the LHS manifest's
determinism + in-bounds guarantee (the resume-stability contract: same seed -> identical
indices/knobs), plus the back-compat default (no sweep_set -> original8).
"""
import numpy as np
import pytest

from twobunch_s2e_rl.datagen.sweep_params import (
    SWEEP_PARAMS, PARAM_KEYS, BOUNDS_LOW, BOUNDS_HIGH, BASELINE_KNOBS,
    SWEEP_PARAMS_EXPANDED_EXTRA, EXPANDED_PARAMS, SWEEP_SETS, resolve_sweep_set,
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
    assert set(SWEEP_SETS) == {"original8", "expanded"}
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
