"""Lightweight datagen tests — no Tao/FACET2_S2E needed.

Covers the authoritative knob table and the LHS manifest's determinism + in-bounds
guarantee (the resume-stability contract: same seed -> identical indices/knobs).
"""
import numpy as np

from twobunch_s2e_rl.datagen.sweep_params import (
    SWEEP_PARAMS, PARAM_KEYS, BOUNDS_LOW, BOUNDS_HIGH, BASELINE_KNOBS,
)
from twobunch_s2e_rl.datagen.run_sweep import build_manifest


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


def _cfg(out_dir, n=64, seed=20260612, repeats=0):
    return {"output_dir": str(out_dir), "n_samples": n, "seed": seed,
            "n_baseline_repeats": repeats}


def test_manifest_deterministic(tmp_path):
    a = build_manifest(_cfg(tmp_path / "a"))
    b = build_manifest(_cfg(tmp_path / "b"))
    assert len(a) == len(b) == 64
    for ra, rb in zip(a, b):
        assert ra["idx"] == rb["idx"]
        assert ra["knobs"] == rb["knobs"]  # exact: same seed -> same LHS draw


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
