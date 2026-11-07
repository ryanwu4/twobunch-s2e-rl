"""Calibration campaigns for conformal prediction (conformal/README.md): the iid design and the guard that keeps
them out of training tables."""
import json

import numpy as np
import pytest

from twobunch_s2e_rl.datagen.injector_campaign import injector_points, make_rows
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
from twobunch_s2e_rl.surrogate.preprocess import preprocess

CFG = {"linac_sweep_set": "xtcav23e_l0b2", "injector_sweep_set": "inj12",
       "design": {"seed": 7, "calibration": {"n": 200, "sampler": "iid", "stratum": "uniform"}}}


def test_calibration_rows_are_independent_draws_inside_the_box():
    rows = make_rows(CFG)
    keys, low, high, _ = resolve_sweep_set("xtcav23e_l0b2_inj12")
    x = np.array([[{**r["knobs"], **r["injector"]}[k] for k in keys] for r in rows])
    assert len(rows) == 200 and {r["block"] for r in rows} == {"calibration"}
    assert np.all(x >= low) and np.all(x <= high)
    assert len(injector_points(rows)) == 200                     # one Impact-T beam per row
    assert len({r["linac_id"] for r in rows}) == 200             # one Tao seed per row
    assert make_rows(CFG) == rows                                # deterministic by seed


def test_calibration_design_takes_no_other_blocks():
    cfg = {**CFG, "design": {**CFG["design"], "pair_golden_frac": 0.25}}
    with pytest.raises(ValueError):
        make_rows(cfg)


def _campaign(root, name, purpose):
    d = root / name
    d.mkdir()
    (d / "manifest_meta.json").write_text(json.dumps({"config": {"purpose": purpose} if purpose else {}}))
    (d / "sample_00000.json").write_text(json.dumps({"idx": 0, **({"purpose": purpose} if purpose else {})}))


def test_preprocess_refuses_calibration_campaigns_in_training_tables(tmp_path):
    _campaign(tmp_path, "cal", "calibration")
    _campaign(tmp_path, "train", None)
    with pytest.raises(SystemExit, match="never enters a training table"):
        preprocess(subdir="train,cal", data_root=tmp_path, sweep_set="xtcav23e_l0b2_inj12", verbose=False)
    with pytest.raises(SystemExit, match="not a calibration campaign"):
        preprocess(subdir="cal,train", data_root=tmp_path, sweep_set="xtcav23e_l0b2_inj12", verbose=False,
                   calibration=True)
