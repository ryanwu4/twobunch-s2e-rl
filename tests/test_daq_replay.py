"""DAQ replay plant: L2 -> multiknob position, per-visit bootstrap of the step's own images, interpolation, validity."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
from twobunch_s2e_rl.rl import daq_replay as dr  # noqa: E402


def _scan(tmp_path, monkeypatch):
    steps = {}
    for i, v in enumerate(np.arange(0, 2.01, 0.25)):
        sp = 200.0 - 50.0 * v                                   # um
        steps[str(i + 1)] = {"scan_val": float(v), "offsets_plus_um": [sp + 1, sp - 1, sp],
                             "offsets_minus_um": [-sp, -sp + 2, -sp - 2], "drive_sigma_z_um": 40.0 + v,
                             "drive_charge_fraction": 0.7, "valid": float(v <= 1.5)}
    (tmp_path / "S.json").write_text(json.dumps({"steps": steps}))
    monkeypatch.setattr(dr, "REPLAY", str(tmp_path))
    return dr.ScanPlant("S", shots_per_step=30)


def test_scan_plant_maps_l2_to_the_scan_and_interpolates(tmp_path, monkeypatch):
    p = _scan(tmp_path, monkeypatch)
    assert p.valid_window() == (0.0, 1.5)
    sp, sz, fq, ok, x = p.read(dr.L2_REF + 0.6)                 # between steps 0.5 and 0.75
    assert x == pytest.approx(0.6) and ok == 1.0
    assert sp == pytest.approx((200 - 50 * 0.6) * 1e-6, abs=1.5e-6)
    assert sz == pytest.approx(40.5e-6)                         # nearest step (0.5)
    assert p.read(dr.L2_REF + 1.9)[3] == 0.0                    # invalid step
    assert p.read(dr.L2_REF + 3.0)[3] == 0.0                    # off the scan: invalid
