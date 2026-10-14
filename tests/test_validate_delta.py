"""validate_delta.compare on synthetic export/track files."""
import json

import twobunch_s2e_rl.rl.validate_delta as V


def test_compare_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_out", lambda logdir, tag="": tmp_path)
    rows = []
    for i in range(3):
        rows.append({"i": i, "knobs": {}, "goals": {"bunch_spacing": {"measured": 2e-4, "cloud": 2e-4}},
                     "surrogate": {"bunch_spacing": 2.1e-4, "drive_sigma_z": 3e-5}})
        truth = {"bunch_spacing": 2.0e-4 + i * 1e-5, "drive_sigma_z": 3.3e-5, "witness_lost": 0.0}
        (tmp_path / f"bmad_{i:03d}.json").write_text(json.dumps({"ok": True, "truth": truth}))
    (tmp_path / "bmad_002.json").write_text(json.dumps({"ok": False, "error": "x"}))   # failed track
    (tmp_path / "setpoints.json").write_text(json.dumps(rows))
    s = V.compare("logs/whatever")
    assert s["n_tracked_ok"] == 2 and s["n_setpoints"] == 3
    sp = s["targets"]["bunch_spacing"]
    assert abs(sp["bmad_vs_goal_abs_median"] - 0.5e-5) < 1e-12
    assert abs(s["targets"]["drive_sigma_z"]["surrogate_vs_bmad_rel_median"] - 0.3e-5 / 3.3e-5) < 1e-9
    assert "bmad_vs_goal_abs_median" not in s["targets"]["drive_sigma_z"]
