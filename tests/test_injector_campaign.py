"""Injector campaign: inj12 / xtcav23_inj12 knob sets, the nested/crossed manifest, absolute arrival,
campaign.json, and how injector campaigns embed / merge with golden-beam campaigns in preprocess.
Pure (no Impact-T / Bmad): synthetic sample jsons, feasibility-only rows."""
import json

import numpy as np
import pytest

from twobunch_s2e_rl.datagen import injector_campaign as ic
from twobunch_s2e_rl.datagen.sweep_params import (INJECTOR_ACTIVE_KEYS, INJECTOR_KEYS, INJECTOR_PARAMS,
                                                  SWEEP_SETS, XTCAV23_PARAMS, injector_golden,
                                                  resolve_sweep_set)
from twobunch_s2e_rl.sim import injector as inj

GOLDEN_SHA = "goldsha"


# ---------------------------------------------------------------- knob sets

def test_inj12_golden_is_the_golden_injector():
    assert inj.resolve_settings(injector_golden()) == inj.resolve_settings({})
    for k, (lo, hi, g) in INJECTOR_PARAMS.items():
        assert lo < g < hi, k
        assert g == pytest.approx(inj.golden_value(k))
    assert len(INJECTOR_KEYS) == 12 and INJECTOR_ACTIVE_KEYS[0] == "laser_timing_ps"


def test_concat_set_is_linac_then_injector_and_sweep_sets_unchanged():
    keys, lo, hi, base = resolve_sweep_set("xtcav23_inj12")
    assert keys == list(XTCAV23_PARAMS) + INJECTOR_KEYS and len(lo) == len(hi) == 35
    assert base["laser_timing_ps"] == 0.0 and base["L1PhaseSet"] == XTCAV23_PARAMS["L1PhaseSet"][2]
    assert "xtcav23_inj12" not in SWEEP_SETS and "inj12" not in SWEEP_SETS
    with pytest.raises(KeyError):
        resolve_sweep_set("inj13")


# ---------------------------------------------------------------- manifest

def _cfg(tmp_path, **design):
    d = {"seed": 7, "injector": {"golden": True, "oat": [{"laser_timing_ps": {"abs": 0.2}},
                                                         {"drive_fraction": {"abs": 0.02}}],
                                  "lhs": 5},
         "linac": {"shared": {"baseline": True, "lhs": 2}, "per_injector_lhs": 3},
         "pair_golden_frac": 0.2, "bridge": True}
    d.update(design)
    return {"name": "t", "output_dir": str(tmp_path / "camp"), "sweep_set": "xtcav23_inj12",
            "linac_sweep_set": "xtcav23", "injector_sweep_set": "inj12", "n_workers": 2,
            "injector": {"n_particles": 1000, "grid": 64, "arrival": "absolute"}, "design": d,
            "csrTF": True, "transverseWakes": True,
            "baseline_config": "setLattice_configs/2024-10-14_twoBunch_baseline.yml",
            "fixed_overrides": {"trackEnd": "PENT"}}


def test_manifest_blocks_pairing_and_bounds(tmp_path):
    rows = ic.make_rows(_cfg(tmp_path))
    blocks = [r["block"] for r in rows]
    n_inj = 1 + 2 + 5
    assert blocks.count("crossed") == n_inj * 3
    assert blocks.count("nested") == 5 * 3
    assert blocks.count("paired_golden") == 3          # round(0.2 * 15)
    assert blocks.count("bridge") == 3
    assert [r["idx"] for r in rows] == list(range(len(rows)))
    golden = injector_golden()
    # rows sharing a linac_id have identical linac knobs; each crossed linac point has every
    # injector point, incl. golden (exact paired differences)
    by_lid = {}
    for r in rows:
        by_lid.setdefault(r["linac_id"], []).append(r)
    for lid, rs in by_lid.items():
        assert all(r["knobs"] == rs[0]["knobs"] for r in rs)
        if lid < 3:
            assert sum(r["injector"] == golden and r["beam"] == "impact" for r in rs) == 1
    for r in rows:
        if r["block"] in ("paired_golden", "bridge"):
            assert r["injector"] == golden
    assert rows[1]["injector"]["laser_timing_ps"] == pytest.approx(0.2)
    assert rows[2]["injector"]["drive_fraction"] == pytest.approx(0.77)
    keys, lo, hi, _ = resolve_sweep_set("xtcav23_inj12")
    for r in rows:
        v = np.array([{**r["knobs"], **r["injector"]}[k] for k in keys])
        assert np.all(v >= np.array(lo) - 1e-12) and np.all(v <= np.array(hi) + 1e-12)
    # Impact-T beams: one per injector point (bridge/paired rows reuse)
    assert len(ic.injector_points(rows)) == n_inj
    assert ic.make_rows(_cfg(tmp_path)) == rows        # deterministic


def test_box_scale_and_manifest_written_once(tmp_path):
    rows = ic.make_rows(_cfg(tmp_path, injector={"golden": False, "lhs": 50, "box_scale": 0.5}))
    t = np.array([r["injector"]["laser_timing_ps"] for r in rows if r["block"] == "nested"])
    assert np.abs(t).max() <= 0.15 + 1e-12 and np.abs(t).max() > 0.1
    cfg = _cfg(tmp_path)
    a = ic.build_manifest(cfg)
    cfg["design"]["seed"] = 8
    assert ic.build_manifest(cfg) == a                  # reused on resume, never reshuffled


def test_row_paths_and_absolute_shift(tmp_path):
    assert ic.row_json_path(tmp_path, {"idx": 3, "beam": "impact"}).name == "sample_00003.json"
    assert ic.row_json_path(tmp_path, {"idx": 3, "beam": "golden_h5"}).name == "bridge_00003.json"
    assert ic.absolute_time_shift({"mean_t_l0afend_s": 1.38e-8 + 1.6e-13},
                                  {"mean_t_l0afend_s": 1.38e-8}) == pytest.approx(1.6e-13, rel=1e-6)


def test_centred_arrival_refused(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["injector"]["arrival"] = "centred"
    p = tmp_path / "c.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="absolute"):
        ic.load_cfg(p)


# ---------------------------------------------------------------- embedding / merging

h5py = pytest.importorskip("h5py")
pytest.importorskip("pmd_beamphysics")
from twobunch_s2e_rl.surrogate.preprocess import preprocess  # noqa: E402

LINAC_GOLDEN = {k: v[2] for k, v in XTCAV23_PARAMS.items()}
FIXED = {"Q19851kG": -63.76, "trackEnd": "PENT"}


def _write_samples(d, rows, q_in=False):
    for r in rows:
        rec = {"idx": r["idx"], "knobs": r["knobs"], "success": True, "specs": {"XTCAVF": {}}}
        if q_in:
            rec["injected_charge_nC"] = {"drive": 1.2 * r["knobs"].get("drive_fraction", 0.75) / 0.75,
                                         "witness": 0.4}
        (d / f"sample_{r['idx']:05d}.json").write_text(json.dumps(rec))


def _golden_campaign(root, name="old", injector_settings=True, sha=GOLDEN_SHA, n=3):
    d = root / name
    d.mkdir()
    keys, lo, hi, _ = resolve_sweep_set("xtcav23")
    rng = np.random.default_rng(0)
    _write_samples(d, [{"idx": i, "knobs": {k: float(rng.uniform(a, b)) for k, a, b in zip(keys, lo, hi)}}
                       for i in range(n)])
    fixed = {"baseline_settings": dict(LINAC_GOLDEN), "fixed_overrides": dict(FIXED),
             "baseline_config_sha256": "abc", "input_beam_sha256": sha, "csrTF": True,
             "transverseWakes": True, "num_macro_particles": 100000}
    if injector_settings:
        fixed["injector_settings"] = injector_golden()
    camp = {"name": name, "swept": {"sweep_set": "xtcav23", "knob_keys": keys}, "fixed": fixed}
    return d, camp


def _injector_campaign(root, name="inj", grid=64, n=4):
    d = root / name
    d.mkdir()
    keys, lo, hi, _ = resolve_sweep_set("xtcav23_inj12")
    rng = np.random.default_rng(1)
    _write_samples(d, [{"idx": i, "knobs": {k: float(rng.uniform(a, b)) for k, a, b in zip(keys, lo, hi)}}
                       for i in range(n)], q_in=True)
    cfg = {"name": name, "sweep_set": "xtcav23_inj12", "linac_sweep_set": "xtcav23",
           "injector_sweep_set": "inj12", "csrTF": True, "transverseWakes": True,
           "baseline_config": "x.yml", "fixed_overrides": dict(FIXED),
           "injector": {"n_particles": 100000, "grid": grid, "arrival": "absolute"}}
    rec = ic.campaign_record(cfg, dict(LINAC_GOLDEN), {"L0AF_phase:theta0_deg": 359.86},
                             {"code_version": "injector-v1", "cache_key": "k", "mean_t_l0afend_s": 1.4e-8},
                             GOLDEN_SHA)
    rec["fixed"]["baseline_config_sha256"] = "abc"
    return d, rec


def _run(root, subdir, sweep_set, cjs=None, allow_drop=()):
    out = root / "out.h5"
    preprocess(subdir=subdir, p=16, sweep_set=sweep_set, treaty_point="XTCAVF", out=str(out),
               verbose=False, data_root=root, campaign_jsons=cjs, allow_drop=allow_drop)
    with h5py.File(out) as h:
        return {k: h[k][...] for k in h}


def test_golden_campaign_embeds_at_injector_golden_via_external_json(tmp_path):
    d_old, camp_old = _golden_campaign(tmp_path)
    d_inj, camp_inj = _injector_campaign(tmp_path)
    # the old campaign's record lives OUTSIDE its data dir (backfill; data/ untouched)
    cj_old = tmp_path / "backfill_old.json"
    cj_old.write_text(json.dumps(camp_old))
    (d_inj / "campaign.json").write_text(json.dumps(camp_inj))
    assert not (d_old / "campaign.json").exists()
    H = _run(tmp_path, "old,inj", "xtcav23_inj12", cjs={"old": str(cj_old)})
    keys = resolve_sweep_set("xtcav23_inj12")[0]
    K, camp = H["knobs"], H["campaign"]
    assert K.shape == (7, 35)
    for k in INJECTOR_KEYS:
        np.testing.assert_allclose(K[camp == 0, keys.index(k)], injector_golden()[k], rtol=1e-6)
    assert np.ptp(K[camp == 1, keys.index("laser_timing_ps")]) > 0
    # injected charges: recorded by the injector campaign, NaN for the old one
    assert np.isnan(H["drive_q_in_nC"][camp == 0]).all() and np.isfinite(H["drive_q_in_nC"][camp == 1]).all()


def test_golden_campaign_without_injector_settings_refused(tmp_path):
    d_old, camp_old = _golden_campaign(tmp_path, injector_settings=False)
    (d_old / "campaign.json").write_text(json.dumps(camp_old))
    with pytest.raises(ValueError, match="no fixed value"):
        _run(tmp_path, "old", "xtcav23_inj12")


def test_injector_merge_checks(tmp_path):
    d_old, camp_old = _golden_campaign(tmp_path, sha="other-beam")
    (d_old / "campaign.json").write_text(json.dumps(camp_old))
    d_inj, camp_inj = _injector_campaign(tmp_path)
    (d_inj / "campaign.json").write_text(json.dumps(camp_inj))
    with pytest.raises(ValueError, match="input beam"):    # golden equivalent != that beam
        _run(tmp_path, "old,inj", "xtcav23_inj12")
    d2, camp2 = _injector_campaign(tmp_path, name="inj32", grid=32)
    (d2 / "campaign.json").write_text(json.dumps(camp2))
    with pytest.raises(ValueError, match="injector grid"):
        _run(tmp_path, "inj,inj32", "xtcav23_inj12")
    d3, camp3 = _injector_campaign(tmp_path, name="inj_b")
    (d3 / "campaign.json").write_text(json.dumps(camp3))
    assert _run(tmp_path, "inj,inj_b", "xtcav23_inj12")["knobs"].shape == (8, 35)


def test_knob_drop_rule_for_injector_knobs(tmp_path):
    d_inj, camp_inj = _injector_campaign(tmp_path)
    (d_inj / "campaign.json").write_text(json.dumps(camp_inj))
    with pytest.raises(ValueError, match="not downstream"):  # 23-knob model can't drop them silently
        _run(tmp_path, "inj", "xtcav23")
    H = _run(tmp_path, "inj", "xtcav23", allow_drop=INJECTOR_KEYS)   # explicit marginalization
    assert H["knobs"].shape == (4, 23)
