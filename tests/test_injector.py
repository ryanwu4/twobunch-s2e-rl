"""Tests for sim/injector.py (the minimal Impact-T injector wrapper).

Pure unit tests run in slac-rl or bmad-qpad-dev without Impact-T. The single
@pytest.mark.impact test runs a tiny Impact-T job (bmad-qpad-dev; skipped elsewhere):
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -m pytest \
      tests/test_injector.py -m impact
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from twobunch_s2e_rl.sim import injector as inj

FACET2_IMPACT = Path(os.environ.get("FACET2_S2E_ROOT", "/home/rwu4/photoinjector-rl/FACET2-S2E")) / "impact"


# ---------------------------------------------------------------- settings mapping

def test_resolve_golden_and_convenience_keys():
    r = inj.resolve_settings({"drive_fraction": 0.7, "sigma_t_ps": 1.2})
    assert r["distgen:t_dist:weights:d2"] == pytest.approx(0.7)
    assert r["distgen:t_dist:weights:d1"] == pytest.approx(0.3)
    assert r["distgen:t_dist:dists:d1:sigma_t:value"] == 1.2
    assert r["distgen:t_dist:dists:d2:sigma_t:value"] == 1.2
    assert inj.resolve_settings() == inj.GOLDEN_SETTINGS | {"hammersley_burnin": -1}


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        inj.resolve_settings({"GUNF:theta_deg": 1.0})   # typo of GUNF:theta0_deg


def test_split_settings_keys_and_laser_timing():
    r = inj.resolve_settings({"laser_timing_ps": 0.2, "distgen:t_dist:dists:d2:avg_t:value": 9.1,
                              "GUNF:theta0_deg": -60.0, "L0AF_phase_offset_deg": 0.3})
    impact, dg = inj.split_settings(r)
    assert set(impact) == {"GUNF:theta0_deg", "GUNF:rf_field_scale", "SOL10111:solenoid_field_scale"}
    assert impact["GUNF:theta0_deg"] == -60.0
    # common-mode shift lands on BOTH pulses; the delay is preserved
    assert dg["t_dist:dists:d1:avg_t:value"] == pytest.approx(0.2)
    assert dg["t_dist:dists:d2:avg_t:value"] == pytest.approx(9.3)
    assert dg["total_charge:value"] == 1600.0
    assert all(not k.startswith("distgen:") for k in dg)
    assert "L0AF_phase_offset_deg" not in impact and "laser_timing_ps" not in dg


def test_golden_values_match_golden_run():
    g = inj.GOLDEN_SETTINGS
    assert g["GUNF:theta0_deg"] == pytest.approx(-60.7)
    assert g["SOL10111:solenoid_field_scale"] == pytest.approx(-0.39 / 1.6)
    assert g["distgen:t_dist:weights:d2"] == 0.75 and g["distgen:t_dist:weights:d1"] == 0.25
    assert inj.golden_value("drive_fraction") == 0.75
    assert inj.golden_value("sigma_t_ps") == 1.17


# ---------------------------------------------------------------- cache key

def test_cache_key_deterministic_and_sensitive():
    ref = {"L0AF_phase:theta0_deg": 359.86, "L0AF_scale:rf_field_scale": 3.26e7}
    a = inj.cache_key(inj.resolve_settings({"laser_timing_ps": 0.1}), 20000, 32, ref)
    b = inj.cache_key(inj.resolve_settings({"laser_timing_ps": 0.1}), 20000, 32, dict(reversed(ref.items())))
    assert a == b and len(a) == 16
    assert a != inj.cache_key(inj.resolve_settings({"laser_timing_ps": 0.2}), 20000, 32, ref)
    assert a != inj.cache_key(inj.resolve_settings({"laser_timing_ps": 0.1}), 20000, 64, ref)
    assert a != inj.cache_key(inj.resolve_settings({"laser_timing_ps": 0.1, "hammersley_burnin": 7}), 20000, 32, ref)
    # convenience vs native spelling of the same point -> same beam
    assert (inj.cache_key(inj.resolve_settings({"drive_fraction": 0.73}), 10, 8, ref)
            == inj.cache_key(inj.resolve_settings({"distgen:t_dist:weights:d2": 0.73,
                                                   "distgen:t_dist:weights:d1": 0.27}), 10, 8, ref))


# ---------------------------------------------------------------- workdir isolation

def test_prepare_workdir_copies_templates_only_into_workdir(tmp_path):
    tpl = tmp_path / "facet2_impact"
    tpl.mkdir()
    for name in inj.TEMPLATE_FILES:
        (tpl / name).write_text(f"template {name}\n")
    before = {p.name: p.stat().st_mtime_ns for p in tpl.iterdir()}
    inputs = inj.prepare_workdir(tmp_path / "run_a", template_dir=tpl)
    assert inputs == tmp_path / "run_a" / "inputs"
    assert sorted(p.name for p in inputs.iterdir()) == sorted(inj.TEMPLATE_FILES)
    # a second run gets its own copies; editing one never touches the template or the other
    inputs_b = inj.prepare_workdir(tmp_path / "run_b", template_dir=tpl)
    (inputs / inj.IMPACT_YAML).write_text("edited\n")
    assert (inputs_b / inj.IMPACT_YAML).read_text().startswith("template")
    assert (tpl / inj.IMPACT_YAML).read_text().startswith("template")
    assert {p.name: p.stat().st_mtime_ns for p in tpl.iterdir()} == before
    assert "ImpactT.yaml" not in os.listdir(tpl)   # never (re)written, unlike runImpact


# ---------------------------------------------------------------- tagging invariant

def _synthetic_pg(n, frac_drive, rng):
    pmd = pytest.importorskip("pmd_beamphysics")
    data = {k: rng.normal(size=n) * 1e-3 for k in ("x", "y", "px", "py")}
    data.update(z=np.zeros(n), pz=np.full(n, 6e7) + rng.normal(size=n) * 1e5,
                t=rng.normal(size=n) * 1e-12, status=np.ones(n, int),
                weight=np.full(n, 1.6e-9 / n), species="electron")
    return pmd.ParticleGroup(data=data), rng.random(n) < frac_drive


@pytest.mark.parametrize("n,frac", [(10, 0.5), (1000, 0.75), (5000, 0.1), (777, 0.9)])
def test_weight_tag_invariant(n, frac):
    rng = np.random.default_rng(n)
    P, is_drive = _synthetic_pg(n, frac, rng)
    is_drive[0], is_drive[1] = True, False          # both bunches present
    q = float(P.weight[0])
    inj.apply_weight_tags(P, is_drive)
    inj.check_tagging(P)
    w = np.unique(P.weight)
    assert len(w) == 2 and w[1] > w[0]
    assert w[1] == pytest.approx(1.001 * q) and w[0] == pytest.approx(0.999 * q)
    d, wt = inj.split_drive_witness(P)
    assert len(d) == is_drive.sum() and len(wt) == (~is_drive).sum()
    # charge ratio is carried by counts; the nudge changes Q by f*0.1% - (1-f)*0.1%
    assert P.charge == pytest.approx(1.6e-9 * (1 + 1e-3 * (2 * is_drive.mean() - 1)))


def test_check_tagging_rejects_bad_beams():
    rng = np.random.default_rng(0)
    P, _ = _synthetic_pg(100, 0.5, rng)
    with pytest.raises(ValueError):
        inj.check_tagging(P)                           # one unique weight
    P.weight = np.where(np.arange(100) % 3 == 0, 1.0, np.where(np.arange(100) % 3 == 1, 2.0, 3.0)) * 1e-15
    with pytest.raises(ValueError):
        inj.check_tagging(P)                           # three unique weights


def test_tag_components_by_emission_time():
    r = inj.resolve_settings({"laser_timing_ps": -0.2})
    t = np.array([-3.0, 0.0, 3.0, 6.0, 8.8, 12.0]) * 1e-12
    assert inj.tag_components(t, r).tolist() == [False, False, False, True, True, True]
    # boundary moves with a larger drive weight (Bayes rule), toward the witness pulse
    t_mid = np.array([4.3e-12])
    assert not inj.tag_components(t_mid, inj.resolve_settings({"drive_fraction": 0.5}))[0]
    assert inj.ambiguous_count(np.array([0.0, 9e-12]), inj.resolve_settings()) == 0


# ---------------------------------------------------------------- tiny Impact-T run

def _impactt_available():
    return (Path(sys.executable).parent / "ImpactTexe").exists() or shutil.which("ImpactTexe")


@pytest.mark.impact
@pytest.mark.slow
def test_tiny_impact_run_tags_and_records_arrival(tmp_path):
    pytest.importorskip("impact")
    pytest.importorskip("distgen")
    if not _impactt_available():
        pytest.skip("ImpactTexe not found")
    if not FACET2_IMPACT.is_dir():
        pytest.skip("FACET2-S2E impact templates not found")
    # Pinned reference = the golden autophase result (recorded in Source.zip ImpactT.yaml);
    # skipping the autophase keeps the test short.
    ref = {"L0AF_phase:theta0_deg": 359.86190784927777, "L0AF_scale:rf_field_scale": 32599504.307795566}
    P, info = inj.injector_beam({}, tmp_path / "run", 2000, grid=8, l0af_reference=ref,
                                template_dir=FACET2_IMPACT)
    inj.check_tagging(P)
    d, w = inj.split_drive_witness(P)
    assert len(d) + len(w) == len(P) == info["n_l0afend"]
    assert abs(len(w) / len(P) - 0.25) < 0.01                   # 3:1 by counts
    assert info["mean_t_drive_s"] > info["mean_t_witness_s"]     # drive trails at L0AFEND
    assert 1e-8 < info["mean_t_l0afend_s"] < 2e-8                # recorded before centring
    assert abs(np.mean(P.t)) < 1e-18                             # then centred (unweighted)
    assert np.all(P.z == 0)
    assert 60e6 < info["mean_energy_eV"] < 75e6
    assert not any(p.name.startswith("tmp") for p in (tmp_path / "run").iterdir())
    # cache hit: same key -> loaded, not rerun
    P2, info2 = inj.injector_beam({}, tmp_path / "run", 2000, grid=8, l0af_reference=ref,
                                  template_dir=FACET2_IMPACT)
    assert info2["cache_key"] == info["cache_key"] and np.allclose(P2.t, P.t)


def test_grid_spec_isotropic_and_anisotropic():
    assert inj._grid(64) == (64, 64, 64) and inj._grid((48, 48, 128)) == (48, 48, 128)
    assert inj._grid_record(64) == 64                      # isotropic: cache keys unchanged
    assert inj._grid_record((48, 48, 128)) == [48, 48, 128]
    with pytest.raises(ValueError):
        inj._grid((32, 32))
