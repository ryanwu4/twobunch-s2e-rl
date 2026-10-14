"""Segment builder tests: crab_cavity intercept, PHI0 passthrough, determinism,
TDC(V=0) == drift. No Tao required (synthetic mini-spec).

Ported from xtcav-gpsr-tuning@c1f68ce tests/test_segment.py; runs on cheetah-accelerator 0.8.4.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xtcav_segment.segment import (  # noqa: E402
    build_segment_from_spec,
    element_by_name,
    set_tdc,
)

cheetah = pytest.importorskip("cheetah")

PHI0 = 0.125  # rad/2pi — deliberately NOT a multiple of anything degree-like


def mini_spec():
    def ed(name, key, s, attrs):
        return {"name": name, "key": key, "s": s, "e_tot": 10e9,
                "e_tot_start": 10e9, "attrs": attrs}

    return {
        "start": "XTCAVF",
        "end": "SCREEN",
        "meta": {"ref_energy_eV": 10e9},
        "elements": [
            ed("XTCAVF", "Crab_Cavity", 1.0,
               {"l": 1.0, "voltage": 0.0, "phi0": PHI0, "rf_frequency": 11.424e9,
                "tilt": 0.0}),
            ed("D1", "Drift", 3.0, {"l": 2.0}),
            ed("QX", "Quadrupole", 4.0, {"l": 1.0, "k1": 0.5, "tilt": 0.0}),
            ed("SCREEN", "Monitor", 4.0, {"l": 0.0}),
        ],
    }


def make_beam(n=64, seed=0):
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 7))
    p[:, 0] = rng.normal(0, 1e-4, n)
    p[:, 1] = rng.normal(0, 1e-5, n)
    p[:, 2] = rng.normal(0, 1e-4, n)
    p[:, 3] = rng.normal(0, 1e-5, n)
    p[:, 4] = rng.normal(0, 3e-5, n)
    p[:, 5] = rng.normal(0, 1e-3, n)
    p[:, 6] = 1.0
    from cheetah.particles import Species

    return cheetah.ParticleBeam(
        particles=torch.tensor(p, dtype=torch.float64),
        energy=torch.tensor(10e9, dtype=torch.float64),
        particle_charges=torch.ones(n, dtype=torch.float64),
        survival_probabilities=torch.ones(n, dtype=torch.float64),
        species=Species("electron", dtype=torch.float64),
    )


def test_crab_cavity_intercept_and_phi0_passthrough():
    seg, report = build_segment_from_spec(mini_spec())
    tdc = element_by_name(seg, "XTCAVF")
    assert isinstance(tdc, cheetah.TransverseDeflectingCavity)
    # PHI0 is rad/2pi in both Bmad and Cheetah TDC: NO degree/sign conversion
    assert float(tdc.phase) == pytest.approx(PHI0, abs=0)
    assert float(tdc.frequency) == pytest.approx(11.424e9)
    # crab_cavity must NOT appear in the approximated list (the bridge alone
    # would have turned it into a drift)
    assert not any(k == "crab_cavity" for _, k, _ in report.approximated)


def test_tracking_method_pinned():
    seg, _ = build_segment_from_spec(mini_spec())
    q = element_by_name(seg, "QX")
    assert q.tracking_method == "drift_kick_drift"
    d = element_by_name(seg, "D1")
    assert d.tracking_method == "drift_kick_drift"


def test_build_determinism():
    seg_a, _ = build_segment_from_spec(mini_spec())
    seg_b, _ = build_segment_from_spec(mini_spec())
    sd_a, sd_b = seg_a.state_dict(), seg_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k


def test_tdc_off_equals_drift():
    seg, _ = build_segment_from_spec(mini_spec())
    set_tdc(seg, 0.0, 0.0)
    tdc = element_by_name(seg, "XTCAVF")
    beam = make_beam()
    out_tdc = tdc.track(beam)
    drift = cheetah.Drift(length=torch.tensor(1.0, dtype=torch.float64))
    drift.tracking_method = "drift_kick_drift"
    out_drift = drift.track(beam)
    np.testing.assert_allclose(
        out_tdc.particles[..., :6].numpy(),
        out_drift.particles[..., :6].numpy(),
        rtol=0, atol=1e-15,
    )


def test_start_after_excludes_tdc():
    seg, report = build_segment_from_spec(mini_spec(), start_after="XTCAVF")
    with pytest.raises(KeyError):
        element_by_name(seg, "XTCAVF")
    assert report.n_in == 3


def test_set_tdc_units():
    """set_tdc takes degrees (Bmad set-command parity) and stores rad/2pi."""
    seg, _ = build_segment_from_spec(mini_spec())
    set_tdc(seg, 22e6, 90.0)
    tdc = element_by_name(seg, "XTCAVF")
    assert float(tdc.voltage) == 22e6
    assert float(tdc.phase) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Added with the port (twobunch-s2e-rl)
# --------------------------------------------------------------------------- #
def test_cheetah_version_pinned():
    import xtcav_segment

    assert cheetah.__version__ == xtcav_segment.CHEETAH_VERSION == "0.8.4"


def test_end_cut_keeps_named_element():
    seg, report = build_segment_from_spec(mini_spec(), start_after="XTCAVF", end="QX")
    element_by_name(seg, "QX")
    with pytest.raises(KeyError):
        element_by_name(seg, "SCREEN")
    assert report.n_in == 2
    with pytest.raises(ValueError):
        build_segment_from_spec(mini_spec(), end="NOPE")


def test_exported_specs_build_without_approximations():
    """Every spec in configs/segment builds, tiles s exactly, approximates nothing,
    and pins DKD on every element that supports it."""
    from xtcav_segment.segment import SPEC_DIR, load_spec

    specs = sorted(SPEC_DIR.glob("segment_*.json"))
    assert specs, "no exported specs in configs/segment"
    for p in specs:
        seg, report = build_segment_from_spec(load_spec(p))
        assert report.approximated == [], p.name
        assert abs(report.bmad_length - report.cheetah_length) < 1e-9, p.name
        assert isinstance(element_by_name(seg, "XTCAVF"),
                          cheetah.TransverseDeflectingCavity)
        for e in seg.flattened().elements:
            if "drift_kick_drift" in getattr(type(e), "supported_tracking_methods", []):
                assert e.tracking_method == "drift_kick_drift", (p.name, e.name)
        element_by_name(seg, "PENT")


def test_quad_kG_conversion_reproduces_spec():
    """defaults.yml golden FF kG -> k1 at the spec reference energy == exported k1."""
    from xtcav_segment.segment import load_spec, quad_k1_from_kG, set_quads_kG, spec_path

    spec = load_spec(spec_path("baseline"))
    E = spec["meta"]["ref_energy_eV"]
    golden = {"Q2FF": 126.35, "Q1FF": -235.218, "Q0FF": 126.353}
    by_name = {e["name"]: e for e in spec["elements"]}
    for name, kG in golden.items():
        a = by_name[name]["attrs"]
        assert quad_k1_from_kG(kG, a["l"], E) == pytest.approx(a["k1"], rel=1e-12)
    seg, _ = build_segment_from_spec(spec, start_after="XTCAVF", end="PENT")
    k1_before = float(element_by_name(seg, "Q1FF").k1)
    set_quads_kG(seg, golden, E)
    assert float(element_by_name(seg, "Q1FF").k1) == pytest.approx(k1_before, rel=1e-12)
    # differentiable in kG
    kG = torch.tensor(127.0, dtype=torch.float64, requires_grad=True)
    set_quads_kG(seg, {"Q0FF": kG}, E)
    beam = make_beam()
    seg.track(beam).x.square().sum().backward()
    assert kG.grad is not None and torch.isfinite(kG.grad)
