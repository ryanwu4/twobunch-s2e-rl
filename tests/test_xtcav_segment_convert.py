"""Converter contract tests (plan gotcha #6): t is authoritative, signs pinned.

Ported from xtcav-gpsr-tuning@c1f68ce tests/test_convert.py; runs on cheetah-accelerator 0.8.4.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xtcav_segment.convert import C_LIGHT, particlegroup_to_beam  # noqa: E402

pmd = pytest.importorskip("pmd_beamphysics")
from pmd_beamphysics import ParticleGroup  # noqa: E402

E_REF = 10e9  # eV
REPO_ROOT = Path(__file__).resolve().parents[1]


def make_pg(t, pz_eV=None, x=None):
    n = len(t)
    data = {
        "x": np.asarray(x if x is not None else np.zeros(n), dtype=float),
        "y": np.zeros(n),
        "z": np.zeros(n),
        "px": np.zeros(n),
        "py": np.zeros(n),
        "pz": np.asarray(pz_eV if pz_eV is not None else np.full(n, E_REF), dtype=float),
        "t": np.asarray(t, dtype=float),
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, 1e-9 / n),
        "species": "electron",
    }
    return ParticleGroup(data=data)


def test_tau_sign_and_reference():
    """Earlier arrival (smaller t) = head = Bmad z>0 = Cheetah tau<0."""
    t0 = 3.3e-6
    dt = 1e-13  # 30 um
    P = make_pg([t0 - dt, t0, t0 + dt])
    beam, info = particlegroup_to_beam(P, E_REF, ref_time_s=t0)
    tau = beam.particles[..., 4].numpy()
    assert info["ref_time_s"] == t0
    # float64 ulp of t~3.3e-6 s is ~7e-22 s -> (t - t0) at dt=1e-13 is good to
    # ~7e-9 relative; that precision floor, not the formula, sets the tolerance
    np.testing.assert_allclose(tau, [-C_LIGHT * dt, 0.0, C_LIGHT * dt], rtol=3e-8)
    # and the beam's own stored z (as getBeamAtElement writes it) is -c*delta_t = -tau
    P.z = -C_LIGHT * (P.t - np.average(P.t, weights=P.weight))
    np.testing.assert_allclose(P.z, -tau, rtol=1e-9, atol=1e-18)


def test_nonzero_t_default_reference_is_weighted_mean():
    """With ref_time_s=None the charge-weighted mean arrival is used (z-vs-t trap)."""
    t = np.array([3.3e-6, 3.3e-6 + 2e-13])
    P = make_pg(t)
    P.weight = np.array([3e-9, 1e-9])
    beam, info = particlegroup_to_beam(P, E_REF)
    t_mean = np.average(t, weights=P.weight)
    assert abs(info["ref_time_s"] - t_mean) < 1e-19
    tau = beam.particles[..., 4].numpy()
    np.testing.assert_allclose(tau, C_LIGHT * (t - t_mean), rtol=1e-9, atol=1e-15)


def test_matches_cheetah_importer_when_t_is_relative():
    """Cheetah's own from_openpmd_particlegroup uses tau = +c*t (absolute).
    On a group whose t is already relative (mean 0) the two agree exactly,
    which pins our convention to Cheetah's."""
    import cheetah

    t = np.array([-1e-13, 0.0, 1e-13])
    P = make_pg(t, x=[1e-6, 0.0, -1e-6])
    ours, _ = particlegroup_to_beam(P, E_REF, ref_time_s=0.0)
    theirs = cheetah.ParticleBeam.from_openpmd_particlegroup(
        P, energy=torch.tensor(E_REF, dtype=torch.float64), dtype=torch.float64
    )
    np.testing.assert_allclose(
        ours.particles[..., :6].numpy(), theirs.particles[..., :6].numpy(),
        rtol=1e-12, atol=1e-18,
    )


def test_cloud_to_beam_matches_particlegroup_path():
    """cloud_to_beam (z-based, differentiable) must agree with particlegroup_to_beam
    (t-based) on a beam whose z = -c*(t - <t>_w) — the campaign save convention."""
    from xtcav_segment.convert import cloud_to_beam

    rng = np.random.default_rng(3)
    n = 200
    z = rng.normal(0, 5e-5, n)
    z -= np.average(z)  # centered frame (weights equal below)
    px = rng.normal(0, 1e5, n)
    py = rng.normal(0, 1e5, n)
    pz = rng.normal(E_REF, 0.01 * E_REF, n)
    t0 = 3.3e-6
    data = {
        "x": rng.normal(0, 1e-4, n), "y": rng.normal(0, 1e-4, n), "z": z,
        "px": px, "py": py, "pz": pz,
        "t": t0 - z / 299792458.0,  # so that c*(t - <t>) = -z exactly
        "status": np.ones(n, dtype=int), "weight": np.full(n, 1e-9 / n),
        "species": "electron",
    }
    P = ParticleGroup(data=data)
    via_t, _ = particlegroup_to_beam(P, E_REF)

    cloud = torch.tensor(
        np.stack([P.x, P.y, z, px, py, pz], axis=1), dtype=torch.float64
    )
    via_z = cloud_to_beam(cloud, E_REF, torch.full((n,), 1e-9 / n, dtype=torch.float64))

    # atol floor: the t-path subtracts ~3.3e-6 s timestamps (float64 ulp ~4e-22 s
    # -> c*ulp ~1.3e-13 m of tau noise); the z-path is exact.
    np.testing.assert_allclose(
        via_z.particles[..., :6].numpy(), via_t.particles[..., :6].numpy(),
        rtol=1e-9, atol=5e-13,
    )


def test_cloud_to_beam_differentiable():
    """δ gradients must flow through the converter (the whole point of the fit loop)."""
    from xtcav_segment.convert import cloud_to_beam

    cloud = torch.randn(64, 6, dtype=torch.float64)
    cloud[:, 5] = E_REF * (1 + 0.001 * cloud[:, 5])
    cloud.requires_grad_(True)
    beam = cloud_to_beam(cloud, E_REF, torch.full((64,), 1e-12, dtype=torch.float64))
    beam.particles[..., :6].square().sum().backward()
    assert cloud.grad is not None and torch.isfinite(cloud.grad).all()


def test_cloud_to_beam_against_real_beam():
    """On a real campaign beam, the two conversion paths agree (skipped if absent)."""
    from xtcav_segment.convert import cloud_to_beam
    f = REPO_ROOT / "data" / "xtcav23_pilot" / "sample_00000_XTCAVF.h5"
    if not f.exists():
        pytest.skip("pilot beam not available")
    P = ParticleGroup(str(f))
    E_ref = float(np.median(P.energy))  # any common reference works for the identity
    via_t, _ = particlegroup_to_beam(P, E_ref)
    cloud = torch.tensor(
        np.stack([P.x, P.y, P.z, P.px, P.py, P.pz], axis=1), dtype=torch.float64
    )
    via_z = cloud_to_beam(cloud, E_ref, torch.as_tensor(P.weight))
    d = (via_z.particles[..., :6] - via_t.particles[..., :6]).abs().max()
    assert float(d) < 1e-9, f"max coord mismatch {float(d)}"


def test_delta_uses_lattice_reference_energy():
    P = make_pg([0.0, 0.0], pz_eV=[E_REF, 1.001 * E_REF])
    beam, info = particlegroup_to_beam(P, E_REF, ref_time_s=0.0)
    delta = beam.particles[..., 5].numpy()
    p0c = info["p0c_eV"]
    E = np.asarray(P.energy)
    np.testing.assert_allclose(delta, (E - E_REF) / p0c, rtol=1e-12)


def test_beam_to_cloud_inverts_cloud_to_beam():
    from xtcav_segment.convert import beam_to_cloud, cloud_to_beam

    rng = np.random.default_rng(5)
    n = 300
    cloud = torch.tensor(np.stack([
        rng.normal(0, 1e-4, n), rng.normal(0, 1e-4, n), rng.normal(0, 5e-5, n),
        rng.normal(0, 1e5, n), rng.normal(0, 1e5, n), rng.normal(E_REF, 0.01 * E_REF, n),
    ], axis=1), dtype=torch.float64)
    q = torch.full((n,), 1e-9 / n, dtype=torch.float64)
    back = beam_to_cloud(cloud_to_beam(cloud, E_REF, q))
    np.testing.assert_allclose(back[:, :3].numpy(), cloud[:, :3].numpy(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(back[:, 3:].numpy(), cloud[:, 3:].numpy(), rtol=1e-9, atol=1e-3)
    # batched beams (vectorized energy-free batch dim) keep the shape
    back_b = beam_to_cloud(cloud_to_beam(cloud.expand(2, n, 6), E_REF, q))
    assert back_b.shape == (2, n, 6)
