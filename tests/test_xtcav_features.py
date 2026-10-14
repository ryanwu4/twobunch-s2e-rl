"""Particle-space XTCAV screen features on synthetic two-bunch data with known answers."""
import pytest

torch = pytest.importorskip("torch")
from xtcav_segment.features import calibrate_streak, screen_features  # noqa: E402

S, ETA, E0 = -13.0, 0.053, 10e9


def _bunches(spacing=150e-6, sz_d=30e-6, sz_w=10e-6, dE=150e6, sx0=0.0, x_off=(0.0, 0.0),
             sd=20e6, n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda: torch.randn(1, n, generator=g, dtype=torch.float64)
    z_d, z_w = sz_d * r(), sz_w * r() - spacing        # spacing = <z>_d - <z>_w
    e_d, e_w = sd * r(), sd * 0.3 * r() - dE           # energy offsets [eV]
    xo_d, xo_w = x_off[0] + sx0 * r(), x_off[1] + sx0 * r()   # TCAV-off x (offset + spot)
    img = lambda z, xo, sgn: sgn * S * z + xo
    return dict(xp_d=img(z_d, xo_d, 1), xm_d=img(z_d, xo_d, -1), y_d=ETA * e_d / E0,
                xp_w=img(z_w, xo_w, 1), xm_w=img(z_w, xo_w, -1), y_w=ETA * e_w / E0,
                q_d=torch.tensor([1.2], dtype=torch.float64), q_w=torch.tensor([0.4], dtype=torch.float64))


def _feat(b, **kw):
    return screen_features(**b, streak=S, eta_y=ETA, ref_energy_eV=E0, **kw)


@pytest.mark.parametrize("split", ["energy", "zlobe"])
def test_well_separated_bunches_recover_truth(split):
    f = _feat(_bunches(), split=split)
    assert float(f["screen_valid"]) == 1.0
    torch.testing.assert_close(f["bunch_spacing"], torch.tensor([150e-6], dtype=torch.float64), rtol=0.02, atol=0)
    torch.testing.assert_close(f["drive_sigma_z"], torch.tensor([30e-6], dtype=torch.float64), rtol=0.03, atol=0)
    torch.testing.assert_close(f["drive_charge_fraction"], torch.tensor([0.75], dtype=torch.float64), rtol=0.01, atol=0)
    # z-lobe: a few drive time-tail particles land witness-side and pull the energy centroids
    # (bunches 150 MeV apart) -> a couple of % on dE; the energy gate has no such leakage here
    torch.testing.assert_close(f["energy_difference"], torch.tensor([150e6], dtype=torch.float64),
                               rtol=0.02 if split == "energy" else 0.05, atol=0)


def test_plus_minus_v_cancels_tcav_off_offsets_and_keeps_spot_blur():
    """Different TCAV-off x offsets per bunch cancel in the spacing; the TCAV-off spot stays in
    the measured sigma_z as resolution (blur included, as deployed)."""
    sx0 = 260e-6                                       # spot -> 20 um blur at |S| = 13
    f = _feat(_bunches(x_off=(400e-6, -300e-6), sx0=sx0), split="energy")
    torch.testing.assert_close(f["bunch_spacing"], torch.tensor([150e-6], dtype=torch.float64), rtol=0.02, atol=0)
    want = (30e-6 ** 2 + (sx0 / abs(S)) ** 2) ** 0.5
    torch.testing.assert_close(f["drive_sigma_z"], torch.tensor([want], dtype=torch.float64), rtol=0.03, atol=0)


def test_validity_flags():
    assert float(_feat(_bunches(spacing=30e-6), split="energy")["screen_valid"]) == 0.0   # < floor
    overlap = _feat(_bunches(dE=5e6), split="energy")                                     # E overlap
    assert float(overlap["screen_misassign"]) > 0.1 and float(overlap["screen_valid"]) == 0.0
    assert float(_feat(_bunches(dE=5e6), split="zlobe")["screen_valid"]) == 1.0           # time-split ok


def test_features_differentiable():
    b = _bunches(n=2000)
    for k in ("xp_d", "xm_d", "xp_w", "xm_w"):
        b[k] = b[k].clone().requires_grad_(True)
    f = _feat(b, split="zlobe")
    (f["bunch_spacing"] + f["drive_sigma_z"]).sum().backward()
    assert all(torch.isfinite(b[k].grad).all() and b[k].grad.abs().sum() > 0
               for k in ("xp_d", "xm_d", "xp_w", "xm_w"))


def test_calibrate_streak():
    z = torch.randn(1, 5000, dtype=torch.float64) * 30e-6
    assert abs(calibrate_streak(S * z + 1e-3, z) - S) < 1e-9


def _lumpy_drive(n=20000, seed=1, **kw):
    """Drive with a flat-top energy distribution and an internal dip deeper than the drive-witness
    valley (what the flow surrogate produces): the two-most-prominent-peaks rule cuts inside the
    drive; the nearest-peak valley rule must not."""
    b = _bunches(n=n, seed=seed, **kw)
    g = torch.Generator().manual_seed(seed + 7)
    u = torch.rand(1, n, generator=g, dtype=torch.float64)
    e = torch.where(u < 0.5, -60e6 + 50e6 * torch.rand(1, n, generator=g, dtype=torch.float64),
                    20e6 + 80e6 * torch.rand(1, n, generator=g, dtype=torch.float64))   # dip at -10..20 MeV
    b["y_d"] = ETA * e / E0
    return b


def test_valley_threshold_and_row_correction():
    b = _lumpy_drive(dE=110e6)
    mid = _feat(b, split="energy")
    val = _feat(b, split="energy", threshold="valley")
    assert float(val["screen_misassign"]) < 0.02 < float(mid["screen_misassign"])
    assert float(val["screen_valid"]) == 1.0
    torch.testing.assert_close(val["bunch_spacing"], torch.tensor([150e-6], dtype=torch.float64), rtol=0.02, atol=0)


def test_row_correction_removes_x_of_energy_shear():
    """A TCAV-off x proportional to energy (residual dispersion) inflates the measured sigma_z;
    the off-shot row correction removes it (and leaves the spacing alone)."""
    b = _bunches()
    for s in ("d", "w"):
        shear = 0.3 * b[f"y_{s}"] / ETA                    # x_off = R16_eff * delta, R16_eff = 0.3 m
        b[f"xp_{s}"], b[f"xm_{s}"] = b[f"xp_{s}"] + shear, b[f"xm_{s}"] + shear
    raw = _feat(b, split="energy", threshold="valley")
    cor = _feat(b, split="energy", threshold="valley", row_correct=True)
    assert float(raw["drive_sigma_z"]) > 1.1 * 30e-6
    torch.testing.assert_close(cor["drive_sigma_z"], torch.tensor([30e-6], dtype=torch.float64), rtol=0.05, atol=0)
    torch.testing.assert_close(cor["bunch_spacing"], raw["bunch_spacing"], rtol=1e-3, atol=0)


def test_v2_features_differentiable():
    b = _bunches(n=2000)
    for k in ("xp_d", "xm_d", "xp_w", "xm_w", "y_d", "y_w"):
        b[k] = b[k].clone().requires_grad_(True)
    f = _feat(b, split="energy", threshold="valley", row_correct=True, resolution_m=15e-6)
    (f["bunch_spacing"] + f["drive_sigma_z"] + f["drive_charge_fraction"]).sum().backward()
    assert all(torch.isfinite(b[k].grad).all() and b[k].grad.abs().sum() > 0
               for k in ("xp_d", "xm_d", "xp_w", "xm_w", "y_d", "y_w"))
