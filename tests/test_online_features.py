"""Online image features agree with the training-time particle-space features (decision D2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.online.features import PIXEL_M, band_stats, image_features, render, step_features  # noqa: E402
from xtcav_segment.features import screen_features  # noqa: E402
from tests.test_xtcav_features import ETA, E0, S, _bunches  # noqa: E402


def _compare(b):
    f = screen_features(**b, streak=S, eta_y=ETA, ref_energy_eV=E0, split="energy", threshold="valley",
                        orientation="charge")
    xp = torch.cat([b["xp_d"], b["xp_w"]], -1)[0].numpy()
    xm = torch.cat([b["xm_d"], b["xm_w"]], -1)[0].numpy()
    y = torch.cat([b["y_d"], b["y_w"]], -1)[0].numpy()
    nd, nw = b["xp_d"].shape[-1], b["xp_w"].shape[-1]
    q = np.r_[np.full(nd, float(b["q_d"]) / nd), np.full(nw, float(b["q_w"]) / nw)]
    lo, hi = min(xp.min(), xm.min()) - 5 * PIXEL_M, max(xp.max(), xm.max()) + 5 * PIXEL_M
    xe = np.arange(lo, hi + PIXEL_M, PIXEL_M)
    ye = np.arange(y.min() - 5 * PIXEL_M, y.max() + 6 * PIXEL_M, PIXEL_M)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    g = image_features(render(xp, y, q, xe, ye), render(xm, y, q, xe, ye), xc, yc, S)
    return f, g, (render(xp, y, q, xe, ye), render(xm, y, q, xe, ye), xc)


def test_image_features_match_particle_features():
    f, g, _ = _compare(_bunches())
    assert g["peaks_ok"] and g["valid"] == 1.0
    assert abs(g["bunch_spacing"] - float(f["bunch_spacing"])) < 1.5e-6           # < 1.5 um
    assert abs(g["drive_sigma_z"] / float(f["drive_sigma_z"]) - 1) < 0.03
    assert abs(g["drive_charge_fraction"] - float(f["drive_charge_fraction"])) < 0.01


def test_image_features_flag_small_spacing_invalid():
    _, g, _ = _compare(_bunches(spacing=30e-6))
    assert g["valid"] == 0.0


def test_step_features_reduce_to_image_features():
    _, g, (ip, im, xc) = _compare(_bunches())
    bp = band_stats(ip, xc)
    bm = band_stats(im, xc, cut_row=bp["cut_row"])
    st = step_features([bp, bp, None], [bm, bm], S)
    assert st["n_plus"] == 2 and st["valid"] == 1.0
    for k in ("bunch_spacing", "drive_sigma_z", "drive_charge_fraction"):
        assert abs(st[k] - g[k]) < 1e-9 * max(1.0, abs(g[k])) + 1e-12
    assert step_features([], [bm], S)["valid"] == 0.0
