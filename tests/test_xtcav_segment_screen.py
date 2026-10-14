"""screen.py contract tests: shapes, unit-sum, charge split preservation,
anchors, gradients, ν_y scale.

Ported from xtcav-gpsr-tuning@c1f68ce tests/test_screen.py; runs on cheetah-accelerator 0.8.4.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xtcav_segment.screen import (  # noqa: E402
    ScreenGrid,
    charge_split,
    render_image,
    render_two_bunch,
)

pytest.importorskip("cheetah")


def beams(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=torch.float64) * 2e-4
    y = torch.randn(n, generator=g, dtype=torch.float64) * 4e-4
    return x, y


def test_shape_and_unit_sum():
    grid = ScreenGrid()
    x, y = beams()
    q = torch.full_like(x, 1e-12)
    img = render_image(x, y, q, grid)
    assert img.shape == (1, grid.ny, grid.nx)
    assert abs(float(img.sum()) - 1.0) < 1e-6


def test_charge_split_preserved_in_single_call():
    """Two bunches at distinct x with 3:1 charge — the image mass on each side
    must reflect the charge ratio (the per-bunch-render trap would give 50/50)."""
    grid = ScreenGrid()
    xd, yd = beams(seed=1)
    xw, yw = beams(seed=2)
    img = render_two_bunch(
        (xd - 1.5e-3, yd), (xw + 1.5e-3, yw),
        torch.tensor(1.2e-9 / 400, dtype=torch.float64),
        torch.tensor(0.4e-9 / 400, dtype=torch.float64),
        grid, anchor="none",
    )[0]
    left = float(img[:, : grid.nx // 2].sum())
    right = float(img[:, grid.nx // 2:].sum())
    assert abs(left / (left + right) - 0.75) < 0.02


def test_anchors_center_the_image():
    grid = ScreenGrid()
    x, y = beams()
    q = torch.full_like(x, 1e-12)
    for anchor in ("com", "major_lobe"):
        img = render_image(x + 5e-3, y - 3e-3, q, grid, anchor=anchor)[0]
        ix = torch.arange(grid.nx, dtype=torch.float64)
        iy = torch.arange(grid.ny, dtype=torch.float64)
        cx = float((img.sum(0) * ix).sum() / img.sum()) - (grid.nx - 1) / 2
        cy = float((img.sum(1) * iy).sum() / img.sum()) - (grid.ny - 1) / 2
        assert abs(cx) < 6 and abs(cy) < 6, anchor  # within ~6 px of frame center


def test_gradients_flow_through_render_and_yscale():
    grid = ScreenGrid()
    x, y = beams()
    q = torch.full_like(x, 1e-12)
    x = x.clone().requires_grad_(True)
    nu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    img = render_image(x, y, q, grid, y_scale=nu)
    img.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert nu.grad is not None and torch.isfinite(nu.grad)


def test_yscale_stretches_energy_axis():
    grid = ScreenGrid()
    x, y = beams()
    q = torch.full_like(x, 1e-12)

    def y_rms_px(img):
        iy = torch.arange(grid.ny, dtype=torch.float64)
        p = img.sum(-1)
        m = (p * iy).sum() / p.sum()
        return float(((p * (iy - m) ** 2).sum() / p.sum()).sqrt())

    a = y_rms_px(render_image(x, y, q, grid)[0])
    b = y_rms_px(render_image(x, y, q, grid,
                              y_scale=torch.tensor(1.3, dtype=torch.float64))[0])
    assert b / a == pytest.approx(1.3, rel=0.05)


def test_charge_split_soft_metric():
    x = torch.cat([torch.full((300,), -1e-3), torch.full((100,), 1e-3)]).double()
    q = torch.cat([torch.full((300,), 3e-12), torch.full((100,), 1e-12)]).double()
    hi = charge_split(x, q)
    # +x side carries 100*1e-12 of the 1e-9 total = 0.10 CHARGE fraction
    # (0.25 would be the particle-count fraction — the wrong quantity)
    assert float(hi) == pytest.approx(0.10, abs=0.02)
