"""Differentiable DTOTR2 screen rendering (Stage C, plan C1).

Ported unchanged (apart from dropping the Cheetah-clone ``sys.path`` hook) from
xtcav-gpsr-tuning @ c1f68ce ``src/xtcav_gpsr_tuning/screen.py``; runs on
cheetah-accelerator 0.8.4.

Conventions:
* Image axes follow the gpsr/xtcav convention: axis -2 = y (energy/dispersion),
  axis -1 = x (streak/time). Default grid = the deployed analysis frame,
  300 x-pixels × 160 y-pixels at 30.5 µm/px, centered on the anchor.
* Anchoring happens on PARTICLE COORDINATES before rendering (differentiable
  mean/soft-peak subtraction), not by shifting images — both sim and target go
  through the exact same op (same-estimator principle).
* Charge: pass per-particle charges for BOTH bunches CONCATENATED into ONE
  render call — `kde_histogram_2d` self-normalizes every call to unit sum, so
  rendering bunches separately would silently destroy the drive/witness charge
  split (Stage-C critic finding 7).
* Stage-C bandwidth = 30.5 µm (pixel scale), identical on both sides. The
  real-PSF bandwidth (screen-µm σ_eff discipline) is a Stage-D setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


PIXEL_UM = 30.5
NX, NY = 300, 160  # deployed analysis frame (x=streak, y=energy)


@dataclass
class ScreenGrid:
    pixel_m: float = PIXEL_UM * 1e-6
    nx: int = NX
    ny: int = NY
    bandwidth_m: float = PIXEL_UM * 1e-6
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"
    bins_x: torch.Tensor = field(init=False)
    bins_y: torch.Tensor = field(init=False)

    def __post_init__(self):
        half_x = self.pixel_m * (self.nx - 1) / 2
        half_y = self.pixel_m * (self.ny - 1) / 2
        self.bins_x = torch.linspace(-half_x, half_x, self.nx,
                                     dtype=self.dtype, device=self.device)
        self.bins_y = torch.linspace(-half_y, half_y, self.ny,
                                     dtype=self.dtype, device=self.device)


def _weighted_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * v).sum(dim=-1, keepdim=True) / w.sum(dim=-1, keepdim=True)


def soft_major_lobe_anchor(x: torch.Tensor, w: torch.Tensor, grid: ScreenGrid,
                           temperature: float = 0.15) -> torch.Tensor:
    """Differentiable proxy of the deployed x-projection-peak anchor: soft-argmax
    (softmax over the 1-D KDE profile, temperature relative to its max) of the
    streak-axis profile. temperature→0 approaches the hard peak; 0.15 keeps
    gradients healthy while staying lobe-selective."""
    from cheetah.utils.kde import kde_histogram_1d

    xc = x - _weighted_mean(x, w)  # pre-center so the wide profile grid suffices
    span = 3.0 * xc.detach().abs().amax() + grid.pixel_m
    bins = torch.linspace(-float(span), float(span), 2 * grid.nx,
                          dtype=x.dtype, device=x.device)
    prof = kde_histogram_1d(xc, bins, torch.tensor(grid.bandwidth_m, dtype=x.dtype,
                                                   device=x.device), weights=w)
    logits = prof / (prof.amax(dim=-1, keepdim=True) * temperature)
    soft = torch.softmax(logits, dim=-1)
    peak = (soft * bins).sum(dim=-1, keepdim=True)
    return _weighted_mean(x, w) + peak  # anchor position in original coords


def render_image(
    x: torch.Tensor,
    y: torch.Tensor,
    charges: torch.Tensor,
    grid: ScreenGrid,
    anchor: str = "com",
    y_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """(..., N) particle coords + charges -> (..., ny, nx) unit-sum image.

    anchor: 'com' (charge-weighted mean removed in x AND y) or 'major_lobe'
    (soft peak anchor in x, COM in y) or 'none'.
    y_scale: optional ν_y nuisance — scales y about its anchor (a dispersion/
    energy-axis scale error), applied identically inside the differentiable graph.
    """
    from cheetah.utils.kde import kde_histogram_2d

    if x.dim() == 1:
        x, y, charges = x.unsqueeze(0), y.unsqueeze(0), charges.unsqueeze(0)

    if anchor == "com":
        x = x - _weighted_mean(x, charges)
        y = y - _weighted_mean(y, charges)
    elif anchor == "major_lobe":
        x = x - soft_major_lobe_anchor(x, charges, grid)
        y = y - _weighted_mean(y, charges)
    elif anchor != "none":
        raise ValueError(f"unknown anchor {anchor!r}")

    if y_scale is not None:
        y = y * y_scale

    bw = torch.as_tensor(grid.bandwidth_m, dtype=x.dtype, device=x.device)
    # kde_histogram_2d output is (..., n_bins_x1, n_bins_x2); we pass (y, x) so
    # the image lands as (..., ny, nx) = (energy, streak), unit-sum per image.
    return kde_histogram_2d(y, x, grid.bins_y, grid.bins_x, bw, weights=charges)


def render_two_bunch(
    drive_xy: tuple[torch.Tensor, torch.Tensor],
    witness_xy: tuple[torch.Tensor, torch.Tensor],
    q_drive: torch.Tensor,
    q_witness: torch.Tensor,
    grid: ScreenGrid,
    anchor: str = "com",
    y_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate both bunches into ONE kde call (charge split preserved).

    q_drive/q_witness: per-particle charges, already scaled by T_k*Q_k_full/n.
    """
    x = torch.cat([drive_xy[0], witness_xy[0]], dim=-1)
    y = torch.cat([drive_xy[1], witness_xy[1]], dim=-1)
    q = torch.cat([torch.broadcast_to(q_drive, drive_xy[0].shape),
                   torch.broadcast_to(q_witness, witness_xy[0].shape)], dim=-1)
    return render_image(x, y, q, grid, anchor=anchor, y_scale=y_scale)


def charge_split(x: torch.Tensor, charges: torch.Tensor, temperature_m: float = 30e-6):
    """Differentiable z-lobe charge split: soft fraction of charge on each side of
    the charge-weighted x median (sigmoid with a temperature ~1 pixel). The z-lobe
    split is the robust real observable (never the energy gate — plan gotcha #4)."""
    med = _weighted_mean(x, charges)  # mean as the differentiable stand-in
    s = torch.sigmoid((x - med) / temperature_m)
    w = charges / charges.sum(dim=-1, keepdim=True)
    hi = (w * s).sum(dim=-1)
    return hi  # fraction of charge on the +x side


def image_zlobe_split(img: torch.Tensor) -> torch.Tensor:
    """Fraction of IMAGE intensity on the +x side of the image x-COM (soft
    sigmoid, ~1 px temperature). This is THE Stage-D charge-split estimator: it
    is applied identically to the real target and to the rendered model image,
    so it is sign-DEPENDENT by construction (the two lobes swap sides with the
    streak sign). Do not substitute the flow's drive-charge fraction, which is
    sign-invariant and therefore compares against the witness side on one of
    the two sign rows (audit 2026-08-07, finding #5)."""
    nx = img.shape[-1]
    xs = (torch.arange(nx, dtype=img.dtype, device=img.device)
          - (nx - 1) / 2.0) * (PIXEL_UM * 1e-6)
    prof = img.sum(dim=-2)
    w = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    com = (w * xs).sum(dim=-1, keepdim=True)
    sgm = torch.sigmoid((xs - com) / 30e-6)
    return (w * sgm).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Stage-D image-space anchoring (critic M6): the real 300x160 frame only exists
# after a COM-anchored crop, so the anchor op is defined ON IMAGES and applied
# verbatim to both target and model (model renders with anchor='none' first).
# --------------------------------------------------------------------------- #
def shift_image(img: torch.Tensor, dy_px: torch.Tensor, dx_px: torch.Tensor) -> torch.Tensor:
    """Differentiable sub-pixel translation of (..., ny, nx) images by
    (dy_px, dx_px) pixels (positive = content moves toward +row/+col).
    Bilinear grid_sample with zero padding; differentiable w.r.t. image AND shift."""
    orig_shape = img.shape
    ny, nx = orig_shape[-2], orig_shape[-1]
    flat = img.reshape(-1, 1, ny, nx)
    b = flat.shape[0]
    dy = torch.broadcast_to(torch.as_tensor(dy_px, dtype=img.dtype, device=img.device),
                            (b,))
    dx = torch.broadcast_to(torch.as_tensor(dx_px, dtype=img.dtype, device=img.device),
                            (b,))
    # normalized sampling offsets: sample at (x - dx), (y - dy)
    theta = torch.zeros(b, 2, 3, dtype=img.dtype, device=img.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx / max(nx - 1, 1)
    theta[:, 1, 2] = -2.0 * dy / max(ny - 1, 1)
    grid = torch.nn.functional.affine_grid(theta, flat.shape, align_corners=True)
    out = torch.nn.functional.grid_sample(flat, grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)
    return out.reshape(orig_shape)


def image_com(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel COM (row, col) of (..., ny, nx) images (nonnegative images assumed)."""
    ny, nx = img.shape[-2], img.shape[-1]
    iy = torch.arange(ny, dtype=img.dtype, device=img.device)
    ix = torch.arange(nx, dtype=img.dtype, device=img.device)
    tot = img.sum(dim=(-2, -1)).clamp_min(1e-30)
    cy = (img.sum(dim=-1) * iy).sum(dim=-1) / tot
    cx = (img.sum(dim=-2) * ix).sum(dim=-1) / tot
    return cy, cx


def com_center_image(img: torch.Tensor) -> torch.Tensor:
    """Shift each image so its pixel COM sits at the frame center (sub-pixel,
    differentiable). THE shared Stage-D anchor op for targets and model alike."""
    ny, nx = img.shape[-2], img.shape[-1]
    cy, cx = image_com(img)
    return shift_image(img, (ny - 1) / 2.0 - cy, (nx - 1) / 2.0 - cx)
