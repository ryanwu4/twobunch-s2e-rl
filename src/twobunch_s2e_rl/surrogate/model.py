"""Two-bunch conditional normalizing-flow surrogate (Lightning).

One shared RealNVP flow selected by a per-bunch index k (0=drive, 1=witness). A whitening
MLP maps (knobs,k) -> per-bunch (mu_k, L_k); the flow models the standardized residual
w = L_k^{-1}(x_std - mu_k), un-whitened x_std = mu_k + L_k w. Feasibility heads (off the
knobs) give witness viability p_surv and per-bunch surviving fractions T_d, T_w.

Reparameterized sampling => observables differentiable w.r.t. knobs (MBRL). Coupling math
reused from the photoinjector flow. See lab-notebook 2026-06-09_nf-two-bunch-surrogate.md.
"""
from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from . import LATENT_DIM
from .properties import compute_emittance_torch, inter_bunch, per_bunch

_LOG_2PI = math.log(2.0 * math.pi)


# ---- affine coupling (reused verbatim from the photoinjector flow) -----------
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, cond_dim, mlp_dim=64, reverse_mask=False):
        super().__init__()
        self.dim, self.d, self.reverse_mask = dim, dim // 2, reverse_mask
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, (dim - self.d) * 2))

    def _split(self, z):
        return (z[:, self.d:], z[:, :self.d]) if self.reverse_mask else (z[:, :self.d], z[:, self.d:])

    def _params(self, z1, cond):
        p = self.net(torch.cat([z1, cond], dim=1))
        scale, shift = p[:, :(self.dim - self.d)], p[:, (self.dim - self.d):]
        return torch.tanh(scale) * 0.5, shift

    def _join(self, a, b):
        return torch.cat([a, b], dim=1) if self.reverse_mask else torch.cat([b, a], dim=1)

    def forward(self, z, cond):
        z1, z2 = self._split(z)
        scale, shift = self._params(z1, cond)
        return self._join(z2 * torch.exp(scale) + shift, z1), scale.sum(dim=1)

    def inverse(self, z, cond):
        z1, z2 = self._split(z)
        scale, shift = self._params(z1, cond)
        return self._join((z2 - shift) * torch.exp(-scale), z1), -scale.sum(dim=1)


# ---- rational-quadratic spline (Durkan et al. 2019; linear tails) -----------
_MIN_W = _MIN_H = _MIN_D = 1e-3


def _rqs_1d(x, uw, uh, ud, inverse, B):
    """Monotonic RQ spline on [-B,B] with linear identity tails. Elementwise over any
    leading shape: x:(...,), uw/uh:(...,K), ud:(...,K-1). Returns (outputs, logabsdet).

    Computed densely (no boolean-mask indexing) and the tail is selected with torch.where,
    which is GPU-friendly (no host syncs / dynamic shapes). The in-spline arithmetic stays
    finite for tail inputs because bin_idx and theta are clamped, so the masked-out branch
    never produces NaN that torch.where could leak into the gradient."""
    K = uw.shape[-1]
    inside = (x >= -B) & (x <= B)

    widths = torch.softmax(uw, dim=-1)
    widths = _MIN_W + (1 - _MIN_W * K) * widths
    cw = torch.nn.functional.pad(torch.cumsum(widths, -1), (1, 0))
    cw = 2 * B * cw - B
    cw[..., 0], cw[..., -1] = -B, B
    widths = cw[..., 1:] - cw[..., :-1]

    heights = torch.softmax(uh, dim=-1)
    heights = _MIN_H + (1 - _MIN_H * K) * heights
    ch = torch.nn.functional.pad(torch.cumsum(heights, -1), (1, 0))
    ch = 2 * B * ch - B
    ch[..., 0], ch[..., -1] = -B, B
    heights = ch[..., 1:] - ch[..., :-1]

    deriv = _MIN_D + torch.nn.functional.softplus(ud)
    deriv = torch.nn.functional.pad(deriv, (1, 1), value=1.0)   # linear tails: boundary slope 1

    edges = ch if inverse else cw
    bin_idx = (torch.sum(x[..., None] >= edges, dim=-1) - 1).clamp(0, K - 1)
    g = lambda t: t.gather(-1, bin_idx[..., None]).squeeze(-1)
    cwk, wk = g(cw), g(widths)
    chk, hk = g(ch), g(heights)
    dk = g(deriv)
    dk1 = deriv.gather(-1, (bin_idx + 1)[..., None]).squeeze(-1)
    s = hk / wk

    if inverse:
        dy = x - chk
        a = dy * (dk1 + dk - 2 * s) + hk * (s - dk)
        b = hk * dk - dy * (dk1 + dk - 2 * s)
        c = -s * dy
        # clamp_min to a small positive (not 0): sqrt'(disc) = 1/(2*sqrt(disc)) -> inf as disc->0
        # (a particle landing on a knot), a finite-forward / NaN-backward trap. A tiny floor
        # keeps the gradient finite while barely perturbing the inverse value.
        disc = (b ** 2 - 4 * a * c).clamp_min(1e-10)
        theta = (2 * c / (-b - torch.sqrt(disc))).clamp(0.0, 1.0)
        o = theta * wk + cwk
        tt = theta * (1 - theta)
        denom = s + (dk1 + dk - 2 * s) * tt
        dnum = s ** 2 * (dk1 * theta ** 2 + 2 * s * tt + dk * (1 - theta) ** 2)
        ld = -(torch.log(dnum) - 2 * torch.log(denom))
    else:
        theta = ((x - cwk) / wk).clamp(0.0, 1.0)
        tt = theta * (1 - theta)
        denom = s + (dk1 + dk - 2 * s) * tt
        o = chk + hk * (s * theta ** 2 + dk * tt) / denom
        dnum = s ** 2 * (dk1 * theta ** 2 + 2 * s * tt + dk * (1 - theta) ** 2)
        ld = torch.log(dnum) - 2 * torch.log(denom)

    return torch.where(inside, o, x), torch.where(inside, ld, torch.zeros_like(ld))


class RQSCouplingLayer(nn.Module):
    """Conditional RQ-spline coupling layer; same interface as the affine layer."""

    def __init__(self, dim, cond_dim, mlp_dim=64, n_bins=12, tail_bound=5.0, reverse_mask=False):
        super().__init__()
        self.dim, self.d = dim, dim // 2
        self.n_t = dim - self.d
        self.K, self.B, self.reverse_mask = n_bins, tail_bound, reverse_mask
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, self.n_t * (3 * n_bins - 1)))

    def _split(self, z):
        return (z[:, self.d:], z[:, :self.d]) if self.reverse_mask else (z[:, :self.d], z[:, self.d:])

    def _join(self, a, b):
        return torch.cat([a, b], dim=1) if self.reverse_mask else torch.cat([b, a], dim=1)

    def _params(self, z1, cond):
        p = self.net(torch.cat([z1, cond], dim=1)).view(-1, self.n_t, 3 * self.K - 1)
        return p[..., :self.K], p[..., self.K:2 * self.K], p[..., 2 * self.K:]

    def _apply_spline(self, z1, z2, cond, inverse):
        # uw,uh: (M, n_t, K); ud: (M, n_t, K-1). Transform all n_t dims in one vectorized
        # _rqs_1d call (no Python loop over dims) -> (M, n_t).
        uw, uh, ud = self._params(z1, cond)
        out, ld = _rqs_1d(z2, uw, uh, ud, inverse, self.B)
        return out, ld.sum(dim=-1)

    def forward(self, z, cond):
        z1, z2 = self._split(z)
        out, lad = self._apply_spline(z1, z2, cond, inverse=False)
        return self._join(out, z1), lad

    def inverse(self, z, cond):
        z1, z2 = self._split(z)
        out, lad = self._apply_spline(z1, z2, cond, inverse=True)
        return self._join(out, z1), lad


class TwoBunchFlow(L.LightningModule):
    def __init__(
        self, condition_dim=8, latent_dim=LATENT_DIM, hidden_dim=128, n_layers=16,
        coupling="affine", n_bins=16, tail_bound=8.0,
        lr=1e-4, weight_decay=1e-5,
        w_cls=1.0, w_tr=1.0, w_emit=0.25, w_emit_z=0.0, w_emit_4d=0.0, w_emit_6d=0.0, w_cov=0.5, nll_dim_norm=6.0,
        n_aux_particles=512,
        bunches=(0, 1),  # which bunch density paths to train (0=drive,1=witness); (1,)=witness-only

        # per-bunch (de)standardization + knob bounds (from preprocess _norm.json)
        drive_mean=None, drive_std=None, witness_mean=None, witness_std=None,
        knob_low=None, knob_high=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # shared encoder takes (knobs, one-hot bunch index)
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        # whitening: per-bunch MLP h -> mu (6) + Cholesky params (6 diag + 15 off-diag).
        # Separate head per bunch (drive-round vs witness-thin no longer share one map) + an
        # MLP (not a single Linear) so it can track per-knob anisotropy.
        self._n_offdiag = latent_dim * (latent_dim - 1) // 2
        n_white = 2 * latent_dim + self._n_offdiag
        self.whiten_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, n_white))
            for _ in range(2)])
        if coupling == "rqs":
            self.flows = nn.ModuleList([
                RQSCouplingLayer(latent_dim, hidden_dim, n_bins=n_bins, tail_bound=tail_bound,
                                 reverse_mask=(i % 2 == 1)) for i in range(n_layers)])
        else:
            self.flows = nn.ModuleList([
                ConditionalCouplingLayer(latent_dim, hidden_dim, reverse_mask=(i % 2 == 1))
                for i in range(n_layers)])
        # feasibility heads (knobs only): [logit p_surv, logit T_d, logit T_w]
        self.feas_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3))

        tril = torch.tril_indices(latent_dim, latent_dim, offset=-1)
        self.register_buffer("_tril_r", tril[0])
        self.register_buffer("_tril_c", tril[1])

        def buf(v, default):
            return torch.tensor(default if v is None else v).float()
        self.register_buffer("drive_mean", buf(drive_mean, [0.0] * latent_dim))
        self.register_buffer("drive_std", buf(drive_std, [1.0] * latent_dim))
        self.register_buffer("witness_mean", buf(witness_mean, [0.0] * latent_dim))
        self.register_buffer("witness_std", buf(witness_std, [1.0] * latent_dim))
        self.register_buffer("knob_low", buf(knob_low, [0.0] * condition_dim))
        self.register_buffer("knob_high", buf(knob_high, [1.0] * condition_dim))

    # ---- conditioning + whitening -----------------------------------------
    def _encode(self, knobs, k):
        oh = torch.zeros(knobs.shape[0], 2, device=knobs.device, dtype=knobs.dtype)
        oh[:, k] = 1.0
        return self.encoder(torch.cat([knobs, oh], dim=1))

    def _whiten(self, h, k):
        out = self.whiten_heads[k](h)
        mu = out[:, :self.latent_dim]
        diag = F.softplus(out[:, self.latent_dim:2 * self.latent_dim]) + 1e-3
        # Bound off-diagonals so the Cholesky factor stays well-conditioned (NLL uses
        # L^{-1}; an ill-conditioned L would blow up w and the loss). |off| <= 5.
        off = 5.0 * torch.tanh(out[:, 2 * self.latent_dim:])
        Lm = torch.diag_embed(diag)
        Lm[:, self._tril_r, self._tril_c] = off
        return mu, Lm

    def _scaler(self, k):
        return (self.drive_mean, self.drive_std) if k == 0 else (self.witness_mean, self.witness_std)

    @staticmethod
    def _cov(parts):  # (M,P,6) -> (M,6,6) sample covariance
        c = parts - parts.mean(dim=1, keepdim=True)
        return torch.bmm(c.transpose(1, 2), c) / (parts.shape[1] - 1)

    # ---- flow primitives ---------------------------------------------------
    def _forward_enc(self, z, cond):
        ld = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for fl in self.flows:
            z, d = fl(z, cond)
            ld = ld + d
        return z, ld

    def _inverse_enc(self, x, cond):
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for fl in reversed(self.flows):
            x, d = fl.inverse(x, cond)
            ld = ld + d
        return x, ld

    # ---- NLL for one bunch -------------------------------------------------
    def bunch_nll(self, parts_std, knobs, k):
        """parts_std: (M,P,6) standardized; knobs: (M,8). Returns scalar mean NLL."""
        m, p = parts_std.shape[0], parts_std.shape[1]
        h = self._encode(knobs, k)
        mu, Lm = self._whiten(h, k)
        xc = parts_std - mu.unsqueeze(1)                                # (M,P,6)
        w = torch.linalg.solve_triangular(Lm, xc.transpose(1, 2), upper=False).transpose(1, 2)
        h_flat = h.unsqueeze(1).expand(m, p, -1).reshape(m * p, -1)
        z, ld_flow = self._inverse_enc(w.reshape(m * p, self.latent_dim), h_flat)
        logpz = (-0.5 * z ** 2 - 0.5 * _LOG_2PI).sum(dim=1)
        ld_white = -torch.log(torch.diagonal(Lm, dim1=-2, dim2=-1)).sum(dim=1)  # (M,)
        logp = logpz + ld_flow + ld_white.repeat_interleave(p)
        return -logp.mean() / self.hparams.nll_dim_norm

    # ---- sampling (reparameterized, differentiable wrt knobs) --------------
    def sample_bunch(self, knobs, k, n, physical=True):
        """(B,n,6) cloud for bunch k. Differentiable w.r.t. knobs."""
        b = knobs.shape[0]
        h = self._encode(knobs, k)
        h_flat = h.unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
        z = torch.randn(b * n, self.latent_dim, device=knobs.device, dtype=knobs.dtype)
        w, _ = self._forward_enc(z, h_flat)
        w = w.reshape(b, n, self.latent_dim)
        mu, Lm = self._whiten(h, k)
        x_std = mu.unsqueeze(1) + torch.einsum("bij,bnj->bni", Lm, w)
        if not physical:
            return x_std
        mean, std = self._scaler(k)
        return x_std * std + mean

    def feasibility(self, knobs):
        logits = self.feas_head(knobs)
        s = torch.sigmoid(logits)
        return s[:, 0], s[:, 1], s[:, 2]  # p_surv(witness), T_drive, T_witness

    # ---- MBRL-facing observables ------------------------------------------
    def observables(self, knobs_norm, n=None):
        """Differentiable dict of per-bunch + inter-bunch + feasibility observables.
        knobs_norm: (B,8) in [0,1]."""
        n = n or self.hparams.n_aux_particles
        drive = self.sample_bunch(knobs_norm, 0, n)
        witness = self.sample_bunch(knobs_norm, 1, n)
        p_surv, t_d, t_w = self.feasibility(knobs_norm)
        out = {}
        for name, parts in (("drive", drive), ("witness", witness)):
            for kk, vv in per_bunch(parts).items():
                out[f"{name}_{kk}"] = vv
        out.update(inter_bunch(drive, witness))
        out.update(p_surv=p_surv, T_drive=t_d, T_witness=t_w)
        return out

    # ---- training ----------------------------------------------------------
    def _step(self, batch, prefix):
        knobs = batch["knobs"]
        dd, wd = batch["drive_density"], batch["witness_density"]
        dv, wv = batch["drive_present"], batch["witness_viable"]
        hp = self.hparams
        tb = hp.bunches  # bunch density paths to train (for the witness-only ablation)
        logs, loss, n_skip = {}, knobs.new_zeros(()), 0

        def add(term, w, name):
            """Accumulate w*term, skipping (and counting) a non-finite term so a rare NaN/Inf
            (e.g. the RQS inverse at a knot for one particle) can't poison the shared weights."""
            nonlocal loss, n_skip
            if torch.isfinite(term):
                loss = loss + w * term
                logs[f"{prefix}_{name}"] = term
            else:
                n_skip += 1

        if 0 in tb and dd.any():
            add(self.bunch_nll(batch["drive"][dd], knobs[dd], 0), 1.0, "nll_drive")
        if 1 in tb and wd.any():
            add(self.bunch_nll(batch["witness"][wd], knobs[wd], 1), 1.0, "nll_witness")

        # feasibility heads -- BCE-with-logits is numerically stable and never asserts on an
        # out-of-[0,1] input (a NaN p_surv from upstream divergence used to hard-crash here).
        feas_logits = self.feas_head(knobs)
        t_d, t_w = torch.sigmoid(feas_logits[:, 1]), torch.sigmoid(feas_logits[:, 2])
        add(F.binary_cross_entropy_with_logits(feas_logits[:, 0], wv.float()), hp.w_cls, "bce_surv")
        if dv.any():
            add(F.mse_loss(t_d[dv], batch["drive_frac"][dv]), hp.w_tr, "mse_Td")
        if wv.any():
            add(F.mse_loss(t_w[wv], batch["witness_frac"][wv]), hp.w_tr, "mse_Tw")

        # per-bunch moment matching on the sampled cloud (standardized frame), one shared draw:
        # 2D emittances (incl. z_delta = sqrt det cov_zpz, the LPS thinness lever), 4D/6D dets,
        # and the relative 6x6 covariance -- the NLL alone tolerates a too-round Sigma_k.
        any_emit = hp.w_emit > 0 or hp.w_emit_z > 0 or hp.w_emit_4d > 0 or hp.w_emit_6d > 0
        if any_emit or hp.w_cov > 0:
            eps = 1e-9
            for k, mask, key in ((0, dd, "drive"), (1, wd, "witness")):
                if k not in tb or not mask.any():
                    continue
                true = batch[key][mask]
                pred = self.sample_bunch(knobs[mask], k, true.shape[1], physical=False)
                if any_emit:
                    ep, et = compute_emittance_torch(pred), compute_emittance_torch(true)
                    ldiff = lambda q: torch.abs(torch.log10(ep[q] + eps) - torch.log10(et[q] + eps)).mean()
                    if hp.w_emit > 0:             # transverse 2D (x-px, y-py)
                        add((ldiff("x_xp") + ldiff("y_yp")) / 2, hp.w_emit, f"emit_{key}")
                    if hp.w_emit_z > 0:          # longitudinal/LPS 2D (z-pz), own (undiluted) weight
                        add(ldiff("z_delta"), hp.w_emit_z, f"emitz_{key}")
                    if hp.w_emit_4d > 0:
                        add(ldiff("fourd"), hp.w_emit_4d, f"emit4d_{key}")
                    if hp.w_emit_6d > 0:
                        add(ldiff("sixd"), hp.w_emit_6d, f"emit6d_{key}")
                if hp.w_cov > 0:
                    cp, ct = self._cov(pred), self._cov(true)
                    lc = (2 * torch.abs(cp - ct) / (torch.abs(cp) + torch.abs(ct) + 1e-6)).mean()
                    add(lc, hp.w_cov, f"cov_{key}")

        logs[f"{prefix}_loss"] = loss
        if n_skip:
            logs[f"{prefix}_nonfinite_skipped"] = float(n_skip)
        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=(prefix == "val"), batch_size=knobs.shape[0])
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
