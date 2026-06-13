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


class TwoBunchFlow(L.LightningModule):
    def __init__(
        self, condition_dim=8, latent_dim=LATENT_DIM, hidden_dim=128, n_layers=16,
        lr=1e-4, weight_decay=1e-5,
        w_cls=1.0, w_tr=1.0, w_emit=0.25, nll_dim_norm=6.0,
        n_aux_particles=512,
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
        # whitening head: h -> mu (6) + Cholesky params (6 diag + 15 off-diag)
        self._n_offdiag = latent_dim * (latent_dim - 1) // 2
        self.whiten_head = nn.Linear(hidden_dim, latent_dim + latent_dim + self._n_offdiag)
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

    def _whiten(self, h):
        out = self.whiten_head(h)
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
        mu, Lm = self._whiten(h)
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
        mu, Lm = self._whiten(h)
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
        logs, loss = {}, knobs.new_zeros(())

        if dd.any():
            nd = self.bunch_nll(batch["drive"][dd], knobs[dd], 0)
            loss = loss + nd
            logs[f"{prefix}_nll_drive"] = nd
        if wd.any():
            nw = self.bunch_nll(batch["witness"][wd], knobs[wd], 1)
            loss = loss + nw
            logs[f"{prefix}_nll_witness"] = nw

        # feasibility heads
        p_surv, t_d, t_w = self.feasibility(knobs)
        bce = F.binary_cross_entropy(p_surv.clamp(1e-6, 1 - 1e-6), wv.float())
        loss = loss + hp.w_cls * bce
        logs[f"{prefix}_bce_surv"] = bce
        if dv.any():
            md = F.mse_loss(t_d[dv], batch["drive_frac"][dv])
            loss = loss + hp.w_tr * md
            logs[f"{prefix}_mse_Td"] = md
        if wv.any():
            mw = F.mse_loss(t_w[wv], batch["witness_frac"][wv])
            loss = loss + hp.w_tr * mw
            logs[f"{prefix}_mse_Tw"] = mw

        # light per-bunch emittance matching: absolute log10 difference (scale-invariant
        # and bounded -- a *relative* error here explodes, since per-dim standardization
        # makes the x-px cloud near-degenerate so true emittance -> ~0). Frame-invariant.
        if hp.w_emit > 0:
            eps = 1e-9
            for k, mask, key in ((0, dd, "drive"), (1, wd, "witness")):
                if not mask.any():
                    continue
                pred = self.sample_bunch(knobs[mask], k, batch[key].shape[1], physical=False)
                ep = compute_emittance_torch(pred)
                et = compute_emittance_torch(batch[key][mask])
                le = (torch.abs(torch.log10(ep["x_xp"] + eps) - torch.log10(et["x_xp"] + eps)).mean()
                      + torch.abs(torch.log10(ep["y_yp"] + eps) - torch.log10(et["y_yp"] + eps)).mean()) / 2
                loss = loss + hp.w_emit * le
                logs[f"{prefix}_emit_{key}"] = le

        logs[f"{prefix}_loss"] = loss
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
