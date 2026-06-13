"""Surrogate unit tests: shapes, flow invertibility, NLL, and the MBRL gradient contract.

Synthetic (no checkpoint/data needed); fast on CPU. Skips if torch is absent.
"""
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402


def _model(n_layers=4, hidden=32):
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=8, n_layers=n_layers, hidden_dim=hidden,
                        n_aux_particles=128).eval()


def test_coupling_invertible():
    m = _model()
    w = torch.randn(20, 6)
    h = torch.randn(20, m.hparams.hidden_dim)
    z, ld_inv = m._inverse_enc(w, h)
    w2, ld_fwd = m._forward_enc(z, h)
    assert torch.allclose(w, w2, atol=1e-4), "flow not invertible"
    assert torch.allclose(ld_inv + ld_fwd, torch.zeros_like(ld_inv), atol=1e-4)


def test_sample_shapes_and_observables():
    m = _model()
    knobs = torch.rand(5, 8)
    for k in (0, 1):
        assert m.sample_bunch(knobs, k, 64).shape == (5, 64, 6)
    obs = m.observables(knobs, n=128)
    for key in ("drive_norm_emit_x", "witness_norm_emit_x", "bunch_spacing",
                "energy_difference", "transverse_offset", "angular_misalignment",
                "p_surv", "T_drive", "T_witness"):
        assert obs[key].shape == (5,), key
        assert torch.isfinite(obs[key]).all(), key


def test_bunch_nll_finite_and_differentiable():
    m = _model()
    parts = torch.randn(6, 256, 6)
    knobs = torch.rand(6, 8)
    nll = m.bunch_nll(parts, knobs, 1)
    assert torch.isfinite(nll)
    nll.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_gradient_flows_to_knobs():
    """MBRL contract: observables must be differentiable w.r.t. the knobs."""
    m = _model()
    knobs = torch.rand(4, 8, requires_grad=True)
    obs = m.observables(knobs, n=256)
    loss = (obs["witness_norm_emit_x"].sum() + obs["transverse_offset"].sum()
            + obs["p_surv"].sum())
    loss.backward()
    assert knobs.grad is not None
    assert torch.isfinite(knobs.grad).all()
    assert knobs.grad.abs().sum() > 0, "no gradient reached the knobs"


def test_whitening_lower_triangular_positive_diag():
    m = _model()
    h = torch.randn(7, m.hparams.hidden_dim)
    mu, Lm = m._whiten(h)
    assert mu.shape == (7, 6) and Lm.shape == (7, 6, 6)
    diag = torch.diagonal(Lm, dim1=-2, dim2=-1)
    assert (diag > 0).all(), "Cholesky diagonal must be positive"
    assert torch.allclose(torch.triu(Lm, diagonal=1), torch.zeros_like(Lm), atol=1e-6)


def test_bunch_nll_decreases_on_overfit():
    """The flow+whitening must actually learn (finite NLL alone isn't enough)."""
    torch.manual_seed(0)
    m = _model(n_layers=6, hidden=64)
    knobs = torch.rand(1, 8)
    A = torch.randn(6, 6) * 0.3
    parts = torch.randn(1, 512, 6) @ A.T + torch.tensor([0.5, -0.3, 0.2, 0.0, 0.1, -0.2])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    init = m.bunch_nll(parts, knobs, 0).item()
    for _ in range(200):
        opt.zero_grad()
        loss = m.bunch_nll(parts, knobs, 0)
        loss.backward()
        opt.step()
    final = m.bunch_nll(parts, knobs, 0).item()
    import math
    assert math.isfinite(final) and final < init - 0.5, (init, final)


def test_feasibility_learns_point_mass():
    """Witness-viability classifier must separate the ~31% destroyed mode."""
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(0)
    m = _model()
    knobs = torch.rand(400, 8)
    viable = (knobs[:, 1] + 0.3 * knobs[:, 4] > 0.7).float()
    opt = torch.optim.Adam(m.feas_head.parameters(), lr=5e-3)
    for _ in range(300):
        opt.zero_grad()
        p_surv, _, _ = m.feasibility(knobs)
        F.binary_cross_entropy(p_surv.clamp(1e-6, 1 - 1e-6), viable).backward()
        opt.step()
    p_surv, _, _ = m.feasibility(knobs)
    assert roc_auc_score(viable.numpy(), p_surv.detach().numpy()) > 0.9


def _batch(B=4, P=128, drive_density=True, witness_density=True):
    f = lambda v: torch.full((B,), v, dtype=torch.bool)
    return {
        "knobs": torch.rand(B, 8),
        "drive": torch.randn(B, P, 6), "witness": torch.randn(B, P, 6),
        "drive_present": f(drive_density), "witness_viable": f(witness_density),
        "drive_density": f(drive_density), "witness_density": f(witness_density),
        "drive_frac": torch.rand(B), "witness_frac": torch.rand(B) * witness_density,
    }


def test_step_handles_empty_witness_and_empty_batch():
    m = _model()
    m.log_dict = lambda *a, **k: None  # no Trainer attached in unit test
    loss_w = m._step(_batch(witness_density=False), "train")
    assert torch.isfinite(loss_w) and loss_w.requires_grad
    # fully empty density (only the always-on BCE term contributes)
    loss_e = m._step(_batch(drive_density=False, witness_density=False), "train")
    assert torch.isfinite(loss_e)


def test_state_dict_roundtrip_buffers_and_sampling():
    """Buffers (de-norm scalers, bounds, tril idx) must travel in the state_dict, and
    sampling must actually use them (deterministic + identical after reload)."""
    a = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32,
                     drive_mean=[1, 2, 3, 4, 5, 6], drive_std=[1, 1, 1, 1, 1, 1],
                     witness_mean=[0.1] * 6, witness_std=[2.0] * 6,
                     knob_low=[0.0] * 8, knob_high=[1.0] * 8).eval()
    b = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32).eval()
    b.load_state_dict(a.state_dict())
    for name in ("drive_mean", "witness_std", "knob_high", "_tril_r"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    knobs = torch.rand(3, 8)
    torch.manual_seed(7); s1 = a.sample_bunch(knobs, 1, 16)
    torch.manual_seed(7); s2 = b.sample_bunch(knobs, 1, 16)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_slopes_match_pxpz():
    from twobunch_s2e_rl.surrogate.properties import _slopes
    parts = torch.randn(3, 100, 6)
    parts[..., 5] = 1e10  # pz >> clamp floor
    xp, yp = _slopes(parts)
    assert torch.allclose(xp, (parts[..., 3] / parts[..., 5]).mean(1), atol=1e-9)


def test_subsample_no_replacement_at_boundary():
    import numpy as np
    from twobunch_s2e_rl.surrogate.preprocess import _subsample
    coords = np.arange(20 * 6, dtype=np.float32).reshape(20, 6)
    out = _subsample(coords, 20, np.random.default_rng(0))
    assert out.shape == (20, 6)
    assert len(np.unique(out[:, 0])) == 20  # a permutation, no repeats
