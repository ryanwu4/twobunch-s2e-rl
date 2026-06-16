"""Reward unit tests for twobunch_s2e_rl.rl.reward.

Synthetic (no checkpoint/data); fast on CPU. Skips if torch absent.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.rl.reward import (  # noqa: E402
    EMIT_KEYS, KNOBS_KEY, RewardTerm, build_twobunch_reward_spec, compute_metric_norms,
    derive_floors_from_campaign, load_norms, reward_spec_from_campaign,
)
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

_ALL_KEYS = ("bunch_spacing", "energy_difference", "transverse_offset",
             "angular_misalignment", "T_drive", "T_witness") + EMIT_KEYS


def _norms():
    from twobunch_s2e_rl.rl.reward import _TRANSFORM
    norms = {}
    for k in _ALL_KEYS:
        tr = _TRANSFORM.get(k, "identity")
        mean = -4.4 if tr == "log10" else (2e-4 if k == "bunch_spacing" else 0.5)
        norms[k] = {"transform": tr, "mean": mean, "std": 0.3}
    return norms


def _floors():
    return {k: 1.4e-5 for k in EMIT_KEYS}   # log10 ~ -4.854 < mean -4.4


def _model(n_layers=4, hidden=32):
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=8, n_layers=n_layers, hidden_dim=hidden,
                        n_aux_particles=128).eval()


def _obs(m, knobs, n=256):
    """observables() + the env-injected knobs (needed by the trust-region term)."""
    o = m.observables(knobs, n=n)
    o[KNOBS_KEY] = knobs
    return o


# ---- term-shape correctness -------------------------------------------------

def test_target_term_zero_at_target():
    t = RewardTerm("bunch_spacing", "target", "identity", mean=0.0, std=0.5, target=2e-4)
    obs = {"bunch_spacing": torch.tensor([2e-4, 3e-4, 1e-4])}
    pen = t.penalty(obs)
    assert torch.isclose(pen[0], torch.tensor(0.0), atol=1e-9)
    assert (pen[1:] > 0).all() and torch.isclose(pen[1], pen[2], atol=1e-9)


def _emit_term(**kw):
    d = dict(key="witness_norm_emit_x", mode="minimize_floor", transform="log10",
             mean=-4.4, std=0.3, floor=1.4e-5)
    d.update(kw)
    return RewardTerm(**d)


def test_minimize_floor_range_normalized_to_minus_one_at_floor():
    """Fix #2: range-norm so the term = -1 exactly at the floor, 0 at the campaign mean."""
    t = _emit_term()
    at_floor = t.penalty({"witness_norm_emit_x": torch.tensor([1.4e-5])})
    at_mean = t.penalty({"witness_norm_emit_x": torch.tensor([10 ** -4.4])})
    assert torch.isclose(at_floor[0], torch.tensor(-1.0), atol=1e-4)
    assert torch.isclose(at_mean[0], torch.tensor(0.0), atol=1e-4)


def test_minimize_floor_sub_floor_is_penalized():
    """Fix #3: below the floor the penalty RISES (barrier), with nonzero gradient -- it no
    longer goes flat, so the policy can't drift into the sub-floor extrapolation regime."""
    eps = torch.tensor([1e-6, 1e-7], requires_grad=True)   # well below floor
    t = _emit_term(below_weight=1.0, gate_key=None)
    pen = t.penalty({"witness_norm_emit_x": eps})
    assert (pen > 0).all()                                  # net penalty below floor
    assert pen[1] > pen[0]                                  # deeper sub-floor -> larger penalty
    pen.sum().backward()
    assert eps.grad.abs().sum() > 0                         # gradient pushes back up


def test_survival_gating_asymmetric():
    """Fix #1: good (negative) emittance reward is gated by T; bad (positive) emittance is NOT
    gated (so the agent can't scrape a bunch to hide its bad emittance)."""
    t = _emit_term(gate_key="T_witness", gate_low=0.7, gate_high=0.9)
    good = torch.tensor([2e-5])           # below mean -> reward
    bad = torch.tensor([1e-3])            # above mean -> penalty
    hiT = {"witness_norm_emit_x": good, "T_witness": torch.tensor([0.95])}
    loT = {"witness_norm_emit_x": good, "T_witness": torch.tensor([0.6])}
    badloT = {"witness_norm_emit_x": bad, "T_witness": torch.tensor([0.6])}
    p_hi = t.penalty(hiT); p_lo = t.penalty(loT); p_badlo = t.penalty(badloT)
    assert p_hi[0] < 0                                      # surviving -> full emittance reward
    assert torch.isclose(p_lo[0], torch.tensor(0.0), atol=1e-6)   # scraped -> reward gated away
    assert p_badlo[0] > 0                                   # bad emittance still penalized when scraped


def test_hinge_survival_one_sided():
    t = RewardTerm("T_witness", "hinge", weight=1.0, hinge_at=0.95)
    T = torch.tensor([0.97, 0.95, 0.7], requires_grad=True)
    pen = t.penalty({"T_witness": T})
    assert torch.isclose(pen[0], torch.tensor(0.0)) and torch.isclose(pen[1], torch.tensor(0.0))
    assert torch.isclose(pen[2], torch.tensor(0.25), atol=1e-6)
    pen.sum().backward()
    assert torch.isclose(T.grad[0], torch.tensor(0.0)) and T.grad[2] < 0


def test_boundary_term():
    """Fix #3: trust-region penalty ~0 in the interior, ramps to 1 at a box edge."""
    t = RewardTerm(KNOBS_KEY, "boundary", margin=0.05)
    interior = torch.full((1, 8), 0.5)
    one_railed = interior.clone(); one_railed[0, 3] = 0.0     # one knob at the edge
    near = interior.clone(); near[0, 0] = 0.02                # within margin
    assert torch.isclose(t.penalty({KNOBS_KEY: interior})[0], torch.tensor(0.0))
    assert torch.isclose(t.penalty({KNOBS_KEY: one_railed})[0], torch.tensor(1.0 / 8), atol=1e-6)
    assert t.penalty({KNOBS_KEY: near})[0] > 0


# ---- spec assembly + reward -------------------------------------------------

def test_reward_finite_and_differentiable():
    m = _model()
    spec = build_twobunch_reward_spec(_norms(), _floors())
    knobs = torch.rand(4, 8, requires_grad=True)
    reward, achieved = spec(_obs(m, knobs))
    assert reward.shape == (4,) and torch.isfinite(reward).all()
    reward.sum().backward()
    assert knobs.grad is not None and torch.isfinite(knobs.grad).all()
    assert knobs.grad.abs().sum() > 0
    for k in ("bunch_spacing", "T_witness", "witness_norm_emit_x", "reward"):
        assert k in achieved


def test_builder_has_survival_margin_and_ood():
    spec = build_twobunch_reward_spec(_norms(), _floors(), surv_T_min=0.9, surv_margin=0.05,
                                      w_ood=0.5)
    hinges = {t.key: t.hinge_at for t in spec.terms if t.mode == "hinge"}
    assert hinges["T_drive"] == pytest.approx(0.95) and hinges["T_witness"] == pytest.approx(0.95)
    assert any(t.mode == "boundary" for t in spec.terms)
    # emittance terms are survival-gated by their own bunch's T head
    gates = {t.key: t.gate_key for t in spec.terms if t.mode == "minimize_floor"}
    assert gates["drive_norm_emit_x"] == "T_drive" and gates["witness_norm_emit_y"] == "T_witness"


def test_zero_weight_ood_and_collinearity_inert():
    norms, floors = _norms(), _floors()
    base = build_twobunch_reward_spec(norms, floors, w_ood=0.0, w_collinearity=0.0)
    assert not any(t.mode in ("boundary",) for t in base.terms)
    assert len(build_twobunch_reward_spec(norms, floors, w_ood=0.0, w_collinearity=2.0).terms) \
        == len(base.terms) + 2


def test_witness_weight_override():
    spec = build_twobunch_reward_spec(_norms(), _floors(), w_emit=1.0, w_emit_witness=0.3)
    wt = {t.key: t.weight for t in spec.terms if t.key in EMIT_KEYS}
    assert wt["drive_norm_emit_x"] == 1.0 and wt["witness_norm_emit_x"] == 0.3


def test_obs_scaled_shape_and_finite():
    spec = build_twobunch_reward_spec(_norms(), _floors())
    obs = _model().observables(torch.rand(5, 8), n=128)
    v = spec.obs_scaled(obs)
    assert v.shape == (5, spec.n_obs_extra) == (5, 7) and torch.isfinite(v).all()


def test_reward_equals_neg_weighted_penalties():
    spec = build_twobunch_reward_spec(_norms(), _floors())
    m = _model()
    obs = _obs(m, torch.rand(2, 8), n=128)
    reward, _ = spec(obs)
    manual = -sum(t.weight * t.penalty(obs) for t in spec.terms)
    assert torch.allclose(reward, manual, atol=1e-6)


def test_missing_metric_raises_clear_error():
    norms = _norms(); del norms["bunch_spacing"]
    with pytest.raises(KeyError):
        build_twobunch_reward_spec(norms, _floors())


# ---- campaign-data norms/floors + cache provenance --------------------------

def _make_h5(path, N=48, P=128):
    import h5py
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h:
        h.create_dataset("knobs", data=rng.random((N, 8)).astype(np.float32))
        scale = np.array([1e-4, 1e-4, 5e-5, 1e6, 1e6, 1e10], np.float32)
        h.create_dataset("drive_parts", data=(rng.standard_normal((N, P, 6)) * scale).astype(np.float32))
        h.create_dataset("witness_parts", data=(rng.standard_normal((N, P, 6)) * scale).astype(np.float32))
        for k in ("drive_present", "witness_viable", "drive_density", "witness_density"):
            h.create_dataset(k, data=np.ones(N, bool))
        h.create_dataset("drive_frac", data=rng.uniform(0.85, 1.0, N).astype(np.float32))
        h.create_dataset("witness_frac", data=rng.uniform(0.4, 1.0, N).astype(np.float32))


def test_compute_metric_norms_and_floors(tmp_path):
    p = str(tmp_path / "campaign.h5")
    _make_h5(p)
    norms = compute_metric_norms(p, chunk=16)
    for k in EMIT_KEYS + ("bunch_spacing", "T_drive", "T_witness"):
        assert k in norms and math.isfinite(norms[k]["mean"]) and norms[k]["std"] > 0
    assert norms["witness_norm_emit_x"]["transform"] == "log10"
    floors = derive_floors_from_campaign(p, pct=10.0, chunk=16)
    assert set(floors) == set(EMIT_KEYS) and all(v > 0 for v in floors.values())


def test_reward_spec_from_campaign_cache_provenance(tmp_path):
    """Fix #5: the cache is invalidated when floor_pct changes -- floors must differ."""
    p = str(tmp_path / "campaign.h5")
    _make_h5(p)
    cache = str(tmp_path / "norms.json")
    s10 = reward_spec_from_campaign(p, cache_json=cache, floor_pct=10.0)
    prov1, _, floors10 = load_norms(cache)
    assert prov1["floor_pct"] == 10.0
    # re-run with a DIFFERENT floor_pct against the same cache path -> must recompute
    s50 = reward_spec_from_campaign(p, cache_json=cache, floor_pct=50.0)
    prov2, _, floors50 = load_norms(cache)
    assert prov2["floor_pct"] == 50.0
    f10 = {t.key: t.floor for t in s10.terms if t.mode == "minimize_floor"}
    f50 = {t.key: t.floor for t in s50.terms if t.mode == "minimize_floor"}
    assert f50["witness_norm_emit_x"] > f10["witness_norm_emit_x"]   # p50 > p10
    # a matched re-run reuses the cache (same provenance) and builds identically
    s50b = reward_spec_from_campaign(p, cache_json=cache, floor_pct=50.0)
    assert len(s50b.terms) == len(s50.terms)
