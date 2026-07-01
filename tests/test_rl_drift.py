"""Physical RF-drift mapping + raw-loss (stricter) survival target.

Covers the DR additions: `rf_drift_std_vector` physical->normalized mapping, the env applying a
per-knob vector as a per-episode latent offset (zero on the sextupole knobs), and the raw-loss
survival hinge (hinge_at == 1.0 => penalty == 1 - T) built from surv_T_min/surv_margin config.
"""
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.rl.diff_env import (  # noqa: E402
    N_KNOB, RF_DRIFT_IDX, TwoBunchFlowEnv, rf_drift_std_vector,
)
from twobunch_s2e_rl.rl.reward import (  # noqa: E402
    EMIT_KEYS, SURV_KEYS, _TRANSFORM, RewardTerm, build_twobunch_reward_spec,
)
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

_KEYS = ("bunch_spacing", "energy_difference", "transverse_offset",
         "angular_misalignment", "T_drive", "T_witness") + EMIT_KEYS
_SEXT_IDX = tuple(i for i in range(N_KNOB) if i not in RF_DRIFT_IDX)   # S1/S2/S3 ELkG


def _spec(**kw):
    norms = {}
    for k in _KEYS:
        tr = _TRANSFORM.get(k, "identity")
        mean = -5.0 if tr == "log10" else (2e-4 if k == "bunch_spacing" else 0.5)
        norms[k] = {"transform": tr, "mean": mean, "std": 0.5}
    floors = {k: 5e-6 for k in EMIT_KEYS}
    return build_twobunch_reward_spec(norms, floors, **kw)


def _env(num_envs=4, **kw):
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=128).eval()
    defaults = dict(device="cpu", episode_length=8, n_particles=128, reward_spec=_spec())
    defaults.update(kw)
    return TwoBunchFlowEnv(num_envs, flow=flow, **defaults)


# ---- rf_drift_std_vector: physical -> normalized ------------------------------

def test_drift_vector_values():
    v = rf_drift_std_vector(phase_drift_deg=0.5, amp_drift_frac=0.005)
    assert v.shape == (N_KNOB,)
    # L1Phase 0.5/5.0, L2Phase 0.5/6.0 (box widened low->-40), energies 0.5% of nominal gain / box width
    assert v[0].item() == pytest.approx(0.100, abs=1e-3)   # L1PhaseSet
    assert v[1].item() == pytest.approx(0.0833, abs=1e-3)  # L2PhaseSet
    assert v[2].item() == pytest.approx(0.250, abs=1e-3)   # L1EnergyOffset (4.2e6 box)
    assert v[3].item() == pytest.approx(0.250, abs=1e-2)   # L2EnergyOffset (83.4e6 box ~ 0.2497)
    assert v[4].item() == pytest.approx(0.250, abs=1e-3)   # L3EnergyOffset (110e6 box)
    # sextupoles get no RF drift
    for j in _SEXT_IDX:
        assert v[j].item() == 0.0


def test_drift_vector_scales_with_specs():
    v1 = rf_drift_std_vector(0.5, 0.005)
    v2 = rf_drift_std_vector(1.0, 0.010)
    assert torch.allclose(v2, 2.0 * v1)


# ---- env applies the vector as a per-episode latent offset --------------------

def test_env_accepts_drift_vector_zero_on_sextupoles():
    v = rf_drift_std_vector()
    env = _env(rf_drift_std=v)
    # non-RF (sextupole) columns must be exactly zero; RF columns must carry drift
    assert torch.count_nonzero(env._drift[:, list(_SEXT_IDX)]) == 0
    assert env._drift[:, list(RF_DRIFT_IDX)].abs().sum() > 0


def test_drift_constant_within_episode():
    """Latent drift is sampled once per episode (at reset), not per step."""
    env = _env(num_envs=3, episode_length=8, rf_drift_std=rf_drift_std_vector())
    d0 = env._drift.clone()
    for _ in range(3):                       # stay inside the episode (no auto-reset)
        env.step(torch.zeros(3, 8))
    assert torch.equal(env._drift, d0)


def test_goal_constant_within_episode_resampled_at_reset():
    """The per-episode target spacing is sampled once at reset and held across the episode; the
    episode-end auto-reset draws a fresh goal."""
    spec = _spec(spacing_goal_key="spacing_goal")
    env = _env(num_envs=3, episode_length=4, reward_spec=spec,
               spacing_goal_lo=1e-4, spacing_goal_hi=3e-4)
    g0 = env._spacing_goal.clone()
    for _ in range(3):                       # steps 1-3: still inside the episode
        env.step(torch.zeros(3, 8))
    assert torch.equal(env._spacing_goal, g0)
    _, _, done, _ = env.step(torch.zeros(3, 8))   # step 4 -> done -> auto-reset
    assert done.all()
    assert (env._spacing_goal >= 1e-4).all() and (env._spacing_goal <= 3e-4).all()


def test_drift_vector_wrong_length_raises():
    with pytest.raises(ValueError):
        _env(rf_drift_std=torch.zeros(5))


# ---- raw-loss (stricter) survival hinge ---------------------------------------

def test_hinge_at_one_is_raw_loss():
    term = RewardTerm("T_witness", "hinge", hinge_at=1.0, weight=1.0)
    T = torch.tensor([0.50, 0.90, 0.98, 1.00])
    pen = term.penalty({"T_witness": T})
    assert torch.allclose(pen, 1.0 - T)      # penalty == particle loss fraction


def test_strict_config_builds_rawloss_hinge():
    spec = _spec(surv_T_min=0.98, surv_margin=0.02, w_surv=4.0)
    hinges = [t for t in spec.terms if t.mode == "hinge" and t.key in SURV_KEYS]
    assert len(hinges) == len(SURV_KEYS)
    for t in hinges:
        assert t.hinge_at == pytest.approx(1.0)   # surv_T_min + surv_margin
        assert t.weight == pytest.approx(4.0)
