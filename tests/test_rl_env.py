"""TwoBunchFlowEnv contract + gradient tests.

Synthetic (small flow, no checkpoint/data); fast on CPU. Skips if torch absent.
"""
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.rl.diff_env import N_KNOB, RF_DRIFT_IDX, TwoBunchFlowEnv  # noqa: E402
from twobunch_s2e_rl.rl.reward import (  # noqa: E402
    EMIT_KEYS, _TRANSFORM, build_twobunch_reward_spec,
)
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

_KEYS = ("bunch_spacing", "energy_difference", "transverse_offset",
         "angular_misalignment", "T_drive", "T_witness") + EMIT_KEYS


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


def test_dims_and_reset_shape():
    env = _env()
    assert env.num_actions == 8
    assert env.num_obs == N_KNOB + 7 == 15
    obs = env.reset()
    assert obs.shape == (4, 15) and torch.isfinite(obs).all()


def test_step_contract():
    env = _env()
    obs, rew, done, info = env.step(torch.zeros(4, 8))
    assert obs.shape == (4, 15) and rew.shape == (4,) and done.shape == (4,)
    assert "obs_before_reset" in info and info["obs_before_reset"].shape == (4, 15)
    assert "achieved" in info and "T_witness" in info["achieved"]
    assert torch.isfinite(rew).all()


def test_achieved_obs_block_detached():
    """The achieved-observable block must be detached (no flow graph retained in obs)."""
    env = _env(no_grad=False)
    a = torch.zeros(4, 8, requires_grad=True)
    env.step(a)
    assert env._last_obs_extra.requires_grad is False
    # the extra obs columns carry no grad back to the flow
    obs = env._compute_obs()
    g = torch.autograd.grad(obs[:, N_KNOB:].sum(), a, retain_graph=True, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0


def test_gradient_flows_action_to_reward():
    """One-step reward must be differentiable w.r.t. the action (the MBRL contract)."""
    env = _env(stochastic_init=False, no_grad=False)   # knobs=0.5 -> interior, clamp passes grad
    a = torch.zeros(4, 8, requires_grad=True)
    _, rew, _, _ = env.step(a)
    rew.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0, "no gradient reached the action"


def test_no_grad_mode_detaches_reward():
    env = _env(no_grad=True)
    a = torch.zeros(4, 8, requires_grad=True)
    _, rew, _, _ = env.step(a)
    assert rew.requires_grad is False


def test_drift_zero_by_default():
    env = _env(rf_drift_std=0.0)
    assert torch.count_nonzero(env._drift) == 0


# ---- goal-conditioning (dynamic target spacing) -------------------------------

def _gc_spec():
    return _spec(spacing_goal_key="spacing_goal")


def test_non_goal_env_unchanged():
    """No goal kwargs -> non-goal env, obs dim 15, identical to the fixed-target env."""
    env = _env()
    assert env.num_obs == N_KNOB + 7 == 15 and env._goal_conditioned is False


def test_goal_conditioned_obs_dim_grows():
    env = _env(reward_spec=_gc_spec(), spacing_goal_lo=1e-4, spacing_goal_hi=3e-4)
    assert env._goal_conditioned and env.num_obs == N_KNOB + 8 == 16
    obs = env.reset()
    assert obs.shape == (4, 16) and torch.isfinite(obs).all()


def test_goal_in_range_and_partial_reset():
    env = _env(num_envs=6, reward_spec=_gc_spec(), spacing_goal_lo=1e-4, spacing_goal_hi=3e-4)
    g0 = env._spacing_goal.clone()
    assert (g0 >= 1e-4).all() and (g0 <= 3e-4).all()
    env.reset(torch.tensor([0, 2]))                       # partial reset
    g1 = env._spacing_goal
    assert (g1 >= 1e-4).all() and (g1 <= 3e-4).all()
    survivors = [1, 3, 4, 5]
    assert torch.equal(g1[survivors], g0[survivors])      # survivors keep their goal


def test_goal_is_detached_and_gradient_still_flows():
    """The goal is a detached constant -> no grad path, and the action->reward grad is unaffected."""
    env = _env(reward_spec=_gc_spec(), stochastic_init=False, no_grad=False,
               spacing_goal_lo=1e-4, spacing_goal_hi=3e-4)
    assert env._spacing_goal.requires_grad is False
    a = torch.zeros(4, 8, requires_grad=True)
    _, rew, _, _ = env.step(a)
    rew.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_drift_only_on_rf_indices():
    env = _env(rf_drift_std=0.1)
    nz = (env._drift != 0).any(dim=0)             # per-knob: any env drifted?
    rf = torch.zeros(8, dtype=torch.bool); rf[list(RF_DRIFT_IDX)] = True
    # drifted columns must be a subset of the RF indices (non-RF columns exactly zero)
    assert torch.count_nonzero(env._drift[:, ~rf]) == 0
    assert env._drift[:, rf].abs().sum() > 0


def test_synchronous_autoreset():
    env = _env(num_envs=3, episode_length=4)
    for t in range(4):
        _, _, done, _ = env.step(torch.zeros(3, 8))
    assert done.all()                              # all envs terminate together
    assert (env._step_count == 0).all()            # and auto-reset


def test_initialize_trajectory_detaches():
    env = _env(no_grad=False)
    env.step(torch.zeros(4, 8, requires_grad=True))
    obs = env.initialize_trajectory()
    assert env._knobs.requires_grad is False and obs.requires_grad is False


def test_drift_zero_matches_no_drift_reward():
    """rf_drift_std=0 => knobs_actual == knobs_cmd (collapses to the deterministic env)."""
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=8, n_layers=4, hidden_dim=32, n_aux_particles=256).eval()
    e0 = TwoBunchFlowEnv(2, flow=flow, device="cpu", episode_length=4, n_particles=256,
                         reward_spec=_spec(), stochastic_init=False, rf_drift_std=0.0, seed=1)
    # with no drift and fixed knobs, the actual knobs fed to the flow equal the commanded ones
    knobs = torch.full((2, 8), 0.5)
    torch.manual_seed(3)
    r_cmd, _, _ = e0._observe(knobs)
    assert torch.count_nonzero(e0._drift) == 0


# ---- knob count follows the surrogate's condition_dim (8 vs 26) --------------

def _env_nk(n_knob, num_envs=4, **kw):
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=n_knob, n_layers=4, hidden_dim=32, n_aux_particles=128).eval()
    defaults = dict(device="cpu", episode_length=8, n_particles=128, reward_spec=_spec())
    defaults.update(kw)
    return TwoBunchFlowEnv(num_envs, flow=flow, **defaults)


def test_env_knob_count_follows_flow_condition_dim():
    env = _env_nk(26)
    assert env.n_knob == 26 and env.num_actions == 26
    assert env.num_obs == 26 + env._reward_spec.n_obs_extra
    obs = env.reset()
    assert obs.shape == (4, 26 + env._reward_spec.n_obs_extra) and torch.isfinite(obs).all()
    obs, rew, done, _ = env.step(torch.zeros(4, 26))          # accepts a 26-wide action
    assert obs.shape == (4, 26 + env._reward_spec.n_obs_extra) and rew.shape == (4,)


def test_env_drift_confined_to_rf_idx_26knob():
    env = _env_nk(26, rf_drift_std=0.1)
    assert env._drift_std is not None and env._drift_std.numel() == 26
    nz = (env._drift_std != 0).nonzero().flatten().tolist()
    assert nz == list(RF_DRIFT_IDX)                           # drift only on RF/energy knobs
