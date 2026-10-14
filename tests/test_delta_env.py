"""DeltaControlEnv: delta-only obs, hidden misspecification/context, achievable goals, noise."""
import math

import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set  # noqa: E402
from twobunch_s2e_rl.knobs import KnobSpace  # noqa: E402
from twobunch_s2e_rl.rl.delta_env import DeltaControlEnv, readback_sigma  # noqa: E402
from twobunch_s2e_rl.rl.reward import build_twobunch_reward_spec  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

KEYS, LO, HI, BASE = resolve_sweep_set("original8")
KS = KnobSpace.from_lists(KEYS, LO, HI)
CTRL = ["L1PhaseSet", "L2PhaseSet"]


def _spec():
    norms = {"bunch_spacing": {"transform": "identity", "mean": 2e-4, "std": 1e-4},
             "drive_sigma_z": {"transform": "log10", "mean": -4.5, "std": 0.3},
             "T_drive": {"transform": "identity", "mean": 0.9, "std": 0.1},
             "T_witness": {"transform": "identity", "mean": 0.9, "std": 0.1}}
    objs = [{"key": "bunch_spacing", "kind": "target", "goal": [1e-4, 3e-4]},
            {"key": "drive_sigma_z", "kind": "target", "goal": [1e-5, 1e-4]},
            {"key": "T_witness", "kind": "hinge", "hinge_at": 0.9},
            {"key": "knobs", "kind": "boundary", "margin": 0.05}]
    return build_twobunch_reward_spec(norms, {}, objectives=objs)


def _env(num_envs=8, **kw):
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=8, n_layers=2, hidden_dim=16, n_aux_particles=64).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
    d = dict(observe_fn=lambda k: flow.observables(k, n=64), knob_space=KS, baseline=BASE,
             controls=CTRL, reward_spec=_spec(), device="cpu", episode_length=4,
             action_step={"L1PhaseSet": 0.3, "L2PhaseSet": 0.3},
             start_box={"L1PhaseSet": (BASE["L1PhaseSet"] - 2, BASE["L1PhaseSet"] + 2),
                        "L2PhaseSet": (-37.5, -34.5)},
             offset_sigma={"L1PhaseSet": 1.0, "L2PhaseSet": 1.0}, gain_halfwidth=0.2)
    d.update(kw)
    return DeltaControlEnv(num_envs, **d)


def test_dims_and_contract():
    env = _env()
    assert env.num_actions == 2
    # cum delta (2) + obs keys (spacing, sigma_z, T_w + 2 goals = 5) + err (2) + trend (2) + last a (2)
    assert env.num_obs == 2 + 5 + 2 + 2 + 2
    obs, rew, done, info = env.step(torch.zeros(8, 2))
    assert obs.shape == (8, env.num_obs) and rew.shape == (8,) and torch.isfinite(rew).all()
    assert "bunch_spacing_true" in info["achieved"] and "spacing_goal" in info["achieved"]


def test_obs_has_no_absolute_knobs_at_reset():
    """Different hidden starts / offsets / gains give identical delta, trend and action columns."""
    env = _env()
    obs = env.reset()
    assert torch.all(obs[:, :2] == 0)                     # cumulative delta
    assert torch.all(obs[:, -4:] == 0)                    # error trend + last action
    assert env._start.std(0).min() > 0 and env._offset.std(0).min() > 0   # starts do differ


def test_cumulative_delta_tracks_commands_not_truth():
    env = _env()
    env.reset()
    obs, *_ = env.step(torch.ones(8, 2))
    torch.testing.assert_close(obs[:, :2], torch.full((8, 2), 0.3))
    true = env.true_controls(env._cmd)
    torch.testing.assert_close(true - env._start, 0.3 * env._gain)
    # the readback offset is invisible to the dynamics: true start == sampled start
    torch.testing.assert_close(env.true_controls(env._cmd0), env._start)


def test_misspecification_statistics():
    env = _env(num_envs=4096)
    assert abs(float(env._offset.std()) - 1.0) < 0.05
    assert float(env._gain.min()) >= 0.8 and float(env._gain.max()) <= 1.2
    st = env._start[:, 0] - BASE["L1PhaseSet"]
    assert float(st.min()) >= -2 and float(st.max()) <= 2


def test_achievable_goal_is_reached_at_goal_point():
    """With no shot noise, commanding the true goal point reproduces the goal: error 0."""
    env = _env(num_envs=6)
    env.reset()
    cmd = env._cmd0 + (env._goal_ctrl - env._start) / env._gain
    torch.testing.assert_close(env.true_controls(cmd), env._goal_ctrl, rtol=0, atol=1e-4)
    torch.manual_seed(1)
    _, _, err, _ = env._evaluate(cmd)
    # flow sampling is stochastic (fresh base draws), so compare in the reward's normalized units
    assert float(err.abs().median()) < 1.0


def test_scale_error_affects_measured_only():
    env = _env(num_envs=16, scale_error_keys=("bunch_spacing",), scale_error_halfwidth=0.2)
    _, _, _, info = env.step(torch.zeros(16, 2))
    a = info["achieved"]
    ratio = a["bunch_spacing"] / a["bunch_spacing_true"]
    torch.testing.assert_close(ratio, env._scale, rtol=1e-4, atol=1e-6)
    assert float(env._scale.min()) >= 0.8 and float(env._scale.max()) <= 1.2


def test_shot_noise_scales_as_one_over_sqrt_m():
    flat = lambda k: {"bunch_spacing": torch.full((k.shape[0],), 2e-4),
                      "drive_sigma_z": torch.full((k.shape[0],), 3e-5),
                      "T_witness": torch.ones(k.shape[0]), "T_drive": torch.ones(k.shape[0])}
    stds = []
    for m in (1, 100):
        env = _env(num_envs=4096, observe_fn=flat, shot_noise={"bunch_spacing": 30e-6},
                   shots_per_step=m)
        obs = env._plant(env._start, noise=True)
        stds.append(float(obs["bunch_spacing"].std()))
    assert abs(stds[0] - 30e-6) / 30e-6 < 0.05
    assert abs(stds[0] / stds[1] - 10.0) < 0.6


def test_gradient_flows_action_to_reward():
    env = _env()
    env.reset()
    a = torch.zeros(8, 2, requires_grad=True)
    _, rew, _, _ = env.step(a)
    rew.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_context_jitter_around_baseline():
    env = _env(num_envs=4096)
    i = KS.index("S1ELkG")
    sd = float(env._context[:, i].std())
    assert abs(sd - readback_sigma("S1ELkG", BASE["S1ELkG"])) / sd < 0.1
    j = KS.index("L1PhaseSet")
    assert float(env._ctx_sigma[j]) == 0.0                # controls are not context-jittered


def test_partial_reset_keeps_survivor_state():
    env = _env(num_envs=4)
    env.step(torch.ones(4, 2))
    keep = env._cmd.detach().clone()
    env.reset(torch.tensor([0]))
    assert torch.equal(env._cmd[1:].detach(), keep[1:])
    assert torch.all(env._cmd[0] == env._cmd0[0])


def test_start_box_outside_plant_box_rejected():
    with pytest.raises(ValueError, match="not inside"):
        _env(start_box={"L1PhaseSet": (BASE["L1PhaseSet"] - 2, BASE["L1PhaseSet"] + 2),
                        "L2PhaseSet": (-36.0, -33.0)})


def test_rollout_shapes_and_obs_dim_guard():
    from twobunch_s2e_rl.rl.rollout import rollout

    class _Actor(torch.nn.Module):
        def __init__(self, nin):
            super().__init__()
            self.lin = torch.nn.Linear(nin, 2)

        def forward(self, obs, deterministic=True):
            return self.lin(obs)

    env = _env(num_envs=3)
    env.reset()
    traj = rollout(env, _Actor(env.num_obs), None)
    assert traj["action"].shape == (4, 3, 2) and traj["reward"].shape == (4, 3)
    assert traj["bunch_spacing"].shape == (4, 3) and traj["true_controls"].shape == (4, 3, 2)
    with pytest.raises(ValueError, match="obs-dim mismatch"):
        rollout(env, _Actor(env.num_obs + 1), None)


def test_feature_offset_is_per_episode_and_hidden():
    flat = lambda k: {"bunch_spacing": torch.full((k.shape[0],), 2e-4),
                      "drive_sigma_z": torch.full((k.shape[0],), 3e-5),
                      "T_witness": torch.ones(k.shape[0]), "T_drive": torch.ones(k.shape[0])}
    env = _env(num_envs=2048, observe_fn=flat, feature_offset_sigma={"bunch_spacing": 5e-6})
    _, _, _, info = env.step(torch.zeros(2048, 2))
    d = info["achieved"]["bunch_spacing"] - info["achieved"]["bunch_spacing_true"]
    assert abs(float(d.std()) - 5e-6) / 5e-6 < 0.08            # one draw per episode
    _, _, _, info2 = env.step(torch.zeros(2048, 2))
    d2 = info2["achieved"]["bunch_spacing"] - info2["achieved"]["bunch_spacing_true"]
    assert torch.allclose(d, d2)                               # constant within the episode


def test_plant_resample_hook_called_with_reset_mask():
    """A plant exposing `resample(mask, generator)` (e.g. hidden post-XTCAVF quad strengths) gets
    every reset: all envs at construction, only the done ones afterwards."""
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=8, n_layers=2, hidden_dim=16, n_aux_particles=64).eval()
    masks = []
    obs_fn = lambda k: flow.observables(k, n=64)
    obs_fn.resample = lambda mask, gen: masks.append(mask.clone())
    env = _env(observe_fn=obs_fn)
    assert masks and masks[-1].all()
    env.reset(torch.tensor([1, 3]))
    assert masks[-1].tolist() == [i in (1, 3) for i in range(8)]


def test_reward_only_objective_not_in_obs():
    """`observe: false` keeps a term in the reward but out of the policy obs."""
    norms = {"bunch_spacing": {"transform": "identity", "mean": 2e-4, "std": 1e-4},
             "T_witness": {"transform": "identity", "mean": 0.9, "std": 0.1}}
    objs = [{"key": "bunch_spacing", "kind": "target", "goal": [1e-4, 3e-4]},
            {"key": "T_witness", "kind": "hinge", "hinge_at": 0.9, "observe": False}]
    spec = build_twobunch_reward_spec(norms, {}, objectives=objs)
    assert "T_witness" not in spec.obs_keys and "bunch_spacing" in spec.obs_keys
    assert any(t.key == "T_witness" for t in spec.terms)
    obs = {"bunch_spacing": torch.full((3,), 2e-4), "T_witness": torch.tensor([0.5, 0.95, 1.0]),
           "spacing_goal": torch.full((3,), 2e-4)}
    r, _ = spec(obs)
    assert r[0] < r[1]                                    # the hidden term still shapes the reward
