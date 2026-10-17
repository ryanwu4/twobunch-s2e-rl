"""DeltaControlEnvV4: actuator models, drift, noise floor, reward on truth, history, limits."""
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.rl.delta_env_v4 import DeltaControlEnvV4  # noqa: E402
from tests.test_delta_env import BASE, CTRL, KS, _spec  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402


def _env(num_envs=16, **kw):
    torch.manual_seed(0)
    flow = TwoBunchFlow(condition_dim=8, n_layers=2, hidden_dim=16, n_aux_particles=64).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
    d = dict(observe_fn=lambda k: flow.observables(k, n=64), knob_space=KS, baseline=BASE,
             controls=CTRL, reward_spec=_spec(), device="cpu", episode_length=6,
             action_step={"L1PhaseSet": 0.3, "L2PhaseSet": 0.3},
             start_box={"L1PhaseSet": (BASE["L1PhaseSet"] - 2, BASE["L1PhaseSet"] + 2),
                        "L2PhaseSet": (-37.5, -34.5)},
             actuators={"L1PhaseSet": {"type": "blen_dlog", "step": 0.1, "gain_range": [1.5, 4.0],
                                       "readback_noise": 0.0, "cum_limit": 0.25},
                        "L2PhaseSet": {"type": "direct", "gain_range": [0.35, 1.1]}})
    d.update(kw)
    return DeltaControlEnvV4(num_envs, **d)


def test_blen_actuator_moves_l1_by_minus_g_dln_and_readback_sees_it():
    env = _env()
    t0 = env.true_controls().clone()
    env.step(torch.tensor([[1.0, 0.0]] * 16))
    d = env.true_controls() - t0
    torch.testing.assert_close(d[:, 0], -env._g[:, 0] * 0.1)        # dln = +0.1 -> dL1 = -g*0.1
    torch.testing.assert_close(env._cum_obs[:, 0], d[:, 0])          # readback = beam delta (no noise)
    assert ((env._g[:, 0] >= 1.5) & (env._g[:, 0] <= 4.0)).all()
    assert ((env._g[:, 1] >= 0.35) & (env._g[:, 1] <= 1.1)).all()


def test_cumulative_limit_and_obs_dim():
    env = _env(history=3)
    for _ in range(4):
        obs, *_ = env.step(torch.tensor([[1.0, 1.0]] * 16))
    assert torch.allclose(env._cum_act[:, 0], torch.full((16,), 0.25))   # capped at 0.25 ln
    assert obs.shape == (16, env.num_obs)


def test_adapter_learns_gain_and_reaches_requested_delta():
    env = _env(actuators={"L1PhaseSet": {"type": "adapter", "step": 0.05, "gain_range": [1.5, 4.0]},
                          "L2PhaseSet": {"type": "direct"}})
    for _ in range(3):
        t0 = env.true_controls().clone()
        env.step(torch.tensor([[-1.0, 0.0]] * 16))      # away from the box top (no clamping)
    torch.testing.assert_close(env._g_hat[:, 0], env._g[:, 0], rtol=1e-3, atol=0)   # noise-free LS
    torch.testing.assert_close((env.true_controls() - t0)[:, 0], torch.full((16,), -0.05), rtol=1e-3, atol=1e-6)


def test_reward_on_truth_ignores_measurement_noise_and_gradient_flows():
    env = _env(shot_noise={"bunch_spacing": 1e-3}, shots_per_step=1, reward_on="true")
    a = torch.zeros(16, 2, requires_grad=True)
    _, r1, _, _ = env.step(a)
    r1.sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    env2 = _env(shot_noise={"bunch_spacing": 1e-3}, shots_per_step=1, reward_on="measured")
    _, r2, _, _ = env2.step(torch.zeros(16, 2))
    assert not torch.allclose(r1.detach(), r2)                        # huge noise moves the measured reward


def test_drift_and_noise_floor_and_offcontext_goals_run():
    env = _env(drift_per_step={"L2PhaseSet": 0.05}, step_noise_floor={"bunch_spacing": 6e-6},
               outlier_prob=0.5, offcontext_goal_frac=0.5, action_cost=0.1)
    t0 = env.true_controls().clone()
    for _ in range(3):
        obs, r, done, info = env.step(torch.zeros(16, 2))
    assert (env.true_controls()[:, 1] != t0[:, 1]).any()              # drift moved L2 with zero action
    assert torch.isfinite(obs).all() and torch.isfinite(r).all()
    for _ in range(4):                                                 # through an episode reset
        obs, r, done, info = env.step(torch.zeros(16, 2))
    assert torch.isfinite(obs).all()


def test_goal_radius_keeps_goals_near_the_start():
    env = _env(num_envs=64, goal_radius={"L1PhaseSet": 0.5, "L2PhaseSet": 0.7})
    d = (env._goal_ctrl - env._start).abs()
    assert (d[:, 0] <= 0.5 + 1e-5).all() and (d[:, 1] <= 0.7 + 1e-5).all()
    for _ in range(7):                                   # through an episode reset
        env.step(torch.zeros(64, 2))
    d = (env._goal_ctrl - env._start).abs()
    assert (d[:, 0] <= 0.5 + 1e-5).all() and (d[:, 1] <= 0.7 + 1e-5).all()


def test_direct_rb_actuator_moves_l1_by_g_a_step_and_observes_the_readback():
    env = _env(actuators={"L1PhaseSet": {"type": "direct_rb", "step": 0.25, "gain_range": [0.7, 1.3],
                                         "readback_noise": 0.0, "cum_limit": 2.5},
                          "L2PhaseSet": {"type": "direct", "gain_range": [0.35, 1.1]}})
    t0 = env.true_controls().clone()
    env.step(torch.tensor([[1.0, 1.0]] * 16))
    d = env.true_controls() - t0
    torch.testing.assert_close(d[:, 0], env._g[:, 0] * 0.25)          # + sign, gain on the command
    torch.testing.assert_close(env._cum_obs[:, 0], d[:, 0])           # readback = true delta (no noise)
    torch.testing.assert_close(env._cum_obs[:, 1], torch.full((16,), 0.3))   # direct: the command
    assert ((env._g[:, 0] >= 0.7) & (env._g[:, 0] <= 1.3)).all()


def test_direct_rb_klystron_readback_hides_beam_gain_and_ramp_drifts():
    acts = {"L1PhaseSet": {"type": "direct_rb", "step": 0.25, "gain_range": [0.7, 1.3], "readback_noise": 0.0,
                           "cum_limit": 2.5, "readback_gain_range": [0.95, 1.05]},
            "L2PhaseSet": {"type": "direct", "gain_range": [0.35, 1.1]}}
    env = _env(actuators=acts)
    t0 = env.true_controls().clone()
    env.step(torch.tensor([[1.0, 0.0]] * 16))
    torch.testing.assert_close(env.true_controls()[:, 0] - t0[:, 0], env._g[:, 0] * 0.25)   # beam: hidden g
    torch.testing.assert_close(env._cum_obs[:, 0], env._g_rf[:, 0] * 0.25)                 # readback: g_rf
    assert ((env._g_rf[:, 0] >= 0.95) & (env._g_rf[:, 0] <= 1.05)).all()
    assert not torch.allclose(env._cum_obs[:, 0], env.true_controls()[:, 0] - t0[:, 0])
    env2 = _env(actuators=acts, drift_ramp={"L1PhaseSet": 0.05})
    r = env2._ramp[:, 0].clone()
    assert (r.abs() <= 0.05).all() and r.abs().max() > 0
    t0 = env2.true_controls().clone()
    env2.step(torch.zeros(16, 2))
    torch.testing.assert_close(env2.true_controls()[:, 0] - t0[:, 0], r)                    # no action: ramp only


BLEN = {"BLEN_LI11_359": {"centers": {"L1PhaseSet": -20.29},
                          "poly": {"L1PhaseSet": -0.218, "L1PhaseSet^2": 0.0246, "L1PhaseSet^3": 0.00233},
                          "slope_scale": [0.8, 1.3], "noise": 0.0},
        "BLEN_LI14_888": {"centers": {"L1PhaseSet": -22.0, "L2PhaseSet": -37.0},
                          "poly": {"L2PhaseSet": 0.199, "L1PhaseSet*L2PhaseSet": 0.014},
                          "slope_scale": [1.0, 1.0], "noise": 0.0}}
RB = {"L1PhaseSet": {"type": "direct_rb", "step": 0.25, "gain_range": [0.7, 1.3], "readback_noise": 0.0,
                     "cum_limit": 2.5, "readback_gain_range": [0.95, 1.05]},
      "L2PhaseSet": {"type": "direct", "gain_range": [0.35, 1.1]}}


def test_blen_obs_log_ratio_follows_the_beam_side_controls():
    base = _env(actuators=RB)
    env = _env(actuators=RB, blen_obs=BLEN)
    assert env.num_obs == base.num_obs + 2
    obs0 = env.reset()
    torch.testing.assert_close(obs0[:, -2:], torch.zeros(16, 2))         # ratio to the start reading
    t0 = env.true_controls().clone()
    obs, *_ = env.step(torch.tensor([[1.0, 1.0]] * 16))
    t1 = env.true_controls()
    d1, d0 = t1[:, 0] + 20.29, t0[:, 0] + 20.29
    p = lambda d: -0.218 * d + 0.0246 * d ** 2 + 0.00233 * d ** 3
    s = env._blen_s[:, 0]
    assert ((s >= 0.8) & (s <= 1.3)).all() and s.std() > 0
    torch.testing.assert_close(obs[:, -2], s * (p(d1) - p(d0)))           # beam-side L1 (hidden gain g)
    q = lambda c: 0.199 * (c[:, 1] + 37) + 0.014 * (c[:, 0] + 22) * (c[:, 1] + 37)
    torch.testing.assert_close(obs[:, -1], q(t1) - q(t0), rtol=1e-4, atol=1e-5)   # realised L2 (hidden L2 gain)
    assert obs.shape == (16, env.num_obs)


def test_blen_obs_off_keeps_the_env_identical_and_bc11_refused_with_loop_on():
    runs = []
    for kw in ({}, {"blen_obs": None}):                                 # global RNG: build + roll one at a time
        e = _env(actuators=RB, **kw)
        runs.append(torch.cat([e.reset(), e.step(torch.ones(16, 2))[0]]))
    torch.testing.assert_close(runs[0], runs[1])
    with pytest.raises(ValueError, match="BC11 loop is on"):
        _env(blen_obs={"BLEN_LI11_359": BLEN["BLEN_LI11_359"]})          # default L1 actuator = blen_dlog


def test_env_factory_forwards_every_v4_option():
    """build_delta_env_fn builds the v4 kwargs by hand: every DeltaControlEnvV4 option must be forwarded
    (2026-10-16: blen_obs was silently dropped, so configs trained without the BLEN columns)."""
    import inspect
    from twobunch_s2e_rl.rl import _train_utils
    src = inspect.getsource(_train_utils.build_delta_env_fn)
    opts = [p for p in inspect.signature(DeltaControlEnvV4.__init__).parameters if p not in ("self", "num_envs", "kw")]
    assert [o for o in opts if f"{o}=" not in src] == []
