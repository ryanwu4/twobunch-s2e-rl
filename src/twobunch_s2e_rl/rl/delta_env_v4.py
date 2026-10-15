"""Delta controller env v4: the real-machine actuator / measurement model on top of DeltaControlEnv.

Separate subclass so v1-v3 envs (and every policy trained on them) stay byte-identical. Decisions it
implements (results/plan_review_2026-10-14/, docs/overnight_2026-10-15.md):

Actuators (`actuators: {control: {...}}`), each with a hidden per-episode gain g:
  direct     true delta = g * a * step                      (L2 via the relative L2_PHASE.MKB; the
             BC14/BC20 loops' SBST trims make g < 1: gain_range ~ [0.35, 1.1])
  blen_dlog  the action is a change of ln(BC11 BLEN setpoint): dln = a * step (step in ln units);
             true dL1 = -g * dln with g in deg/ln (DAQ: 2.2-3.4 deg/ln, local 1.9-3.7); the BC11
             loop is on, so the OBSERVED dL1 is the FAST_PACT readback = true dL1 + readback noise +
             the loop's drift compensation (a hidden random walk that moves the readback, not the
             beam-relative phase)
  adapter    the policy asks for dL1 = a * step deg; a deterministic adapter turns it into a dln
             with its own gain estimate g_hat (exponentially-forgotten least squares on the observed
             (dln, dL1) pairs, prior = the range's geometric mean); true dL1 = a*step*g/g_hat. The
             round-2 ML review's "structured action" alternative to blen_dlog.
  Gains are drawn uniform (or log-uniform) in gain_range; cum_limit caps |cumulative action| (in the
  action's own units: deg, or ln for blen_dlog).
Drift (`drift_per_step: {control: sigma}`): hidden random walk on the beam-effective control.
Measurement (`measurement` block, on top of shot_noise / M and the streak scale error):
  step_noise_floor {feature: sigma}   per-step noise NOT reduced by M (the +/-V images are ~11 s apart)
  outlier_prob, outlier_scale         a step's features get an extra N(0, (scale*floor)^2)
Reward: reward_on = "true" scores the goal terms on the privileged true feature (cloud twin) against
the true goal value (no reward-on-noise, no invalid-reading fallback); action_cost * sum(a^2).
Goals: goal_valid_retries > 0 redraws goal points whose noise-free screen reading is deployed-invalid
(screen mode). offcontext_goal_frac of episodes take their goal from a different hidden context (the
deployment case of a goal recorded under another machine state; may be unreachable).
Obs: v3 obs (cumulative delta column = what the operator side can see: readback dL1 for blen_dlog,
commanded delta otherwise) + adapter gain estimate (log ratio to prior) + `history` k-1 previous
(goal error, action, observed delta) frames.
"""
from __future__ import annotations

import math

import torch

from .delta_env import DeltaControlEnv


class DeltaControlEnvV4(DeltaControlEnv):
    def __init__(self, num_envs: int, *, actuators: dict, drift_per_step: dict | None = None,
                 step_noise_floor: dict | None = None, outlier_prob: float = 0.0,
                 outlier_scale: float = 5.0, reward_on: str = "true", action_cost: float = 0.0,
                 offcontext_goal_frac: float = 0.0, history: int = 1, goal_radius: dict | None = None,
                 goal_valid_retries: int = 0,
                 **kw):
        if reward_on not in ("true", "measured"):
            raise ValueError(f"reward_on must be 'true' or 'measured', got {reward_on!r}")
        self._v4_ready = False
        controls = list(kw["controls"])
        dev = torch.device(kw.get("device", "cuda:0"))
        f = dict(device=dev, dtype=torch.float32)
        acts = [dict(actuators.get(c, {"type": "direct"})) for c in controls]
        for c, a in zip(controls, acts):
            if a.get("type", "direct") not in ("direct", "blen_dlog", "adapter"):
                raise ValueError(f"actuator type for {c} must be direct | blen_dlog | adapter")
            a.setdefault("type", "direct")
            a.setdefault("gain_range", [1.0 - kw.get("gain_halfwidth", 0.2), 1.0 + kw.get("gain_halfwidth", 0.2)]
                         if a["type"] == "direct" else [1.5, 4.0])
        self._acts = acts
        self._a_type = [a["type"] for a in acts]
        self._a_step = torch.tensor([float(a.get("step", kw["action_step"][c])) for c, a in zip(controls, acts)], **f)
        self._g_lo = torch.tensor([float(a["gain_range"][0]) for a in acts], **f)
        self._g_hi = torch.tensor([float(a["gain_range"][1]) for a in acts], **f)
        self._g_log = torch.tensor([bool(a.get("log_uniform", False)) for a in acts], device=dev)
        self._g_prior = (self._g_lo * self._g_hi).sqrt()
        self._sign = torch.tensor([-1.0 if a["type"] in ("blen_dlog", "adapter") else 1.0 for a in acts], **f)
        self._rb_noise = torch.tensor([float(a.get("readback_noise", 0.0)) for a in acts], **f)
        self._rb_drift = torch.tensor([float(a.get("readback_drift", 0.0)) for a in acts], **f)
        self._cum_limit = torch.tensor([float(a.get("cum_limit", math.inf)) for a in acts], **f)
        self._forget = float(next((a.get("forgetting", 0.8) for a in acts if a["type"] == "adapter"), 0.8))
        self._drift = torch.tensor([float((drift_per_step or {}).get(c, 0.0)) for c in controls], **f)
        self._floor = dict(step_noise_floor or {})
        self._p_out, self._out_scale = float(outlier_prob), float(outlier_scale)
        self._reward_on, self._a_cost = reward_on, float(action_cost)
        self._off_goal = float(offcontext_goal_frac)
        # goals within reach of the start (the actuator caps make far goals unreachable): the goal's
        # control point = start + U(+-radius) per control, clipped to the start box. None = anywhere in
        # the start box (v1-v3 behaviour).
        self._goal_radius = (torch.tensor([float(goal_radius[c]) for c in controls], **f)
                             if goal_radius else None)
        self._goal_retries = int(goal_valid_retries)
        self._k = max(int(history), 1)
        super().__init__(num_envs, **kw)

    # ----- obs layout -------------------------------------------------------------------------
    def _v4_init_state(self):
        B, n = self.num_envs, len(self.controls)
        z = lambda *s: torch.zeros(*s, device=self.device)
        n_goal = len(self._goal_terms)
        self._frame_dim = n_goal + n + n                      # (err, action, observed delta) per step
        self._true = z(B, n)                                  # beam-effective controls (grad)
        self._cum_act, self._cum_obs = z(B, n), z(B, n)
        self._g = torch.ones(B, n, device=self.device)
        self._g_hat = self._g_prior.expand(B, -1).clone()
        self._ls_xx, self._ls_xy = z(B, n), z(B, n)
        self._last_dobs = z(B, n)
        self._hist = z(B, self._k - 1, self._frame_dim) if self._k > 1 else None
        self.num_obs = (n + self._spec.n_obs_extra + 2 * n_goal + n) + n + (self._k - 1) * self._frame_dim
        self._v4_ready = True

    def _compute_obs(self) -> torch.Tensor:
        if not self._v4_ready:
            return super()._compute_obs()
        parts = [self._cum_obs, self._last_extra, self._err, self._err - self._err_prev, self._last_action,
                 (self._g_hat / self._g_prior).log()]
        if self._hist is not None:
            parts.append(self._hist.flatten(1))
        return torch.cat(parts, -1)

    # ----- plant ------------------------------------------------------------------------------
    def true_controls(self, cmd=None) -> torch.Tensor:
        if not self._v4_ready:
            return super().true_controls(cmd)
        return torch.minimum(torch.maximum(self._true, self._lo[self._ci]), self._hi[self._ci])

    def _plant(self, true_ctrl, noise):
        obs = super()._plant(true_ctrl, noise)
        if noise and self._v4_ready:
            for key, sig in self._floor.items():
                obs[key] = obs[key] + self._randn(self.num_envs) * sig
            if self._p_out > 0:
                bad = (self._rand(self.num_envs) < self._p_out).float()
                for key, sig in self._floor.items():
                    obs[key] = obs[key] + bad * self._randn(self.num_envs) * sig * self._out_scale
        return obs

    def _reward_view(self, obs):
        if self._reward_on == "measured" or not self._v4_ready:
            return super()._reward_view(obs)
        view = dict(obs)
        for key, g, _ in self._goal_terms:                    # privileged truth vs the true goal
            view[key] = obs.get(f"{key}_cloud", obs.get(f"{key}_true", obs[key]))
            view[g] = self._goals_cloud[g]
        return view

    # ----- episode ----------------------------------------------------------------------------
    def reset(self, env_ids=None):
        first = not self._v4_ready
        DeltaControlEnv.reset(self, env_ids)          # start / context / goals / hidden draws
        if first:
            self._v4_init_state()
        B, n = self.num_envs, len(self.controls)
        m = torch.zeros(B, dtype=torch.bool, device=self.device)
        m[torch.arange(B, device=self.device) if env_ids is None or first else env_ids] = True
        mc = m.unsqueeze(-1)
        u = self._rand(B, n)
        g_lin = self._g_lo + u * (self._g_hi - self._g_lo)
        g_log = (self._g_lo.log() + u * (self._g_hi.log() - self._g_lo.log())).exp()
        self._g = torch.where(mc, torch.where(self._g_log, g_log, g_lin), self._g)
        z = torch.zeros(B, n, device=self.device)
        for name in ("_cum_act", "_ls_xx", "_ls_xy", "_last_dobs"):
            setattr(self, name, torch.where(mc, z, getattr(self, name)))
        self._cum_obs = torch.where(mc, z, self._cum_obs.detach() if m.all() else self._cum_obs)
        self._g_hat = torch.where(mc, self._g_prior.expand(B, -1), self._g_hat)
        if self._hist is not None:
            self._hist = torch.where(m.view(-1, 1, 1), torch.zeros_like(self._hist), self._hist)
        # beam-effective controls start at the true start; re-evaluate the reset envs there
        self._true = torch.where(mc, self._start.detach(), self._true.detach() if m.all() else self._true)
        with torch.no_grad():
            _, extra, err, achieved = self._evaluate(self._cmd.detach())
        self._last_extra = torch.where(mc, extra, self._last_extra)
        self._err = torch.where(mc, err, self._err)
        self._err_prev = torch.where(mc, err, self._err_prev)
        for k, v in achieved.items():
            old = self._last_achieved.get(k)
            self._last_achieved[k] = v if old is None or old.shape != v.shape else torch.where(
                m.view(-1, *([1] * (v.dim() - 1))), v, old)
        # hidden context the goal was drawn under (= the episode's, except off-context goals; eval only)
        gctx = getattr(self, "_goal_context", None)
        self._goal_context = self._context.clone() if gctx is None or gctx.shape != self._context.shape \
            else torch.where(mc[:, :1].expand_as(self._context), self._context, gctx)
        if self._goal_radius is not None:
            self._reachable_goals(m)
        if self._off_goal > 0:
            self._offcontext_goals(m)
        return self._compute_obs()

    @torch.no_grad()
    def _reachable_goals(self, m):
        """Redraw the reset envs' goal control points within goal_radius of the start, then the goals
        (noise-free measured value + the true value there, as the base reset does)."""
        B, n = self.num_envs, len(self.controls)
        draw = lambda: torch.minimum(torch.maximum(self._start + (2 * self._rand(B, n) - 1) * self._goal_radius,
                                                   self._start_lo), self._start_hi)
        self._goal_ctrl = torch.where(m.unsqueeze(-1), draw(), self._goal_ctrl)
        g_obs = self._plant(self._goal_ctrl, noise=False)
        # goals the deployed screen analysis can read (an operator picks a measurable goal): redraw the
        # reset envs whose noise-free goal reading is deployed-invalid, up to goal_valid_retries times
        for _ in range(self._goal_retries):
            if "screen_valid_deployed" not in g_obs:
                break
            bad = m & (g_obs["screen_valid_deployed"] < 0.5)
            if not bad.any():
                break
            self._goal_ctrl = torch.where(bad.unsqueeze(-1), draw(), self._goal_ctrl)
            g_obs = self._plant(self._goal_ctrl, noise=False)
        g_view = DeltaControlEnv._reward_view(self, g_obs)
        for key, g, _ in self._goal_terms:
            self._goals[g] = torch.where(m, g_view[key], self._goals[g])
            truth = g_obs.get(f"{key}_cloud", g_obs[f"{key}_true"])
            self._goals_cloud[g] = torch.where(m, truth, self._goals_cloud[g])
        _, extra, err, _ = self._evaluate(self._cmd.detach())
        mc = m.unsqueeze(-1)
        self._last_extra = torch.where(mc, extra, self._last_extra)
        self._err = torch.where(mc, err, self._err)
        self._err_prev = torch.where(mc, err, self._err_prev)

    @torch.no_grad()
    def _offcontext_goals(self, m):
        """Redraw the goals of a fraction of the reset envs under a DIFFERENT hidden context."""
        pick = m & (self._rand(self.num_envs) < self._off_goal)
        if not pick.any():
            return
        unit = self._randn(self.num_envs, self._ks.n)
        other = torch.minimum(torch.maximum(self._base + unit * self._ctx_sigma, self._lo), self._hi)
        saved, self._context = self._context, torch.where(pick.unsqueeze(-1), other, self._context)
        self._goal_context = torch.where(pick.unsqueeze(-1), other, self._goal_context)
        g_obs = self._plant(self._goal_ctrl, noise=False)
        self._context = saved
        for key, g, _ in self._goal_terms:
            self._goals[g] = torch.where(pick, g_obs[key], self._goals[g])
            truth = g_obs.get(f"{key}_cloud", g_obs[f"{key}_true"])
            self._goals_cloud[g] = torch.where(pick, truth, self._goals_cloud[g])
        _, extra, err, _ = self._evaluate(self._cmd.detach())
        pc = pick.unsqueeze(-1)
        self._last_extra = torch.where(pc, extra, self._last_extra)
        self._err = torch.where(pc, err, self._err)
        self._err_prev = torch.where(pc, err, self._err_prev)

    def step(self, action):
        if action.shape != (self.num_envs, len(self.controls)):
            raise ValueError(f"action shape {tuple(action.shape)} != ({self.num_envs}, {len(self.controls)})")
        a = action.clamp(-1.0, 1.0)
        # cumulative-delta limits, in the action's own units
        act = a * self._a_step
        lim = self._cum_limit
        act = torch.minimum(torch.maximum(self._cum_act + act, -lim), lim) - self._cum_act
        adapter = torch.tensor([t == "adapter" for t in self._a_type], device=self.device)
        ratio = torch.where(adapter, self._g / self._g_hat.detach(), self._g)
        d_true = torch.where(adapter, act * ratio, self._sign * self._g * act)
        drift_step = self._randn(*act.shape) * self._drift            # hidden random walk
        new_true = self._true + d_true + drift_step
        self._true = new_true
        # observed delta: readback dL1 (+ noise + the loop's drift compensation) for BLEN / adapter
        # actuators, the commanded delta for direct ones
        readback = torch.tensor([t in ("blen_dlog", "adapter") for t in self._a_type], device=self.device)
        rb_step = self._randn(*act.shape) * self._rb_noise + self._randn(*act.shape) * self._rb_drift
        d_obs = torch.where(readback, d_true.detach() + rb_step, act.detach())
        # adapter gain estimate: forgetting least squares of -dL1_obs on dln
        with torch.no_grad():
            dln = torch.where(adapter, -act / self._g_hat, torch.zeros_like(act))
            lam = self._forget
            self._ls_xx = lam * self._ls_xx + dln * dln
            self._ls_xy = lam * self._ls_xy + dln * (-d_obs)
            est = self._ls_xy / self._ls_xx.clamp_min(1e-12)
            ok = adapter & (self._ls_xx > 1e-6) & (est > 0)
            self._g_hat = torch.where(ok, est.clamp(self._g_lo * 0.5, self._g_hi * 2), self._g_hat)
        self._cum_act = (self._cum_act + act).detach()
        self._cum_obs = self._cum_obs + d_obs
        self._last_dobs = d_obs
        # base step machinery with the v4 controls (cmd is not used by v4's true_controls)
        if self.no_grad:
            with torch.no_grad():
                reward, extra, err, achieved = self._evaluate(self._cmd)
        else:
            reward, extra, err, achieved = self._evaluate(self._cmd)
        if self._a_cost > 0:
            reward = reward - self._a_cost * (a ** 2).sum(-1)
        if self._hist is not None:
            frame = torch.cat([self._err, self._last_action, self._last_dobs], -1).detach()
            self._hist = torch.cat([self._hist[:, 1:], frame.unsqueeze(1)], 1)
        self._err_prev, self._err = self._err, err
        self._last_extra, self._last_action = extra, action.detach()
        self._last_achieved = achieved
        self._step_count = self._step_count + 1
        done = self._step_count >= self.episode_length
        obs_pre = self._compute_obs().detach()
        achieved_info = {k: v.clone() for k, v in achieved.items()}
        achieved_info.update({g: v.clone() for g, v in self._goals.items()})
        achieved_info.update({f"{g}_cloud": v.clone() for g, v in self._goals_cloud.items()})
        achieved_info["true_gain"] = self._g.clone()
        achieved_info["gain_estimate"] = self._g_hat.clone()
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset(done_ids)
        return self._compute_obs(), reward, done, {"obs_before_reset": obs_pre, "achieved": achieved_info}

    def initialize_trajectory(self):
        self._true = self._true.detach()
        self._cum_obs = self._cum_obs.detach()
        return super().initialize_trajectory()

    def clear_grad(self):
        self._true = self._true.detach()
        self._cum_obs = self._cum_obs.detach()
        super().clear_grad()
