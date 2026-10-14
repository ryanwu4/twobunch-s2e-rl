"""Delta-only control env: the policy acts by setpoint deltas and reads the beam, never absolute knobs.

Separate from TwoBunchFlowEnv on purpose (that env, and every policy trained on it, stay untouched).
Same DiffRL API (reset / step / initialize_trajectory / clear_grad, num_obs, num_actions), so the
vendored BPTT trains it unchanged. See docs/architecture_plan.md, "First priority: delta controller".

Per episode (all hidden from the policy):
- true start of the controls: uniform in `start_box` (physical)
- context: every non-control knob ~ N(baseline, sigma) at readback-error scale, clipped to the box
- misspecification of each control: the commanded (readback) value differs from the truth by a
  constant offset ~ N(0, offset_sigma) [physical], `cmd0 = start - offset`, and moves with a gain
  error ~ U(1 - g, 1 + g): `true = start + gain * (cmd - cmd0)`. The offset never reaches the
  dynamics or the obs — delta control is offset-immune by construction (the reason to use it); it
  is kept so readbacks / deployment logs are honest. The gain error is what the policy must
  tolerate.
- measurement scale error on chosen features (streak calibration): measured = true * s,
  s ~ U(1 - h, 1 + h)
- plant-side hidden parameters (e.g. the post-XTCAVF quad strengths in screen mode): redrawn for the
  reset envs through `observe_fn.resample(mask, generator)` if the plant provides it
- goals: "achievable" — the measured features of a random true control point in the start box,
  under this episode's context and scale error (noise-free), so every goal is reachable

Per step: cmd += action * action_step (physical units per unit action); the plant
`observe_fn(knobs_norm)` is evaluated at the true knobs; measured features get the scale error and
a shot-averaged noise draw sigma / sqrt(M). Goals and the reward are defined on MEASURED features;
the true values are kept as `<key>_true` for logging and eval.

Obs = [cumulative delta since start (physical), z-scored measured features (+ goal columns), goal
error per goal (reward transform space, / std), error trend (change of that error over the last
step; 0 at reset), last action]. No absolute knob value appears anywhere in the obs.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

from ..knobs import KnobSpace
from .reward import TwoBunchRewardSpec, _t

from ..datagen.sweep_params import SECTION_GAIN_EV, readback_sigma  # noqa: F401  (re-exported)


class DeltaControlEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        render: bool = False,
        seed: int = 0,
        episode_length: int = 32,
        stochastic_init: bool = True,
        MM_caching_frequency: int = 1,       # unused; API compat
        no_grad: bool = False,
        *,
        observe_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
        knob_space: KnobSpace,                # the plant's input frame (all knobs, physical bounds)
        baseline: dict[str, float],           # physical baseline of every knob (golden)
        controls: list[str],
        reward_spec: TwoBunchRewardSpec,
        action_step: dict[str, float],        # physical change per unit action, per control
        start_box: dict[str, tuple[float, float]],
        context_sigma: dict[str, float] | None = None,   # per-knob overrides of readback_sigma
        context_dist: dict[str, dict] | None = None,     # per-knob {dist: normal|uniform, width}
        offset_sigma: dict[str, float] | None = None,    # per-control misspecification offset 1-sigma
        gain_halfwidth: float = 0.2,
        scale_error_keys: tuple[str, ...] = (),
        scale_error_halfwidth: float = 0.0,
        shot_noise: dict[str, float] | None = None,      # per-feature single-shot 1-sigma
        shots_per_step: int = 50,
        fallback_keys: tuple[str, ...] = (),   # screen features whose reward falls back to the cloud
        feature_offset_sigma: dict[str, float] | None = None,   # hidden per-episode additive offset
    ):
        del MM_caching_frequency, render, stochastic_init   # starts are always random in start_box
        self.num_envs, self.episode_length = int(num_envs), int(episode_length)
        self.device, self.no_grad = torch.device(device), bool(no_grad)
        self._observe_fn, self._ks, self._spec = observe_fn, knob_space, reward_spec
        self.controls = list(controls)
        missing = [k for k in knob_space.keys if k not in baseline]
        if missing:
            raise KeyError(f"baseline missing for knobs {missing}")

        dev = dict(device=self.device, dtype=torch.float32)
        self._ci = torch.tensor([knob_space.index(k) for k in self.controls], device=self.device)
        self._base = torch.tensor([baseline[k] for k in knob_space.keys], **dev)
        self._lo = torch.tensor(knob_space.low, **dev)
        self._hi = torch.tensor(knob_space.high, **dev)
        # hidden context: N(baseline, sigma) per knob (readback_sigma rule, or context_sigma), or an
        # explicit context_dist entry {dist: normal | uniform, width: sigma | half-width}. Injector
        # knobs have no readback rule: each must be given in context_dist (or context_sigma).
        from ..datagen.sweep_params import INJECTOR_PARAMS
        ctx, cdist = dict(context_sigma or {}), dict(context_dist or {})
        missing_inj = [k for k in knob_space.keys if k in INJECTOR_PARAMS and k not in ctx and k not in cdist]
        if missing_inj:
            raise ValueError(f"injector knobs need an explicit context_dist entry: {missing_inj}")
        bad = [k for k, v in cdist.items() if v.get("dist") not in ("normal", "uniform")]
        if bad:
            raise ValueError(f"context_dist dist must be 'normal' or 'uniform' for {bad}")

        def width(k):
            if k in self.controls:
                return 0.0
            if k in cdist:
                return float(cdist[k]["width"])
            return float(ctx.get(k, readback_sigma(k, baseline[k])))
        self._ctx_sigma = torch.tensor([width(k) for k in knob_space.keys], **dev)
        self._ctx_uniform = torch.tensor([k in cdist and cdist[k]["dist"] == "uniform"
                                          for k in knob_space.keys], device=self.device)
        self._step_phys = torch.tensor([action_step[k] for k in self.controls], **dev)
        self._start_lo = torch.tensor([start_box[k][0] for k in self.controls], **dev)
        self._start_hi = torch.tensor([start_box[k][1] for k in self.controls], **dev)
        for k in self.controls:
            lo, hi = start_box[k]
            if not (knob_space.low[knob_space.index(k)] <= lo < hi <= knob_space.high[knob_space.index(k)]):
                raise ValueError(f"start_box[{k!r}] = {start_box[k]} is not inside the plant box")
        off = dict(offset_sigma or {})
        self._off_sigma = torch.tensor([off.get(k, 0.0) for k in self.controls], **dev)
        self._gain_hw = float(gain_halfwidth)
        self._scale_keys, self._scale_hw = tuple(scale_error_keys), float(scale_error_halfwidth)
        self._shot = dict(shot_noise or {})
        self._m = int(shots_per_step)
        self._fallback = tuple(fallback_keys)
        # output-side stand-in for injector variation the plant does not model (e.g. the drive charge
        # fraction moves directly with the laser charge ratio): one hidden offset per episode
        self._fo_keys = tuple((feature_offset_sigma or {}).keys())
        self._fo_sigma = torch.tensor([feature_offset_sigma[k] for k in self._fo_keys], **dev)

        # goals: one per goal-conditioned target (metric key -> goal key), achievable sampling
        self._goal_terms = [(t.key, t.goal_key, t) for t in reward_spec.terms if t.goal_key]
        self._rng = torch.Generator(device=self.device).manual_seed(int(seed))

        n_ctrl, n_goal = len(self.controls), len(self._goal_terms)
        self.num_actions = n_ctrl
        self.num_obs = n_ctrl + reward_spec.n_obs_extra + 2 * n_goal + n_ctrl

        z = lambda *s: torch.zeros(*s, **dev)
        B = self.num_envs
        self._cmd, self._cmd0 = z(B, n_ctrl), z(B, n_ctrl)       # commanded (grad) / at reset
        self._start, self._offset, self._gain = z(B, n_ctrl), z(B, n_ctrl), torch.ones(B, n_ctrl, **dev)
        self._context = self._base.expand(B, -1).clone()
        self._scale = torch.ones(B, **dev)
        self._fo = z(B, len(self._fo_keys))
        self._goals = {g: z(B) for _, g, _ in self._goal_terms}
        self._goal_ctrl = z(B, n_ctrl)                            # true control point of the goal
        self._goals_cloud = {g: z(B) for _, g, _ in self._goal_terms}   # ground truth at that point
        self._err, self._err_prev = z(B, n_goal), z(B, n_goal)
        self._last_action = z(B, n_ctrl)
        self._last_extra = z(B, reward_spec.n_obs_extra)
        self._last_achieved: dict[str, torch.Tensor] = {}
        self._step_count = torch.zeros(B, dtype=torch.long, device=self.device)
        self.reset()

    # ----- plant + measurement ----------------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self._rng, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self._rng, device=self.device)

    def true_controls(self, cmd: torch.Tensor) -> torch.Tensor:
        """Commanded -> true control values (physical) through the hidden gain, clamped."""
        true = self._start + self._gain * (cmd - self._cmd0)
        return torch.minimum(torch.maximum(true, self._lo[self._ci]), self._hi[self._ci])

    def _plant(self, true_ctrl: torch.Tensor, noise: bool) -> dict[str, torch.Tensor]:
        """Evaluate the plant at (context + true controls); return measured features + `_true`."""
        knobs = self._context.index_copy(1, self._ci, true_ctrl)
        obs = self._observe_fn(self._ks.normalize(knobs))
        for k in [k for k in obs if not k.endswith("_cloud")]:
            obs[f"{k}_true"] = obs[k]
        for j, k in enumerate(self._fo_keys):          # hidden per-episode feature offsets
            obs[k] = obs[k] + self._fo[:, j]
        for k in self._scale_keys:                     # streak-calibration error
            obs[k] = obs[k] * self._scale
        if noise:
            for k, sig in self._shot.items():          # shot jitter averaged over M shots
                obs[k] = obs[k] + self._randn(self.num_envs) * (sig / math.sqrt(self._m))
        # boundary term sees the CONTROLS only, normalized in the plant frame
        obs["knobs"] = (true_ctrl - self._lo[self._ci]) / (self._hi[self._ci] - self._lo[self._ci])
        return obs

    def _goal_errors(self, obs) -> torch.Tensor:
        cols = [(_t(obs[key], t.transform) - _t(self._goals[g], t.transform)) / t.std
                for key, g, t in self._goal_terms]
        return torch.stack(cols, -1) if cols else torch.zeros(self.num_envs, 0, device=self.device)

    def _reward_view(self, obs: dict) -> dict:
        """Where a screen feature is invalid, the REWARD uses its ground-truth cloud twin
        (privileged, training-only: never in the obs) so the policy keeps a gradient out of invalid
        regions; the fixed penalty is a `screen_valid` hinge objective in the config."""
        if not self._fallback or "screen_valid" not in obs:
            return obs
        ok = obs["screen_valid"] > 0.5
        view = dict(obs)
        for k in self._fallback:
            view[k] = torch.where(ok, obs[k], obs[f"{k}_cloud"])
        return view

    def _evaluate(self, cmd: torch.Tensor):
        obs = self._plant(self.true_controls(cmd), noise=True)
        obs.update(self._goals)
        reward, achieved = self._spec(self._reward_view(obs))
        achieved.update({k: v.detach() for k, v in obs.items() if k.endswith("_cloud")})
        if "screen_valid" in obs:
            achieved["screen_valid"] = obs["screen_valid"].detach()
        achieved.update({k: v.detach() for k, v in obs.items() if k.endswith("_true")})
        achieved["true_controls"] = self.true_controls(cmd).detach()
        extra = self._spec.obs_scaled(obs).detach()
        return reward, extra, self._goal_errors(obs).detach(), {k: v.detach() for k, v in achieved.items()}

    def _compute_obs(self) -> torch.Tensor:
        return torch.cat([self._cmd - self._cmd0, self._last_extra, self._err,
                          self._err - self._err_prev, self._last_action], dim=-1)

    # ----- DiffRL env API -----------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        B = self.num_envs
        if env_ids is None:
            env_ids = torch.arange(B, device=self.device)
        if env_ids.numel() == 0:
            return self._compute_obs()
        m = torch.zeros(B, dtype=torch.bool, device=self.device)
        m[env_ids] = True
        mc = m.unsqueeze(-1)
        n_ctrl = len(self.controls)

        # hidden per-episode draws (drawn for all envs, applied to the reset ones)
        start = self._start_lo + self._rand(B, n_ctrl) * (self._start_hi - self._start_lo)
        offset = self._randn(B, n_ctrl) * self._off_sigma
        gain = 1.0 + (2 * self._rand(B, n_ctrl) - 1) * self._gain_hw
        unit = self._randn(B, self._ks.n)
        if self._ctx_uniform.any():          # extra draw only then: existing configs keep their RNG stream
            unit = torch.where(self._ctx_uniform, 2 * self._rand(B, self._ks.n) - 1, unit)
        context = torch.minimum(torch.maximum(self._base + unit * self._ctx_sigma, self._lo), self._hi)
        scale = 1.0 + (2 * self._rand(B) - 1) * self._scale_hw
        goal_ctrl = self._start_lo + self._rand(B, n_ctrl) * (self._start_hi - self._start_lo)
        fo = self._randn(B, len(self._fo_keys)) * self._fo_sigma

        self._start = torch.where(mc, start, self._start)
        self._offset = torch.where(mc, offset, self._offset)
        self._gain = torch.where(mc, gain, self._gain)
        self._context = torch.where(mc, context, self._context)
        self._scale = torch.where(m, scale, self._scale)
        self._fo = torch.where(mc, fo, self._fo)
        resample = getattr(self._observe_fn, "resample", None)
        if resample is not None:
            resample(m, self._rng)
        # commanded start = true start minus the offset (what the operator's readback would say);
        # the policy only ever sees cmd - cmd0, which is 0 here
        cmd0 = (start - offset).detach()
        self._cmd0 = torch.where(mc, cmd0, self._cmd0)
        self._cmd = torch.where(mc, cmd0, self._cmd.detach() if m.all() else self._cmd)
        self._last_action = torch.where(mc, torch.zeros_like(self._last_action), self._last_action)
        self._step_count[env_ids] = 0

        with torch.no_grad():
            # achievable goals: measured features at a random true control point (noise-free)
            self._goal_ctrl = torch.where(mc, goal_ctrl, self._goal_ctrl)
            g_obs = self._plant(goal_ctrl, noise=False)
            g_view = self._reward_view(g_obs)          # invalid goal reading -> cloud value
            for key, g, _ in self._goal_terms:
                self._goals[g] = torch.where(m, g_view[key].detach(), self._goals[g])
                truth = g_obs.get(f"{key}_cloud", g_obs[f"{key}_true"])
                self._goals_cloud[g] = torch.where(m, truth.detach(), self._goals_cloud[g])
            _, extra, err, achieved = self._evaluate(self._cmd.detach())
        self._last_extra = torch.where(mc, extra, self._last_extra)
        self._err = torch.where(mc, err, self._err)
        self._err_prev = torch.where(mc, err, self._err_prev)       # trend = 0 at reset
        for k, v in achieved.items():
            old = self._last_achieved.get(k)
            self._last_achieved[k] = v if old is None or old.shape != v.shape else torch.where(
                m.view(-1, *([1] * (v.dim() - 1))), v, old)
        return self._compute_obs()

    def initialize_trajectory(self) -> torch.Tensor:
        self._cmd = self._cmd.detach()
        return self._compute_obs()

    def clear_grad(self) -> None:
        self._cmd = self._cmd.detach()

    def step(self, action: torch.Tensor):
        if action.shape != (self.num_envs, len(self.controls)):
            raise ValueError(f"action shape {tuple(action.shape)} != ({self.num_envs}, {len(self.controls)})")
        cmd_next = self._cmd + action.clamp(-1.0, 1.0) * self._step_phys
        if self.no_grad:
            with torch.no_grad():
                reward, extra, err, achieved = self._evaluate(cmd_next)
        else:
            reward, extra, err, achieved = self._evaluate(cmd_next)
        self._cmd = cmd_next
        self._err_prev, self._err = self._err, err
        self._last_extra, self._last_action = extra, action.detach()
        self._last_achieved = achieved
        self._step_count = self._step_count + 1
        done = self._step_count >= self.episode_length

        obs_pre = self._compute_obs().detach()
        achieved_info = {k: v.clone() for k, v in achieved.items()}
        achieved_info.update({g: v.clone() for g, v in self._goals.items()})
        achieved_info.update({f"{g}_cloud": v.clone() for g, v in self._goals_cloud.items()})
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset(done_ids)
        info: dict[str, Any] = {"obs_before_reset": obs_pre, "achieved": achieved_info}
        return self._compute_obs(), reward, done, info

    def render(self) -> None:
        return None

    @property
    def reward_spec(self) -> TwoBunchRewardSpec:
        return self._spec
