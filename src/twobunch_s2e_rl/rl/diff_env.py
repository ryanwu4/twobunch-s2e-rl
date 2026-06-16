"""Differentiable batched env over the two-bunch flow surrogate, for SHAC/BPTT.

Implements the same env API as the photoinjector `DiffPhotoinjectorEnv` (consumed by
`diffrl.{SHAC,BPTT}`), adapted to 8 knobs and the two-bunch `TwoBunchFlow.observables()`:

State per env i: 8 commanded knobs in [0,1] (the integrator). Action: 8-D delta in [-1,1];
clamped, scaled by `action_scale`, added to the knobs, clamped back to [0,1]. Reward: a
composite `TwoBunchRewardSpec` over the surrogate observables (spacing target + per-bunch
emittance + survival hinge). Obs: concat(commanded knobs, z-scored achieved observables).
Episodes end at `episode_length`; on `done` the env auto-resets.

Gradient design (mirrors FlowBunchEnv):
- The dynamics gradient flows through the REWARD via the grad-attached `_knobs` integrator
  (action_t affects reward_{t+}); the surrogate is frozen (requires_grad_(False)) but called
  without no_grad so input grads propagate. Only the actor learns.
- The achieved-observable block in the obs is DETACHED (it is an expensive flow-sampling graph;
  retaining it across `steps_num` would OOM and adds no needed signal -- the reward path
  already carries the gradient). `_knobs` in the obs stays attached (cheap linear integrator).

Domain randomization:
- `stochastic_init=True` -> random starting knobs each episode (the active DR).
- `rf_drift_std>0` -> a hidden PER-EPISODE additive offset on the RF-subset knobs (L1/L2 phase,
  L1/L2/L3 energy), unobserved by the policy: `knobs_actual = clamp(knobs_cmd + drift, 0,1)` is
  fed to the flow, but `knobs_cmd` is the state/obs. No surrogate retraining (drift stays in the
  trained [0,1]^8 box). Default 0 -> collapses exactly to the deterministic env.
"""
from __future__ import annotations

from typing import Any

import torch

from ..surrogate.model import TwoBunchFlow
from .reward import TwoBunchRewardSpec

N_KNOB = 8
ACTION_DIM = N_KNOB
# RF-related knob indices in the sweep_params order (L1Phase, L2Phase, L1/L2/L3 EnergyOffset).
RF_DRIFT_IDX = (0, 1, 2, 3, 4)


class TwoBunchFlowEnv:
    """Differentiable batched env over the two-bunch flow surrogate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        render: bool = False,
        seed: int = 0,
        episode_length: int = 64,
        stochastic_init: bool = True,
        MM_caching_frequency: int = 1,   # unused; API compat
        no_grad: bool = False,
        *,
        flow_ckpt: str | None = None,
        flow: TwoBunchFlow | None = None,
        reward_spec: TwoBunchRewardSpec | None = None,
        n_particles: int = 512,
        action_scale: float = 0.05,
        rf_drift_std: float = 0.0,
        common_random_numbers: bool = False,
    ):
        del MM_caching_frequency, render
        if reward_spec is None:
            raise ValueError("pass a `reward_spec` (build via rl.reward.reward_spec_from_campaign)")
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.device = torch.device(device)
        self.stochastic_init = bool(stochastic_init)
        self.no_grad = bool(no_grad)
        self.action_scale = float(action_scale)
        self.rf_drift_std = float(rf_drift_std)
        self.common_random_numbers = bool(common_random_numbers)
        self._n = int(n_particles)
        self._reward_spec = reward_spec

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed))

        # ---- load + freeze the flow ----------------------------------------
        if flow is None:
            if flow_ckpt is None:
                raise ValueError("pass `flow` (instance) or `flow_ckpt` (path)")
            flow = TwoBunchFlow.load_from_checkpoint(str(flow_ckpt), map_location=self.device)
        flow = flow.to(self.device).eval()
        for p in flow.parameters():
            p.requires_grad_(False)
        self._flow = flow

        self.num_actions = ACTION_DIM
        self.num_obs = N_KNOB + self._reward_spec.n_obs_extra

        # ---- state buffers --------------------------------------------------
        z = lambda *s: torch.zeros(*s, device=self.device)
        self._knobs = z(self.num_envs, N_KNOB)                 # commanded (grad leaf)
        self._drift = z(self.num_envs, N_KNOB)                 # hidden, detached
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_obs_extra = z(self.num_envs, self._reward_spec.n_obs_extra)  # detached
        self._last_achieved: dict[str, torch.Tensor] = {}
        self._rf_idx = torch.tensor(RF_DRIFT_IDX, device=self.device)

        self.reset()

    # ----- internals --------------------------------------------------------

    def _uniform(self, n: int) -> torch.Tensor:
        return torch.rand(n, N_KNOB, generator=self._rng, device=self.device)

    def _sample_drift(self, n: int) -> torch.Tensor:
        d = torch.zeros(n, N_KNOB, device=self.device)
        if self.rf_drift_std > 0:
            noise = torch.randn(n, len(RF_DRIFT_IDX), generator=self._rng,
                                device=self.device) * self.rf_drift_std
            d[:, self._rf_idx] = noise
        return d

    def _observe(self, knobs_cmd: torch.Tensor):
        """Run the flow at the actual (drifted) knobs -> (reward (B,), obs_extra (B,k) detached,
        achieved dict detached). Keeps grad on `reward` iff knobs_cmd has grad and not no_grad."""
        knobs_act = torch.clamp(knobs_cmd + self._drift, 0.0, 1.0)
        obs_dict = self._flow.observables(knobs_act, n=self._n)
        obs_dict["knobs"] = knobs_act          # for the trust-region (boundary) reward term
        reward, achieved = self._reward_spec(obs_dict)
        obs_extra = self._reward_spec.obs_scaled(obs_dict).detach()
        return reward, obs_extra, {k: v.detach() for k, v in achieved.items()}

    def _compute_obs(self) -> torch.Tensor:
        # commanded knobs (attached integrator) + detached z-scored achieved block
        return torch.cat([self._knobs, self._last_obs_extra], dim=-1)

    # ----- DiffRL env API ---------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        if n == 0:
            return self._compute_obs()

        if self.stochastic_init:
            new_knobs = self._uniform(n)
        else:
            new_knobs = torch.full((n, N_KNOB), 0.5, device=self.device)
        new_drift = self._sample_drift(n)

        if n == self.num_envs:
            self._knobs = new_knobs
            self._drift = new_drift
        else:
            # _knobs is grad-attached -> torch.where grad-safety for survivors (mirror pattern).
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            knobs_full = self._knobs.detach().clone()
            knobs_full[env_ids] = new_knobs
            self._knobs = torch.where(mask.unsqueeze(-1), knobs_full, self._knobs)
            self._drift[env_ids] = new_drift   # detached buffer -> plain assign

        self._step_count[env_ids] = 0

        with torch.no_grad():
            r, obs_extra, achieved = self._observe(self._knobs.detach())
        if n == self.num_envs:
            self._last_obs_extra = obs_extra
            self._last_achieved = achieved
        else:
            self._last_obs_extra[env_ids] = obs_extra[env_ids]
            for k, v in achieved.items():
                if k in self._last_achieved:
                    self._last_achieved[k][env_ids] = v[env_ids]
                else:
                    self._last_achieved[k] = v
        return self._compute_obs()

    def initialize_trajectory(self) -> torch.Tensor:
        """Cut the graph between epochs; return current obs."""
        self._knobs = self._knobs.detach()
        self._last_obs_extra = self._last_obs_extra.detach()
        return self._compute_obs()

    def clear_grad(self) -> None:
        self._knobs = self._knobs.detach()
        self._last_obs_extra = self._last_obs_extra.detach()

    def step(self, action: torch.Tensor):
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"action shape {tuple(action.shape)} != "
                             f"({self.num_envs}, {ACTION_DIM})")
        knobs_next = torch.clamp(self._knobs + self.action_scale * action, 0.0, 1.0)

        if self.no_grad:
            with torch.no_grad():
                reward, obs_extra, achieved = self._observe(knobs_next)
        else:
            reward, obs_extra, achieved = self._observe(knobs_next)

        self._knobs = knobs_next
        self._last_obs_extra = obs_extra
        self._last_achieved = achieved
        self._step_count = self._step_count + 1
        done = self._step_count >= self.episode_length

        obs_pre = self._compute_obs().detach()   # SHAC terminal bootstrap reads this
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset(done_ids)

        info: dict[str, Any] = {"obs_before_reset": obs_pre, "achieved": achieved}
        return self._compute_obs(), reward, done, info

    def render(self) -> None:
        return None

    # ----- eval helper ------------------------------------------------------

    @property
    def reward_spec(self) -> TwoBunchRewardSpec:
        return self._reward_spec
