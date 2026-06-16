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
- `rf_drift_std` -> a hidden PER-EPISODE additive offset on the RF-subset knobs (L1/L2 phase,
  L1/L2/L3 energy), unobserved by the policy: `knobs_actual = clamp(knobs_cmd + drift, 0,1)` is
  fed to the flow, but `knobs_cmd` stays the state/obs. The policy must thus be robust to an
  unknown-but-fixed-per-episode RF offset (the physical regime of slow RF drift). No surrogate
  retraining: the clamp keeps `knobs_actual` inside the trained [0,1]^8 box.

  `rf_drift_std` accepts a scalar (applied uniformly to RF_DRIFT_IDX in NORMALIZED units; 0 ->
  no drift, collapses to the deterministic env) OR a length-8 per-knob normalized-std vector.
  Use `rf_drift_std_vector(phase_drift_deg, amp_drift_frac)` to build the vector from PHYSICAL
  jitter specs. With the default 0.5 deg / 0.5% amplitude the per-knob normalized 1-sigma stds are:

      knob              phys 1-sigma            box width        normalized std
      L1PhaseSet        0.5 deg                 5.0 deg          0.100
      L2PhaseSet        0.5 deg                 4.0 deg          0.125
      L1EnergyOffset    0.5% * 210 MeV          4.2e6 eV (+-1%)  0.250
      L2EnergyOffset    0.5% * 4.165 GeV        8.34e7 eV        0.250
      L3EnergyOffset    0.5% * 5.5 GeV          1.10e8 eV        0.250
      S{1,2,3}ELkG      -- (not RF) --                           0

  NB: 0.5% amplitude is a QUARTER of the narrow +-1% energy-offset box (~2x the deck's
  0.25-0.3% gradient-jitter spec) -- a deliberately stress-y "to start" value; dial
  `amp_drift_frac` down for nominal jitter. Amplitude jitter is mapped to on-crest energy-gain
  jitter (first order; the second-order phase->energy coupling is left to the PhaseSet drift).
"""
from __future__ import annotations

from typing import Any

import torch

from ..datagen.sweep_params import PARAM_KEYS, SWEEP_PARAMS
from ..surrogate.model import TwoBunchFlow
from .reward import TwoBunchRewardSpec

N_KNOB = 8
ACTION_DIM = N_KNOB
# RF-related knob indices in the sweep_params order (L1Phase, L2Phase, L1/L2/L3 EnergyOffset).
RF_DRIFT_IDX = (0, 1, 2, 3, 4)
RF_PHASE_KEYS = ("L1PhaseSet", "L2PhaseSet")
# nominal section energy gains (eV) for the amplitude -> energy-offset drift mapping
# (L1 210 MeV, L2 4.165 GeV, L3 5.5 GeV; setLinacsHelper, per sweep_params docstring).
RF_NOMINAL_ENERGY_EV = {
    "L1EnergyOffset": 210e6,
    "L2EnergyOffset": 4.165e9,
    "L3EnergyOffset": 5.5e9,
}


def rf_drift_std_vector(phase_drift_deg: float = 0.5,
                        amp_drift_frac: float = 0.005) -> torch.Tensor:
    """Per-knob 1-sigma latent RF drift in NORMALIZED [0,1] knob units (length N_KNOB).

    Maps physical RF jitter onto the swept-knob box:
    - phase knobs (L1/L2 PhaseSet): `phase_drift_deg` deg / (box width in deg)
    - energy-offset knobs (L1/L2/L3): `amp_drift_frac` * nominal section energy gain (eV)
      / (box width in eV) -- treats RF amplitude jitter as on-crest energy-gain jitter
    - sextupole knobs (S1/S2/S3 ELkG): 0 (not RF).

    See the module docstring for the worked default values (0.5 deg / 0.5% -> L1Phase 0.100,
    L2Phase 0.125, each energy ~0.250)."""
    std = [0.0] * N_KNOB
    for j, k in enumerate(PARAM_KEYS):
        lo, hi, _ = SWEEP_PARAMS[k]
        width = float(hi - lo)
        if k in RF_PHASE_KEYS:
            std[j] = phase_drift_deg / width
        elif k in RF_NOMINAL_ENERGY_EV:
            std[j] = (amp_drift_frac * RF_NOMINAL_ENERGY_EV[k]) / width
    return torch.tensor(std, dtype=torch.float32)


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
        rf_drift_std: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
        common_random_numbers: bool = False,
        spacing_goal_lo: float | None = None,
        spacing_goal_hi: float | None = None,
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
        self._drift_std = self._build_drift_std(rf_drift_std)   # (N_KNOB,) normalized, or None
        self.common_random_numbers = bool(common_random_numbers)
        self._n = int(n_particles)
        self._reward_spec = reward_spec
        # goal-conditioning: per-episode target spacing (physical meters) sampled in [lo, hi].
        # When unset the env is byte-identical to the fixed-target env (no goal in obs/reward).
        self._goal_conditioned = spacing_goal_lo is not None and spacing_goal_hi is not None
        self._goal_lo = float(spacing_goal_lo) if self._goal_conditioned else 0.0
        self._goal_hi = float(spacing_goal_hi) if self._goal_conditioned else 0.0

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
        self._spacing_goal = z(self.num_envs)                  # (B,) per-episode goal, detached
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_obs_extra = z(self.num_envs, self._reward_spec.n_obs_extra)  # detached
        self._last_achieved: dict[str, torch.Tensor] = {}

        self.reset()

    # ----- internals --------------------------------------------------------

    def _build_drift_std(self, rf_drift_std) -> torch.Tensor | None:
        """Normalize the `rf_drift_std` arg into an (N_KNOB,) device tensor of per-knob 1-sigma
        normalized stds, or None when there is no drift. Accepts a scalar (applied uniformly to
        RF_DRIFT_IDX, back-compat) or a length-N_KNOB vector (e.g. `rf_drift_std_vector(...)`)."""
        if rf_drift_std is None:
            return None
        if torch.is_tensor(rf_drift_std) or isinstance(rf_drift_std, (list, tuple)):
            v = torch.as_tensor(rf_drift_std, dtype=torch.float32, device=self.device).flatten()
            if v.numel() != N_KNOB:
                raise ValueError(f"rf_drift_std vector must have {N_KNOB} entries, got {v.numel()}")
            return v if float(v.abs().sum()) > 0 else None
        s = float(rf_drift_std)
        if s <= 0:
            return None
        v = torch.zeros(N_KNOB, device=self.device)
        v[list(RF_DRIFT_IDX)] = s
        return v

    def _uniform(self, n: int) -> torch.Tensor:
        return torch.rand(n, N_KNOB, generator=self._rng, device=self.device)

    def _sample_drift(self, n: int) -> torch.Tensor:
        if self._drift_std is None:
            return torch.zeros(n, N_KNOB, device=self.device)
        return torch.randn(n, N_KNOB, generator=self._rng,
                           device=self.device) * self._drift_std

    def _sample_goal(self, n: int) -> torch.Tensor:
        """(n,) per-episode target spacing in physical meters, uniform in [goal_lo, goal_hi].
        Zeros (unused) when not goal-conditioned. Uses `_rng` for CRN-reproducibility."""
        if not self._goal_conditioned:
            return torch.zeros(n, device=self.device)
        u = torch.rand(n, generator=self._rng, device=self.device)
        return self._goal_lo + u * (self._goal_hi - self._goal_lo)

    def _observe(self, knobs_cmd: torch.Tensor):
        """Run the flow at the actual (drifted) knobs -> (reward (B,), obs_extra (B,k) detached,
        achieved dict detached). Keeps grad on `reward` iff knobs_cmd has grad and not no_grad."""
        knobs_act = torch.clamp(knobs_cmd + self._drift, 0.0, 1.0)
        obs_dict = self._flow.observables(knobs_act, n=self._n)
        obs_dict["knobs"] = knobs_act          # for the trust-region (boundary) reward term
        if self._goal_conditioned:             # per-env target spacing (detached constant)
            obs_dict["spacing_goal"] = self._spacing_goal   # seen by both obs_scaled and the reward term
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
        new_goal = self._sample_goal(n)

        if n == self.num_envs:
            self._knobs = new_knobs
            self._drift = new_drift
            self._spacing_goal = new_goal
        else:
            # _knobs is grad-attached -> torch.where grad-safety for survivors (mirror pattern).
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            knobs_full = self._knobs.detach().clone()
            knobs_full[env_ids] = new_knobs
            self._knobs = torch.where(mask.unsqueeze(-1), knobs_full, self._knobs)
            self._drift[env_ids] = new_drift          # detached buffer -> plain assign
            self._spacing_goal[env_ids] = new_goal    # detached buffer -> plain assign

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
        # Snapshot achieved BEFORE any reset: a partial reset mutates self._last_achieved
        # (== this dict) in place, which would corrupt info["achieved"] for survivors.
        achieved_info = {k: v.clone() for k, v in achieved.items()}
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.reset(done_ids)

        info: dict[str, Any] = {"obs_before_reset": obs_pre, "achieved": achieved_info}
        return self._compute_obs(), reward, done, info

    def render(self) -> None:
        return None

    # ----- eval helper ------------------------------------------------------

    @property
    def reward_spec(self) -> TwoBunchRewardSpec:
        return self._reward_spec
