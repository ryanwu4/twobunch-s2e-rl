"""Bmad-in-the-loop env for EVALUATING a trained two-bunch policy on real FACET2-S2E physics.

Mirrors `TwoBunchFlowEnv`'s observation/reward layout exactly, but each step tracks the knobs
through Bmad (via `BmadBridge`) instead of sampling the flow, then computes the SAME observables
with `per_bunch`/`inter_bunch` and the SAME `reward_spec`. So a policy trained on the surrogate
plugs in unchanged and its closed-loop behavior can be measured on the ground-truth simulator.

Eval-only: Bmad is not differentiable (no_grad), and a track is ~minutes, so this is for a
handful of episodes / operating points, not training. Bunches that scrape below `min_particles`
get neutral (campaign-mean) emittances + a near-zero survival fraction, so the survival hinge --
not a garbage emittance -- drives the policy back.
"""
from __future__ import annotations

import numpy as np
import torch

from ..datagen.sweep_params import BOUNDS_HIGH, BOUNDS_LOW
from ..surrogate.properties import inter_bunch, per_bunch
from .reward import EMIT_KEYS, KNOBS_KEY, TwoBunchRewardSpec

N_KNOB = 8
ACTION_DIM = N_KNOB
_INTER_KEYS = ("bunch_spacing", "energy_difference", "transverse_offset", "angular_misalignment")


class BmadTwoBunchEnv:
    def __init__(self, bridge, reward_spec: TwoBunchRewardSpec, *, num_envs: int = 1,
                 device: str = "cpu", episode_length: int = 24, action_scale: float = 0.05,
                 stochastic_init: bool = True, seed: int = 0, min_particles: int = 64,
                 spacing_goal: float | None = None):
        self.bridge = bridge
        self._reward_spec = reward_spec
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.episode_length = int(episode_length)
        self.action_scale = float(action_scale)
        self.stochastic_init = bool(stochastic_init)
        self.min_particles = int(min_particles)
        self.no_grad = True
        self.num_actions = ACTION_DIM
        self.num_obs = N_KNOB + reward_spec.n_obs_extra
        self._rng = torch.Generator(device=self.device).manual_seed(int(seed))
        self._lo = torch.tensor(BOUNDS_LOW, device=self.device, dtype=torch.float32)
        self._hi = torch.tensor(BOUNDS_HIGH, device=self.device, dtype=torch.float32)

        # neutral fallbacks (campaign means / spacing target) for scraped-bunch steps
        n = reward_spec.obs_norms
        self._neutral = {}
        for b in ("drive", "witness"):
            for k in ("norm_emit_x", "norm_emit_y", "norm_emit_4d"):
                key = f"{b}_{k}"
                mean = n.get(key, {}).get("mean", -5.0)
                self._neutral[key] = float(10 ** mean)   # emit keys are log10-normed
        self._spacing_target = next((t.target for t in reward_spec.terms
                                     if t.key == "bunch_spacing"), 2e-4)
        # goal-conditioning: if the policy/spec expects a "spacing_goal" obs column, inject a fixed
        # eval goal each step (defaults to the reward's spacing target when none is given).
        self._goal_conditioned = "spacing_goal" in reward_spec.obs_keys
        self._spacing_goal = float(spacing_goal) if spacing_goal is not None else self._spacing_target
        self._knobs = torch.zeros(self.num_envs, N_KNOB, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_obs_extra = torch.zeros(self.num_envs, reward_spec.n_obs_extra, device=self.device)
        self._last_achieved: dict[str, torch.Tensor] = {}
        self.reset()

    # ----- observation from one Bmad track --------------------------------
    def _per_env(self, res: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        ok = {}
        for name in ("drive", "witness"):
            arr = np.asarray(res[name], dtype=np.float32)
            if arr.shape[0] >= self.min_particles:
                pb = per_bunch(torch.from_numpy(arr).unsqueeze(0).to(self.device))
                for k, v in pb.items():
                    out[f"{name}_{k}"] = v
                ok[name] = True
            else:
                for k in ("norm_emit_x", "norm_emit_y", "norm_emit_4d"):
                    out[f"{name}_{k}"] = torch.tensor([self._neutral[f"{name}_{k}"]], device=self.device)
                ok[name] = False
        if ok["drive"] and ok["witness"]:
            d = torch.from_numpy(np.asarray(res["drive"], np.float32)).unsqueeze(0).to(self.device)
            w = torch.from_numpy(np.asarray(res["witness"], np.float32)).unsqueeze(0).to(self.device)
            out.update(inter_bunch(d, w))
        else:
            for k in _INTER_KEYS:
                out[k] = torch.tensor([self._spacing_goal if k == "bunch_spacing" else 0.0],
                                      device=self.device)
        out["T_drive"] = torch.tensor([float(res["T_drive"])], device=self.device)
        out["T_witness"] = torch.tensor([float(res["T_witness"])], device=self.device)
        return out

    def _observe(self, knobs_norm: torch.Tensor):
        phys = (self._lo + knobs_norm * (self._hi - self._lo)).cpu().numpy()
        per = []
        for i in range(self.num_envs):
            res = self.bridge.track(phys[i])
            per.append(self._per_env(res))
        obs_dict = {k: torch.cat([p[k] for p in per], dim=0) for k in per[0]}
        obs_dict[KNOBS_KEY] = knobs_norm
        if self._goal_conditioned:             # load-bearing: a goal-conditioned policy expects it
            obs_dict["spacing_goal"] = torch.full((self.num_envs,), self._spacing_goal,
                                                  device=self.device)
        reward, achieved = self._reward_spec(obs_dict)
        obs_extra = self._reward_spec.obs_scaled(obs_dict)
        return reward, obs_extra, {k: v for k, v in achieved.items()}

    def _compute_obs(self):
        return torch.cat([self._knobs, self._last_obs_extra], dim=-1)

    # ----- env API (eval subset) ------------------------------------------
    def reset(self, env_ids=None):
        if self.stochastic_init:
            self._knobs = torch.rand(self.num_envs, N_KNOB, generator=self._rng, device=self.device)
        else:
            self._knobs = torch.full((self.num_envs, N_KNOB), 0.5, device=self.device)
        self._step_count[:] = 0
        _, self._last_obs_extra, self._last_achieved = self._observe(self._knobs)
        return self._compute_obs()

    def step(self, action: torch.Tensor):
        # Bmad is non-differentiable (eval-only): detach so `self._knobs` stays a grad-free leaf
        # (the policy is called outside no_grad, so the action carries grad) -- otherwise _observe
        # can't .numpy() the knobs and a graph would accrue across steps.
        action = action.detach().to(self.device)
        self._knobs = torch.clamp(self._knobs + self.action_scale * action, 0.0, 1.0)
        reward, obs_extra, achieved = self._observe(self._knobs)
        self._last_obs_extra, self._last_achieved = obs_extra, achieved
        self._step_count += 1
        done = self._step_count >= self.episode_length
        obs_pre = self._compute_obs()
        info = {"obs_before_reset": obs_pre, "achieved": achieved}
        return self._compute_obs(), reward, done, info
