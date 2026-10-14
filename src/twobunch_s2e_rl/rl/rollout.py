"""One deterministic policy rollout, shared by eval and validation.

`rollout` steps any DiffRL-API env (TwoBunchFlowEnv or DeltaControlEnv) with a trained actor for
`n_steps` and returns the per-step `info["achieved"]` dicts stacked as (T, B, ...) tensors, plus
actions and rewards. It starts from whatever state the env is in (call `env.reset()` first, or set
the env's start state explicitly for a validation replay).
"""
from __future__ import annotations

import torch
from torch import nn

from .diffrl.utils import RunningMeanStd


def load_policy(path: str, device: str):
    """(actor, obs_rms) from a diffrl policy pickle ([actor, obs_rms] for BPTT; SHAC adds critics)."""
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, (list, tuple)):
        loaded = [loaded]
    actor = loaded[0].to(device).eval()
    obs_rms = next((x for x in loaded if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


def actor_obs_dim(actor) -> int | None:
    return next((m.in_features for m in actor.modules() if isinstance(m, nn.Linear)), None)


@torch.no_grad()
def act(actor, obs_rms, obs: torch.Tensor) -> torch.Tensor:
    o = obs_rms.normalize(obs) if obs_rms is not None else obs
    return torch.tanh(actor(o, deterministic=True))


@torch.no_grad()
def rollout(env, actor, obs_rms, n_steps: int | None = None) -> dict[str, torch.Tensor]:
    n_steps = n_steps or env.episode_length
    nin = actor_obs_dim(actor)
    if nin is not None and nin != env.num_obs:
        raise ValueError(f"obs-dim mismatch: env.num_obs={env.num_obs}, actor expects {nin}")
    obs = env._compute_obs()
    steps, actions, rewards = [], [], []
    for _ in range(n_steps):
        a = act(actor, obs_rms, obs)
        obs, rew, done, info = env.step(a)
        steps.append(info["achieved"])
        actions.append(a)
        rewards.append(rew)
        if bool(done.any()) and len(steps) < n_steps:
            raise RuntimeError("an episode ended inside the rollout window; use n_steps <= episode_length")
    out = {k: torch.stack([s[k] for s in steps]) for k in steps[0]}
    out["action"] = torch.stack(actions)
    out["reward"] = torch.stack(rewards)
    return out
