"""Inventory of existing (gitignored) artifacts guarded by the compatibility tests, plus the
deterministic probes used to freeze their behaviour into tests/golden/.

Kept separate from the tests so `tests/golden/make_goldens.py` and `tests/test_compat.py` run the
exact same probe code. Paths are relative to the repo root.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"

# Checkpoints that ALREADY failed to load before the refactor (recorded 2026-10-14, not guarded):
# v1 predates the per-bunch whitening heads (has `whiten_head`, code expects `whiten_heads.{0,1}`).
# No policy depends on it (the 8-knob policies use v4).
KNOWN_BROKEN_CKPTS = {
    "twobunch_flow": "trained/twobunch_flow/checkpoints/best-epoch=144-val_loss=-1.3655.ckpt",
}

# trained flow checkpoints
CKPTS = {
    "twobunch_flow_v2": "trained/twobunch_flow_v2/checkpoints/best-epoch=147-val_loss=0.3784.ckpt",
    "twobunch_flow_v3": "trained/twobunch_flow_v3/checkpoints/best-epoch=139-val_loss=0.4190.ckpt",
    "twobunch_flow_v4": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "twobunch_flow_witness": "trained/twobunch_flow_witness/checkpoints/best-epoch=106-val_loss=0.8835.ckpt",
    "twobunch_combined": "trained/twobunch_combined/checkpoints/best-epoch=298-val_loss=0.9830.ckpt",
    "twobunch_combined_ft": "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt",
    "xtcav23": "trained/xtcav23/checkpoints/best-epoch=498-val_loss=0.1747.ckpt",
}

# policy run -> the flow ckpt it was trained on. cfg.yaml never recorded this; the mapping comes
# from each config's campaign_h5 (v4 for the 8-knob runs) and the setpoint provenance blocks /
# config "Run with:" comments (combined_ft for the 26-knob runs).
POLICY_FLOW = {
    "bptt": "twobunch_flow_v4",
    "bptt_dr": "twobunch_flow_v4",
    "shac": "twobunch_flow_v4",
    "shac_dr": "twobunch_flow_v4",
    "bptt_gc": "twobunch_flow_v4",
    "bptt_gc_combined": "twobunch_combined_ft",
    "bptt_gc_matched": "twobunch_combined_ft",
    "bptt_gc_tightbox": "twobunch_combined_ft",
    "bptt_gc_sigmaz": "twobunch_combined_ft",
}

NORM_JSONS = ["twobunch_flow", "twobunch_flow_v2", "twobunch_flow_v4", "twobunch_combined",
              "xtcav23_smoke", "xtcav23_pilot", "xtcav23_combined"]

NORMS_CACHES = ["metric_norms", "metric_norms_dr", "metric_norms_combined",
                "metric_norms_matched", "metric_norms_sigmaz"]

# one sample json per schema variant
SAMPLES = {
    "full_8knob": "data/full/sample_00000.json",
    "tightbox_26knob": "data/tightbox_v2_full/sample_00000.json",
    "xtcav23_23knob": "data/xtcav23_full/sample_00000.json",
}

OBS_KEYS = ("bunch_spacing", "energy_difference", "drive_norm_emit_x", "drive_norm_emit_y",
            "witness_norm_emit_x", "witness_norm_emit_y", "drive_sigma_z", "witness_sigma_z",
            "p_surv", "T_drive", "T_witness")


def path(rel: str) -> Path:
    return REPO / rel


def have(*rels: str) -> bool:
    return all(path(r).exists() for r in rels)


def load_flow(name: str):
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    return TwoBunchFlow.load_from_checkpoint(str(path(CKPTS[name])), map_location="cpu").eval()


@torch.no_grad()
def flow_probe(flow, n_points: int = 4, n: int = 256, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic surrogate outputs: fixed knob points + frozen base samples (CRN)."""
    g = torch.Generator().manual_seed(seed)
    d = int(flow.hparams.condition_dim)
    knobs = torch.cat([torch.full((1, d), 0.5), torch.rand(n_points - 1, d, generator=g)])
    z_d = torch.randn(n, flow.latent_dim, generator=g)
    z_w = torch.randn(n, flow.latent_dim, generator=g)
    obs = flow.observables(knobs, n=n, z_drive=z_d, z_witness=z_w)
    return {k: obs[k].double().numpy() for k in OBS_KEYS}


def load_policy(run: str, which: str = "best"):
    from twobunch_s2e_rl.rl.eval import _load_policy
    return _load_policy(str(path(f"logs/{run}/{which}_policy.pt")), "cpu")


@torch.no_grad()
def policy_probe(actor, obs_rms, n: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic actions on fixed random observations (catches pickle/obs_rms drift)."""
    nin = next(m.in_features for m in actor.modules() if isinstance(m, torch.nn.Linear))
    obs = torch.randn(n, nin, generator=torch.Generator().manual_seed(seed))
    o = obs_rms.normalize(obs) if obs_rms is not None else obs
    return {"obs": obs.double().numpy(),
            "action": torch.tanh(actor(o, deterministic=True)).double().numpy()}


@torch.no_grad()
def env_probe(run: str, tmp_dir: Path, n_envs: int = 4, n_steps: int = 3,
              seed: int = 0) -> dict[str, np.ndarray]:
    """Rebuild the training env from logs/<run>/cfg.yaml exactly as training did (CPU), then roll
    the best policy a few steps: freezes obs layout, reward spec, norms, knob box and goals.

    The reward-norms cache is COPIED into tmp_dir so the test can never rewrite the real cache.
    Latent RF drift is switched off here (it is tested separately against train-time values)."""
    import yaml
    from twobunch_s2e_rl.rl._train_utils import build_env_fn

    with open(path(f"logs/{run}/cfg.yaml")) as f:
        cfg = yaml.safe_load(f)
    de = cfg["params"]["diff_env"]
    de["campaign_h5"] = str(path(de["campaign_h5"]))
    cache = de.get("reward_norms_json")
    if cache:
        dst = Path(tmp_dir) / Path(cache).name
        shutil.copy(path(cache), dst)
        de["reward_norms_json"] = str(dst)
    de["rf_drift"] = {"enabled": False}
    de["rf_drift_std"] = 0.0

    torch.manual_seed(seed)
    env_fn = build_env_fn(cfg, str(path(CKPTS[POLICY_FLOW[run]])))
    env = env_fn(num_envs=n_envs, device="cpu", seed=seed,
                 episode_length=int(de.get("episode_length", 64)), stochastic_init=True,
                 no_grad=True)
    actor, obs_rms = load_policy(run)
    obs = env.reset()
    out = {"obs0": obs.double().numpy()}
    rews, acts = [], []
    for _ in range(n_steps):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        a = torch.tanh(actor(o, deterministic=True))
        obs, rew, _, _ = env.step(a)
        acts.append(a.double().numpy())
        rews.append(rew.double().numpy())
    out.update(actions=np.stack(acts), rewards=np.stack(rews), obs_last=obs.double().numpy())
    if cache:
        out["cache_unchanged"] = np.array(
            json.loads(Path(dst).read_text()) == json.loads(path(cache).read_text()))
    return out


def sample_probe(name: str, p: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    """preprocess.parse_one on one sample json of a given schema variant."""
    from twobunch_s2e_rl.surrogate.preprocess import parse_one

    rel = SAMPLES[name]
    d = json.loads(path(rel).read_text())
    keys = list(d["knobs"])
    treaty = "XTCAVF" if "XTCAVF" in d.get("specs", {}) else "PENT"
    rec = parse_one(str(path(rel)), p, np.random.default_rng(seed), keys, treaty_point=treaty)
    return {
        "knobs": np.asarray(rec["knobs"], np.float64),
        "flags": np.array([rec["drive_present"], rec["witness_viable"],
                           rec["drive_density"], rec["witness_density"]]),
        "counts": np.array([rec["n_drive"], rec["n_witness"]]),
        "charges": np.array([rec["drive_charge"], rec["witness_charge"]]),
        "drive_parts_mean": rec["drive_parts"].astype(np.float64).mean(0),
    }


def golden_path(kind: str, name: str) -> Path:
    return GOLDEN / kind / f"{name}.npz"
