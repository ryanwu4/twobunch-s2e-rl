"""Artifact helpers: checkpoint resolution, run.json provenance, and the legacy tables that fill in
what older artifacts did not record.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import time
from pathlib import Path

from .datagen.paths import repo_root

# ---- checkpoint / policy resolution ------------------------------------------------------------

_VAL_LOSS = re.compile(r"val_loss=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_REWARD = re.compile(r"reward(-?\d+(?:\.\d+)?)")


def resolve_ckpt(pattern: str) -> str:
    """A path, or a glob. For a glob, pick the best match by the number in the file name:
    lowest `val_loss=` (flow checkpoints; may be negative) or highest `reward` (policy files).
    Fails loudly when no file matches, when names carry no score, or on a tie."""
    if not any(c in pattern for c in "*?["):
        return pattern
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matches {pattern!r}")
    if len(files) == 1:
        return files[0]
    scored = []
    for f in files:
        name = Path(f).name
        m_loss, m_rew = _VAL_LOSS.search(name), _REWARD.search(name)
        if m_loss:
            scored.append((float(m_loss.group(1)), f))
        elif m_rew:
            scored.append((-float(m_rew.group(1)), f))      # higher reward is better
        else:
            raise ValueError(f"cannot rank {name!r} (no val_loss/reward in name); pass a path")
    scored.sort()
    if scored[0][0] == scored[1][0]:
        raise ValueError(f"ambiguous best match for {pattern!r}: {scored[0][1]}, {scored[1][1]}")
    return scored[0][1]


# ---- run.json provenance -----------------------------------------------------------------------

def _git(*args, cwd=None) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(path=None) -> dict:
    path = str(path or repo_root())
    # tracked changes only: untracked result folders don't change the code that ran
    return {"git_hash": _git("rev-parse", "HEAD", cwd=path),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", cwd=path))}


def facet2_s2e_hash() -> str | None:
    from .datagen.paths import facet2_root
    try:
        root = Path(facet2_root())
    except Exception:  # noqa: BLE001 - no checkout configured
        return None
    return _git("rev-parse", "HEAD", cwd=root) if root.exists() else None


def write_run_json(out_dir, kind: str, config: dict, parents=(), overwrite: bool = False,
                   **extra) -> Path:
    """Write <out_dir>/run.json: what this artifact is, how it was made, and from what.
    Refuses to overwrite an existing run.json unless overwrite=True."""
    out = Path(out_dir) / "run.json"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass overwrite=True / --overwrite to replace)")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **git_state(),
           "facet2_s2e_hash": facet2_s2e_hash(), "config": config,
           "parents": [str(p) for p in parents], **extra}
    out.write_text(json.dumps(rec, indent=2, default=str))
    return out


def read_run_json(path) -> dict:
    p = Path(path)
    return json.loads((p / "run.json" if p.is_dir() else p).read_text())


# ---- legacy tables (artifacts made before knob_keys / flow_ckpt were recorded) -------------------

# condition_dim -> (sweep set whose KEY ORDER the ckpt used, treaty point). Bounds always come from
# the ckpt's own knob_low/high buffers; only the names are filled in here. All 26-D sets share the
# `expanded` key order (they are {**EXPANDED_PARAMS, **override}).
LEGACY_KNOB_KEYS = {
    8: ("original8", "PENT"),
    23: ("xtcav23", "XTCAVF"),
    26: ("expanded", "PENT"),
}

# policy run dir name -> flow ckpt it was trained on (cfg.yaml never recorded it). Evidence: the
# configs' campaign_h5 / "Run with:" comments and the setpoint json provenance blocks.
LEGACY_POLICY_FLOW = {
    "bptt": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "bptt_dr": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "shac": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "shac_dr": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "bptt_gc": "trained/twobunch_flow_v4/checkpoints/best-epoch=460-val_loss=-0.1579.ckpt",
    "bptt_gc_combined": "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt",
    "bptt_gc_matched": "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt",
    "bptt_gc_tightbox": "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt",
    "bptt_gc_sigmaz": "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt",
}


def flow_knob_space(flow):
    """KnobSpace of a loaded TwoBunchFlow: names from its hparams (new ckpts) or the legacy table,
    bounds from its own buffers."""
    from .datagen.sweep_params import resolve_sweep_set
    from .knobs import KnobSpace
    keys = getattr(flow.hparams, "knob_keys", None)
    if not keys:
        dim = int(flow.hparams.condition_dim)
        if dim not in LEGACY_KNOB_KEYS:
            raise ValueError(f"no knob_keys stored and no legacy entry for condition_dim={dim}")
        keys = resolve_sweep_set(LEGACY_KNOB_KEYS[dim][0])[0]
    return KnobSpace.from_lists(keys, flow.knob_low.tolist(), flow.knob_high.tolist())


def flow_treaty_point(flow) -> str:
    tp = getattr(flow.hparams, "treaty_point", None)
    return tp or LEGACY_KNOB_KEYS[int(flow.hparams.condition_dim)][1]


def policy_flow_ckpt(logdir) -> str:
    """Flow ckpt a policy run used: run.json (new runs), else the legacy map."""
    logdir = Path(logdir)
    if (logdir / "run.json").exists():
        return read_run_json(logdir)["flow_ckpt"]
    if logdir.name in LEGACY_POLICY_FLOW:
        return str(repo_root() / LEGACY_POLICY_FLOW[logdir.name])
    raise KeyError(f"{logdir} has no run.json and is not in LEGACY_POLICY_FLOW; pass --flow-ckpt")
