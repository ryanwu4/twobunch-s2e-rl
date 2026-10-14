"""Widen a trained flow checkpoint to a bigger knob space (surrogate/widen.py) and SAVE it in a form
`surrogate.train --init-ckpt` can warm-start from (same shapes as a fresh model built on the new
processed norm), then verify the widened model reproduces the original exactly at golden values of
the new knobs (and at any value: new knobs get zero columns).

Why a saved ckpt: train.py's warm start is `load_state_dict(strict=False)`, which cannot load the
old encoder.0 / feas_head.0 weights into wider layers (size mismatch). The widened state dict has
the new frame's knob_low/high buffers and keeps the ORIGINAL per-bunch scalers (the flow's output
frame), so fine-tuning starts from exactly the old predictions.

Usage (slac-rl env, CPU is fine):
  PYTHONPATH=$PWD/src python scripts/widen_flow_ckpt.py \
      --ckpt 'trained/xtcav23/checkpoints/best-*.ckpt' --sweep-set xtcav23_inj12 \
      --out trained/xtcav23_inj12_widened_2026-10-14/widened.ckpt
  (or --norm processed/<new>_norm.json to take the frame from a processed table)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from twobunch_s2e_rl.artifacts import flow_knob_space, resolve_ckpt
from twobunch_s2e_rl.knobs import KnobSpace
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.widen import widen_conditioning


@torch.no_grad()
def verify(old: TwoBunchFlow, wide: TwoBunchFlow, new: KnobSpace, n_points=8, n=512, seed=0) -> float:
    """Max relative difference of observables, old vs widened, at random old-frame points with the
    new knobs at their baseline (golden) value and at random values."""
    g = torch.Generator().manual_seed(seed)
    old_space = flow_knob_space(old)
    phys_old = old_space.denormalize(torch.rand(n_points, old_space.n, generator=g, dtype=torch.float64))
    z = (torch.randn(n, old.latent_dim, generator=g), torch.randn(n, old.latent_dim, generator=g))
    a = old.observables(old_space.normalize(phys_old).float(), n=n, z_drive=z[0], z_witness=z[1])
    worst = 0.0
    per_key = verify.per_key = {}
    for mode in ("baseline", "random"):
        cols = []
        for k in new.keys:
            if k in old_space.keys:
                cols.append(phys_old[:, old_space.index(k)])
            elif mode == "baseline" and k in new.baseline:
                cols.append(torch.full((n_points,), float(new.baseline[k]), dtype=torch.float64))
            else:
                i = new.index(k)
                cols.append(new.low[i] + (new.high[i] - new.low[i]) *
                            torch.rand(n_points, generator=g, dtype=torch.float64))
        b = wide.observables(new.normalize(torch.stack(cols, -1)).float(), n=n, z_drive=z[0], z_witness=z[1])
        for key in ("bunch_spacing", "drive_sigma_z", "witness_sigma_z", "drive_norm_emit_x",
                    "energy_difference", "T_drive", "T_witness", "p_surv"):
            # relative to the observable's scale over the probe points (near-zero values, e.g. an
            # energy difference crossing 0, would otherwise inflate a plain relative error)
            scale = a[key].abs().max().clamp_min(1e-30)
            rel = ((b[key] - a[key]).abs() / scale).max().item()
            worst = max(worst, rel)
            per_key[f"{mode}:{key}"] = max(per_key.get(f"{mode}:{key}", 0.0), rel)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep-set")
    g.add_argument("--norm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = resolve_ckpt(a.ckpt)
    old = TwoBunchFlow.load_from_checkpoint(ck, map_location="cpu").eval()
    new = KnobSpace.from_sweep_set(a.sweep_set) if a.sweep_set else KnobSpace.from_norm_json(a.norm)
    if a.norm:   # baselines for the verification only (golden injector / linac baseline)
        try:
            import json as _json
            sweep = _json.load(open(a.norm)).get("sweep_set", "xtcav23_inj12")
            new = KnobSpace.from_lists(new.keys, new.low, new.high,
                                       KnobSpace.from_sweep_set(sweep).baseline)
        except KeyError:
            pass
    wide = widen_conditioning(old, new)
    worst = verify(old, wide, new)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import lightning
    torch.save({"state_dict": wide.state_dict(), "hyper_parameters": dict(wide.hparams),
                "pytorch-lightning_version": lightning.__version__}, out)
    reloaded = TwoBunchFlow.load_from_checkpoint(str(out), map_location="cpu").eval()
    worst_reload = verify(old, reloaded, new, seed=1)
    meta = {"source_ckpt": ck, "out": str(out), "old_knob_keys": list(flow_knob_space(old).keys),
            "new_knob_keys": list(new.keys), "max_rel_diff_verify": worst,
            "max_rel_diff_after_reload": worst_reload, "per_key_after_reload": verify.per_key}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if max(worst, worst_reload) > 1e-3:
        raise SystemExit("widened model does NOT reproduce the original")


if __name__ == "__main__":
    main()
