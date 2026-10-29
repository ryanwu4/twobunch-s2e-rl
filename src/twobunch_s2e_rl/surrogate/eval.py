"""Evaluate a trained two-bunch flow: moved to facet2-twobunch-cnf (`twobunch_cnf.eval`).

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.eval --ckpt trained/twobunch_flow/checkpoints/best-*.ckpt

This shim keeps the old CLI and defaults (--processed processed/twobunch_flow.h5, --out results/surrogate/default,
relative to the repo root) and evaluates ANY twobunch checkpoint: it loads with `loading.load_flow` (legacy
checkpoints through the frozen model) and hands the model to twobunch_cnf's evaluation.
Writes <--out>/{parity.png, phase_space.png, metrics.json}.
"""

from __future__ import annotations

import argparse

from twobunch_cnf.eval import PARITY_KEYS, _destd  # noqa: F401

from ..artifacts import resolve_ckpt
from ..datagen.paths import repo_root
from .loading import load_flow


def evaluate(ckpt, processed, out_dir, n=2048, device="cuda"):
    import torch
    from twobunch_cnf import eval as cnf_eval

    device = device if torch.cuda.is_available() else "cpu"
    model = ckpt if isinstance(ckpt, torch.nn.Module) else load_flow(ckpt, map_location=device)
    return cnf_eval.evaluate(model, processed, repo_root() / out_dir, n=n, device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint path (glob ok)")
    ap.add_argument("--processed", default=str(repo_root() / "processed" / "twobunch_flow.h5"))
    ap.add_argument("--out", default="results/surrogate/default")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    evaluate(resolve_ckpt(args.ckpt), args.processed, args.out, n=args.n)


if __name__ == "__main__":
    main()
