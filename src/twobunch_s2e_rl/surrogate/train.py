"""Train the two-bunch conditional-flow surrogate.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.train --config configs/surrogate/<model>.yaml
  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.train \
      --processed processed/twobunch_flow.h5 --epochs 300

A `--config` yaml sets any of the flags below by their option name (e.g. `n_layers: 16`); flags
given on the command line override it. The resolved settings + provenance go to <out>/run.json.

Checkpoints -> trained/twobunch_flow/, CSV logs alongside. Model (de)norm buffers are
loaded from the preprocess _norm.json so the checkpoint is self-contained.
"""
from __future__ import annotations

import argparse

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..datagen.paths import repo_root
from .dataset import TwoBunchFlowDataModule
from .model import TwoBunchFlow
from ..artifacts import resolve_ckpt, write_run_json
import yaml


def build_model(norm: dict, **overrides) -> TwoBunchFlow:
    kw = dict(
        condition_dim=len(norm["knob_keys"]),
        drive_mean=norm["drive_mean"], drive_std=norm["drive_std"],
        witness_mean=norm["witness_mean"], witness_std=norm["witness_std"],
        knob_low=norm["knob_low"], knob_high=norm["knob_high"],
        knob_keys=list(norm["knob_keys"]), treaty_point=norm.get("treaty_point", "PENT"),
    )
    kw.update(overrides)
    print(f"[build_model] scaler={norm.get('scaler', 'pooled(v1)')}, coupling={kw.get('coupling','affine')}, "
          f"n_bins={kw.get('n_bins')}, tail_bound={kw.get('tail_bound')}, "
          f"w_emit={kw.get('w_emit')}, w_emit_z={kw.get('w_emit_z')}, w_emit_4d={kw.get('w_emit_4d')}, "
          f"w_emit_6d={kw.get('w_emit_6d')}, w_cov={kw.get('w_cov')}")
    return TwoBunchFlow(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="yaml of flag values (option names as keys)")
    ap.add_argument("--processed", default=str(repo_root() / "processed" / "twobunch_flow.h5"))
    ap.add_argument("--out", default=str(repo_root() / "trained" / "twobunch_flow"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--coupling", choices=["affine", "rqs"], default="rqs")
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--tail-bound", type=float, default=8.0)
    ap.add_argument("--w-emit", type=float, default=0.25)        # transverse 2D (x-px, y-py)
    ap.add_argument("--w-emit-z", type=float, default=0.5)       # longitudinal 2D (z-pz / LPS)
    ap.add_argument("--w-emit-4d", type=float, default=0.125)    # transverse 4D
    ap.add_argument("--w-emit-6d", type=float, default=0.025)    # full 6D
    ap.add_argument("--w-cov", type=float, default=0.5)
    ap.add_argument("--bunches", choices=["both", "drive", "witness"], default="both",
                    help="which bunch density paths to train (witness = witness-only ablation)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau",
                    help="cosine = smooth anneal to lr/100 over --epochs (for warm-started fine-tunes)")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start weights from this checkpoint (glob ok) before a fresh fit")
    pre, _ = ap.parse_known_args()
    if pre.config:
        cfg = yaml.safe_load(open(pre.config)) or {}
        unknown = set(cfg) - {a.dest for a in ap._actions}
        if unknown:
            raise SystemExit(f"unknown keys in {pre.config}: {sorted(unknown)}")
        ap.set_defaults(**cfg)
    args = ap.parse_args()
    write_run_json(args.out, "surrogate", vars(args), parents=[args.processed] +
                   ([args.init_ckpt] if args.init_ckpt else []))

    L.seed_everything(args.seed, workers=True)
    dm = TwoBunchFlowDataModule(processed_h5=args.processed, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    dm.setup()
    bunches = {"both": (0, 1), "drive": (0,), "witness": (1,)}[args.bunches]
    model = build_model(dm.norm, lr=args.lr, n_layers=args.n_layers,
                        hidden_dim=args.hidden_dim, coupling=args.coupling,
                        n_bins=args.n_bins, tail_bound=args.tail_bound,
                        w_emit=args.w_emit, w_emit_z=args.w_emit_z, w_emit_4d=args.w_emit_4d,
                        w_emit_6d=args.w_emit_6d, w_cov=args.w_cov, bunches=bunches,
                        lr_schedule=args.lr_schedule, t_max=args.epochs)
    if args.init_ckpt:
        # warm-start weights only (new hparams + fresh optimizer/LR schedule); not a full resume
        ck = resolve_ckpt(args.init_ckpt)
        state = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"warm-started weights from {ck} (missing={len(missing)}, unexpected={len(unexpected)})")

    ckpt = ModelCheckpoint(dirpath=f"{args.out}/checkpoints", monitor="val_loss",
                           mode="min", save_top_k=1,
                           filename="best-{epoch:03d}-{val_loss:.4f}")
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        gradient_clip_val=1.0, limit_train_batches=args.limit_train_batches,
        logger=CSVLogger(args.out, name="csv"),
        callbacks=[ckpt, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
        log_every_n_steps=10)
    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {ckpt.best_model_path} (val_loss={ckpt.best_model_score})")


if __name__ == "__main__":
    main()
