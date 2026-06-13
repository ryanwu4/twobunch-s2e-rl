"""Train the two-bunch conditional-flow surrogate.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.train \
      --processed processed/twobunch_flow.h5 --epochs 300

Checkpoints -> trained/twobunch_flow/, CSV logs alongside. Model (de)norm buffers are
loaded from the preprocess _norm.json so the checkpoint is self-contained.
"""
from __future__ import annotations

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..datagen.paths import repo_root
from .dataset import TwoBunchFlowDataModule
from .model import TwoBunchFlow


def build_model(norm: dict, **overrides) -> TwoBunchFlow:
    kw = dict(
        condition_dim=len(norm["knob_keys"]),
        drive_mean=norm["drive_mean"], drive_std=norm["drive_std"],
        witness_mean=norm["witness_mean"], witness_std=norm["witness_std"],
        knob_low=norm["knob_low"], knob_high=norm["knob_high"],
    )
    kw.update(overrides)
    return TwoBunchFlow(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default=str(repo_root() / "processed" / "twobunch_flow.h5"))
    ap.add_argument("--out", default=str(repo_root() / "trained" / "twobunch_flow"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--w-emit", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    dm = TwoBunchFlowDataModule(processed_h5=args.processed, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    dm.setup()
    model = build_model(dm.norm, lr=args.lr, n_layers=args.n_layers,
                        hidden_dim=args.hidden_dim, w_emit=args.w_emit)

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
