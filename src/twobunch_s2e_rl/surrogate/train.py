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
import json
from pathlib import Path

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..datagen.paths import repo_root
from .dataset import SCALER_KEYS, TwoBunchFlowDataModule, orig_split_keys
from .model import TwoBunchFlow
from ..artifacts import resolve_ckpt, write_run_json
import yaml


def check_knob_frame(state: dict, norm: dict, ckpt) -> None:
    """A warm start must condition in the processed table's knob frame (else the inputs are
    silently re-scaled)."""
    keys = state.get("_knob_keys")
    if keys is not None and list(keys) != list(norm.get("knob_keys", keys)):
        raise SystemExit(f"{ckpt}: knob names differ from the processed table")
    lo, hi = state["knob_low"].numpy(), state["knob_high"].numpy()
    if lo.shape[0] != len(norm["knob_low"]) or not (
            np.allclose(lo, norm["knob_low"], rtol=1e-6, atol=0) and
            np.allclose(hi, norm["knob_high"], rtol=1e-6, atol=0)):
        raise SystemExit(f"{ckpt}: knob frame differs from the processed table "
                         f"({lo.shape[0]} vs {len(norm['knob_low'])} knobs); widen the checkpoint "
                         "to this frame first (scripts/widen_flow_ckpt.py --norm <processed>_norm.json)")


class PairSource:
    """Draws (point knobs, reference knobs, true spacing difference [m]) batches from fixed TRAIN pairs,
    crossed pairs capped at `crossed_frac` of the draws."""

    def __init__(self, kp, kr, dtrue, crossed, crossed_frac):
        self.kp, self.kr, self.d = kp, kr, dtrue
        n_c, n_k = int(crossed.sum()), int((~crossed).sum())
        share = min(crossed_frac, n_c / max(n_c + n_k, 1)) if n_k else 1.0
        w = np.where(crossed, share / max(n_c, 1), (1 - share) / max(n_k, 1))
        self.w = torch.tensor(w / w.sum(), dtype=torch.float)
        self._dev = None

    def __call__(self, b, device):
        if self._dev != device:
            self.kp, self.kr, self.d, self._dev = self.kp.to(device), self.kr.to(device), self.d.to(device), device
        i = torch.multinomial(self.w, b, replacement=True).to(device)
        return self.kp[i], self.kr[i], self.d[i]


def build_pair_source(dm, model, crossed_frac):
    """TRAIN-only, non-holdout pairs with both rows density-valid: nested / top-up rows vs their golden-injector
    partner (same pair_key), and crossed rows vs the golden crossed row of the same (campaign, linac_id)."""
    import h5py
    from .properties import inter_bunch
    ds = dm.full
    tr = set(int(i) for i in dm.train.indices)
    ok = (ds.drive_density.numpy() & ds.witness_density.numpy() & ~ds.holdout)
    use = lambda i: i in tr and ok[i]
    with h5py.File(dm.hparams.processed_h5) as h:
        key, ref = h["pair_key"][...], h["pair_ref"][...]
        blk = h["block"][...] if "block" in h else np.zeros(len(key), np.int8)
        lid = h["linac_id"][...] if "linac_id" in h else np.full(len(key), -1)
        ig = h["injector_golden"][...] if "injector_golden" in h else np.zeros(len(key), bool)
    pairs, crossed = [], []
    by = {}
    for i in range(len(key)):
        if key[i] >= 0 and use(i):
            by.setdefault(int(key[i]), {}).setdefault("ref" if ref[i] else "pt", []).append(i)
    for d in by.values():
        for r in d.get("ref", [])[:1]:
            for p in d.get("pt", []):
                pairs.append((p, r)); crossed.append(False)
    cr = {}
    for i in np.flatnonzero(blk == 1):
        if use(int(i)):
            cr.setdefault((int(ds.row_campaign[i]), int(lid[i])), []).append(int(i))
    for rows in cr.values():
        g = [i for i in rows if ig[i]]
        if g:
            for p in rows:
                if not ig[p]:
                    pairs.append((p, g[0])); crossed.append(True)
    if not pairs:
        raise SystemExit("--w-pair > 0 but no train pairs in the table")
    p_i, r_i = [p for p, _ in pairs], [r for _, r in pairs]
    sc = lambda k: (getattr(model, f"{k}_mean").cpu().double(), getattr(model, f"{k}_std").cpu().double())
    (dmu, dsd), (wmu, wsd) = sc("drive"), sc("witness")
    sp = lambda rows: inter_bunch(ds.drive[rows].double() * dsd + dmu, ds.witness[rows].double() * wsd + wmu)["bunch_spacing"]
    dtrue = (sp(p_i) - sp(r_i)).float()
    crossed = np.array(crossed)
    print(f"[pair_source] {len(pairs)} train pairs: {int((~crossed).sum())} nested/top-up, {int(crossed.sum())} crossed "
          f"(capped at {crossed_frac:.0%} of draws); true delta rms {1e6 * dtrue.std():.1f} um")
    return PairSource(ds.knobs[p_i], ds.knobs[r_i], dtrue, crossed, crossed_frac)


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
          f"w_emit_6d={kw.get('w_emit_6d')}, w_cov={kw.get('w_cov')}, w_mean={kw.get('w_mean')}, "
          f"w_spacing={kw.get('w_spacing')}")
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
    ap.add_argument("--w-mean", type=float, default=0.0,
                    help="per-bunch centroid (sample-mean) matching, standardized frame (0 = off)")
    ap.add_argument("--w-spacing", type=float, default=0.0,
                    help="Huber loss on the physical bunch spacing in units of --spacing-scale-m (0 = off)")
    ap.add_argument("--spacing-scale-m", type=float, default=1e-5)
    ap.add_argument("--w-pair", type=float, default=0.0,
                    help="paired injector-effect loss on TRAIN pairs (point vs golden injector at the same linac "
                         "knobs + crossed rows vs the golden crossed row); 0 = off")
    ap.add_argument("--pair-batch", type=int, default=32)
    ap.add_argument("--pair-particles", type=int, default=512)
    ap.add_argument("--pair-crossed-frac", type=float, default=0.3,
                    help="max share of crossed (golden-linac) pairs in the pair draws (they all share one linac point)")
    ap.add_argument("--t-two-part", action="store_true", help="two-part transmission head (P(T=1) + tail)")
    ap.add_argument("--t-sharp", type=float, default=4.0)
    ap.add_argument("--t-head-lr", type=float, default=1e-3, help="lr of the fresh two-part T head")
    ap.add_argument("--role-map", default=None,
                    help="json {'roles': {'campaign/idx': train|val|test}} from scripts/export_role_map.py: the exact "
                         "split of an earlier flow; rows not in it inherit their split group's role")
    ap.add_argument("--save-top-k", type=int, default=1)
    ap.add_argument("--bunches", choices=["both", "drive", "witness"], default="both",
                    help="which bunch density paths to train (witness = witness-only ablation)")
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stopping patience on val_loss; 0 = no early stopping (fixed-epoch fine-tunes)")
    ap.add_argument("--select-metric", default="val_loss",
                    help="metric for the best checkpoint: val_loss | val_paired_delta_rms_um | "
                         "val_old_spacing_mae_um (the last two need --split group; surrogate/selection.py)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau",
                    help="cosine = smooth anneal to lr/100 over --epochs (for warm-started fine-tunes)")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start weights from this checkpoint (glob ok) before a fresh fit; its "
                         "particle scalers are used to standardize the data (one frame) and its knob "
                         "frame must equal the processed table's (widen it first: "
                         "scripts/widen_flow_ckpt.py)")
    ap.add_argument("--orig-split", default=None, metavar="H5=CAMP1,CAMP2",
                    help="earlier table (and its campaigns, in order) the warm-start flow was trained on "
                         "with the default random split: its val rows stay in val, its train rows in train")
    ap.add_argument("--split", choices=["random", "group"], default="random",
                    help="train/val split: 'group' keeps each injector point (and its paired rows) on "
                         "one side (preprocess `group` column)")
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
    init_state, scaler = None, None
    if args.init_ckpt:
        ck = resolve_ckpt(args.init_ckpt)
        init_state = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        scaler = {k: init_state[k].tolist() for k in SCALER_KEYS}
    orig = None
    if args.role_map and args.orig_split:
        raise SystemExit("--role-map and --orig-split are exclusive")
    if args.role_map:
        orig = json.loads(Path(args.role_map).read_text())["roles"]
    if args.orig_split:
        h5, camps = args.orig_split.split("=", 1)
        orig = orig_split_keys(h5, camps.split(","))
    dm = TwoBunchFlowDataModule(processed_h5=args.processed, batch_size=args.batch_size,
                                num_workers=args.num_workers, scaler=scaler, split=args.split,
                                orig_split=orig)
    dm.setup()
    if init_state is not None:
        hp = torch.load(ck, map_location="cpu", weights_only=False).get("hyper_parameters", {})
        check_knob_frame({**init_state, "_knob_keys": hp.get("knob_keys")}, dm.norm, ck)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "norm.json").write_text(json.dumps(dm.norm, indent=2))   # plant charges etc.
    bunches = {"both": (0, 1), "drive": (0,), "witness": (1,)}[args.bunches]
    model = build_model(dm.norm, lr=args.lr, n_layers=args.n_layers,
                        hidden_dim=args.hidden_dim, coupling=args.coupling,
                        n_bins=args.n_bins, tail_bound=args.tail_bound,
                        w_emit=args.w_emit, w_emit_z=args.w_emit_z, w_emit_4d=args.w_emit_4d,
                        w_emit_6d=args.w_emit_6d, w_cov=args.w_cov, bunches=bunches,
                        w_mean=args.w_mean, w_spacing=args.w_spacing, spacing_scale_m=args.spacing_scale_m,
                        w_pair=args.w_pair, pair_batch=args.pair_batch, pair_particles=args.pair_particles,
                        t_two_part=args.t_two_part, t_sharp=args.t_sharp,
                        t_head_lr=args.t_head_lr if args.t_two_part else None,
                        lr_schedule=args.lr_schedule, t_max=args.epochs)
    if init_state is not None:
        # warm-start weights only (new hparams + fresh optimizer/LR schedule); not a full resume
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        new_ok = [k for k in missing if args.t_two_part and k.startswith("t_full_head.")]
        missing = [k for k in missing if k not in new_ok]
        if missing or unexpected:
            raise SystemExit(f"warm start from {ck} incomplete: missing={missing} unexpected={unexpected}")
        print(f"warm-started all weights from {ck}" + (f" (fresh: {len(new_ok)} t_full_head tensors)" if new_ok else ""))
    fresh_t = args.t_two_part and (init_state is None or not any(k.startswith("t_full_head.") for k in init_state))
    if fresh_t:
        # fresh head starts at the training data's P(T = 1) per bunch (a warm-started trained head keeps its bias)
        tr = np.asarray(dm.train.indices)
        ds = dm.full
        with torch.no_grad():
            for j, (frac, mask) in enumerate(((ds.drive_frac, ds.drive_present), (ds.witness_frac, ds.witness_viable))):
                m = mask[tr].numpy()
                p1 = float(np.clip((frac[tr].numpy()[m] >= 1 - 1e-4).mean(), 0.02, 0.98))
                model.t_full_head[-1].bias[j] = float(np.log(p1 / (1 - p1)))
                print(f"[t_two_part] bunch {j}: train P(T=1) = {p1:.3f}")
    if args.w_pair > 0:
        model.pair_source = build_pair_source(dm, model, args.pair_crossed_frac)

    callbacks = []
    if args.split == "group":
        from .selection import PairedDeltaMetrics
        callbacks.append(PairedDeltaMetrics(dm))
    metric = args.select_metric
    ckpt = ModelCheckpoint(dirpath=f"{args.out}/checkpoints", monitor=metric,
                           mode="min", save_top_k=args.save_top_k, save_last=True,
                           filename="best-{epoch:03d}-{" + metric + ":.4f}")
    callbacks.append(ckpt)
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=args.patience))
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        gradient_clip_val=1.0, limit_train_batches=args.limit_train_batches,
        logger=CSVLogger(args.out, name="csv"),
        callbacks=callbacks,
        log_every_n_steps=10)
    trainer.fit(model, datamodule=dm)
    # the end-of-training weights, always (save_last=True kept last.ckpt at the best epoch in the
    # 2026-10-15 fine-tunes, so the final weights were lost)
    trainer.save_checkpoint(f"{args.out}/checkpoints/final.ckpt")
    print(f"Best checkpoint: {ckpt.best_model_path} ({metric}={ckpt.best_model_score}); "
          f"final weights: {args.out}/checkpoints/final.ckpt")


if __name__ == "__main__":
    main()
