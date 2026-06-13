"""Dataset + DataModule for the two-bunch conditional-flow surrogate.

Loads processed/twobunch_flow.h5 into memory and serves per-sample dicts:
  knobs           (8,)      knobs normalized to [0,1] (from sweep bounds)
  drive_parts     (P,6)     drive cloud, per-bunch StandardScaler-standardized
  witness_parts   (P,6)     witness cloud, standardized (zeros where not density-trainable)
  drive_present / witness_viable / drive_density / witness_density   bool masks
  drive_frac / witness_frac   surviving fractions (feasibility-head targets)

Normalization mirrors the photoinjector flow: knobs min-max to [0,1] (the RL box),
particles per-dim per-bunch StandardScaler (raw scales span ~x:1e-4 m to pz:1e10 eV/c).
"""
from __future__ import annotations

import json

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _load(h, k):
    return np.asarray(h[k][...])


class TwoBunchFlowDataset(Dataset):
    def __init__(self, processed_h5: str, norm_json: str | None = None):
        with h5py.File(processed_h5, "r") as h:
            knobs = _load(h, "knobs").astype(np.float32)              # (N,8)
            drive = _load(h, "drive_parts").astype(np.float32)        # (N,P,6)
            witness = _load(h, "witness_parts").astype(np.float32)    # (N,P,6)
            masks = {k: _load(h, k).astype(bool) for k in
                     ("drive_present", "witness_viable", "drive_density", "witness_density")}
            dfrac = _load(h, "drive_frac").astype(np.float32)
            wfrac = _load(h, "witness_frac").astype(np.float32)

        norm_json = norm_json or processed_h5.replace(".h5", "_norm.json")
        with open(norm_json) as f:
            norm = json.load(f)
        self.norm = norm

        lo = np.array(norm["knob_low"], np.float32)
        hi = np.array(norm["knob_high"], np.float32)
        dm = np.array(norm["drive_mean"], np.float32)
        ds = np.array(norm["drive_std"], np.float32)
        wm = np.array(norm["witness_mean"], np.float32)
        ws = np.array(norm["witness_std"], np.float32)

        self.knobs = torch.tensor((knobs - lo) / np.maximum(hi - lo, 1e-12))
        self.drive = torch.tensor((drive - dm) / ds)
        self.witness = torch.tensor((witness - wm) / ws)
        self.drive_present = torch.tensor(masks["drive_present"])
        self.witness_viable = torch.tensor(masks["witness_viable"])
        self.drive_density = torch.tensor(masks["drive_density"])
        self.witness_density = torch.tensor(masks["witness_density"])
        self.drive_frac = torch.tensor(dfrac)
        self.witness_frac = torch.tensor(wfrac)
        self.P = int(self.drive.shape[1])
        self.cond_dim = int(self.knobs.shape[1])

    def __len__(self):
        return self.knobs.shape[0]

    def __getitem__(self, i):
        return {
            "knobs": self.knobs[i],
            "drive": self.drive[i], "witness": self.witness[i],
            "drive_present": self.drive_present[i], "witness_viable": self.witness_viable[i],
            "drive_density": self.drive_density[i], "witness_density": self.witness_density[i],
            "drive_frac": self.drive_frac[i], "witness_frac": self.witness_frac[i],
        }


class TwoBunchFlowDataModule(L.LightningDataModule):
    def __init__(self, processed_h5="processed/twobunch_flow.h5", norm_json=None,
                 batch_size=64, val_fraction=0.1, num_workers=0, split_seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.full = None
        self.train = None
        self.val = None

    def setup(self, stage=None):
        if self.full is None:
            self.full = TwoBunchFlowDataset(self.hparams.processed_h5, self.hparams.norm_json)
        n_val = int(round(len(self.full) * self.hparams.val_fraction))
        n_train = len(self.full) - n_val
        self.train, self.val = random_split(
            self.full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.hparams.split_seed))

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._dl(self.train, True)

    def val_dataloader(self):
        return self._dl(self.val, False)

    @property
    def norm(self):
        assert self.full is not None
        return self.full.norm
