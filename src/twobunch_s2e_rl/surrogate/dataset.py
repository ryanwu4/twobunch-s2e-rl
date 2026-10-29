"""Dataset + DataModule for the two-bunch flow: moved to facet2-twobunch-cnf (`twobunch_cnf.dataset`).

Re-exports its classes / functions (the same objects), plus two legacy helpers that twobunch_cnf dropped with
its --orig-split option and that this repo's scripts still use (verbatim from 7ac3617).
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from twobunch_cnf.dataset import (  # noqa: F401
    SCALER_KEYS,
    TwoBunchFlowDataModule,
    TwoBunchFlowDataset,
    _load,
    group_split,
)


def reproduce_random_split(n: int, val_fraction: float = 0.1, seed: int = 42):
    """Row indices of the validation part of the DataModule's `random` split (torch random_split
    with a seeded generator), e.g. to recover which rows an earlier flow was validated on."""
    n_val = int(round(n * val_fraction))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(perm[n - n_val:])


def orig_split_keys(processed_h5: str, campaigns: list[str], val_fraction=0.1, seed=42):
    """{"campaign/idx" -> 'train' | 'val'} for an earlier table built from `campaigns` (in order)
    whose rows carry no campaign column: the idx resets mark the campaign boundaries."""
    with h5py.File(processed_h5, "r") as h:
        idx = np.asarray(h["idx"][...])
        camp = np.asarray(h["campaign"][...]) if "campaign" in h else None
    if camp is None:
        camp = np.concatenate([[0], np.cumsum(np.diff(idx) < 0)])
    if camp.max() + 1 != len(campaigns):
        raise ValueError(f"{processed_h5}: {camp.max() + 1} campaign blocks, {len(campaigns)} names given")
    val = set(reproduce_random_split(len(idx), val_fraction, seed))
    return {f"{campaigns[c]}/{int(i)}": ("val" if r in val else "train") for r, (c, i) in enumerate(zip(camp, idx))}
