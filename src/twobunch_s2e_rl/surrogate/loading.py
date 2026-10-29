"""One loader for every TwoBunchFlow checkpoint this repo has made.

load_flow(ckpt) returns, in eval mode:
  - the facet2-twobunch-cnf model (twobunch_cnf.model.TwoBunchFlow) when `twobunch_cnf.migrate` accepts the
    checkpoint (it records knob_keys + treaty_point and uses no option removed from twobunch_cnf main); the
    conversion happens in memory and the outputs are bit-identical to the legacy model's (gate 2026-10-28);
  - otherwise the frozen legacy model of this package (model.py), as `TwoBunchFlow.load_from_checkpoint` did.

Both expose the same sampling / observables / feasibility API, so callers need not care which they got.
"""

from __future__ import annotations

import io


def _read(path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - older checkpoints may carry non-allowlisted objects (trusted local files)
        return torch.load(str(path), map_location="cpu", weights_only=False)


def converts(path) -> bool:
    """Whether twobunch_cnf main can serve this checkpoint (after an in-memory migration)."""
    from twobunch_cnf.migrate import migrate_checkpoint

    try:
        migrate_checkpoint(_read(path), source=str(path))
    except ValueError:
        return False
    return True


def load_flow(path, map_location="cpu"):
    import torch
    from twobunch_cnf.migrate import migrate_checkpoint
    from twobunch_cnf.model import TwoBunchFlow as CNFFlow

    try:
        ck = migrate_checkpoint(_read(path), source=str(path))
    except ValueError:
        from .model import TwoBunchFlow as LegacyFlow

        return LegacyFlow.load_from_checkpoint(str(path), map_location=map_location).eval()
    buf = io.BytesIO()
    torch.save(ck, buf)
    buf.seek(0)
    return CNFFlow.load_from_checkpoint(buf, map_location=map_location).eval()
