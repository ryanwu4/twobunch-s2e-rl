"""KnobSpace: the single knob-frame object (names, physical bounds, baseline) for the pipeline.

Moved to facet2-twobunch-cnf (`twobunch_cnf.knobs.KnobSpace`, the same class, re-exported here). Every place
that maps knobs between physical units and the surrogate's normalized [0, 1] frame goes through its
`normalize` / `denormalize`. A KnobSpace is always built from what an artifact STORED (a checkpoint's
knob_low/high buffers, a processed norm json, a manifest) — never from today's knob tables, which may have
changed since the artifact was made (L2PhaseSet was widened in place from [-38, -34] to [-40, -34] after the
8-knob campaigns; see docs/architecture_plan.md).

`from_sweep_set` (a named knob table, for NEW campaigns) stays here: the tables are this repo's.
"""

from __future__ import annotations

from twobunch_cnf.knobs import KnobSpace, _is_torch  # noqa: F401


def from_sweep_set(name: str) -> KnobSpace:
    """A named knob table (datagen/sweep_params.py). Use only for NEW campaigns; existing
    artifacts carry their own frame."""
    from .datagen.sweep_params import resolve_sweep_set

    keys, low, high, baseline = resolve_sweep_set(name)
    return KnobSpace.from_lists(keys, low, high, baseline)
