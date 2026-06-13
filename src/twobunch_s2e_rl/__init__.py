"""FACET-II two-bunch S2E pipeline: data campaign -> NF surrogate -> RL."""

__version__ = "0.1.0"

# Convenience re-export of the authoritative knob table (no heavy deps).
from .datagen.sweep_params import (  # noqa: F401
    SWEEP_PARAMS,
    PARAM_KEYS,
    BOUNDS_LOW,
    BOUNDS_HIGH,
    BASELINE_KNOBS,
)
