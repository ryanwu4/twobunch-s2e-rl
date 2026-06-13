"""Conditional normalizing-flow surrogate over the two-bunch beamline (NOT YET IMPLEMENTED).

Trains on the datagen campaign output (data/<name>/sample_*.json knobs + *_PENT.h5
particles) to generate the 6D drive+witness beam at PENT conditioned on the 8 sweep knobs.

Design and the reusable-ingest plan live in docs/surrogate_roadmap.md. Runs in the
`slac-rl` (torch) env once implemented.
"""

raise NotImplementedError(
    "twobunch_s2e_rl.surrogate is a stub — see docs/surrogate_roadmap.md"
)
