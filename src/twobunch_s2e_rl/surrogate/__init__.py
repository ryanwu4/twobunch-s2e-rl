"""Conditional normalizing-flow surrogate over the two-bunch beam: now facet2-twobunch-cnf.

The CNF lives in its own repo, facet2-twobunch-cnf (import `twobunch_cnf`,
/home/rwu4/photoinjector-rl/facet2-twobunch-cnf); model changes and surrogate experiments happen there
(experiments on its worktree branches). This package keeps twobunch-s2e-rl working on top of it:

  loading.load_flow(ckpt)   any twobunch checkpoint: the twobunch_cnf model when the checkpoint converts
                            (migrate; bit-identical outputs), else the frozen legacy model (model.py)
  model.py, widen.py        FROZEN legacy copy of the unpruned TwoBunchFlow (@ 7ac3617) for checkpoints
                            twobunch_cnf main cannot load (no knob_keys, or removed options in use)
  properties, selection     are twobunch_cnf's modules (aliases)
  dataset                   twobunch_cnf's, plus the legacy split helpers
  train, eval, plot_training  shims onto twobunch_cnf's CLIs with this repo's old defaults
  preprocess, diagnostics, lps_scraping, eval_r2_diagnostics   this repo's own (campaign / study tools)

Design: lab-notebook 2026-06-09_nf-two-bunch-surrogate.md (v3).
"""

from twobunch_cnf import BUNCHES, COORD_KEYS, DEFAULT_P, ELECTRON_MC2_EV, LATENT_DIM

__all__ = ["COORD_KEYS", "LATENT_DIM", "ELECTRON_MC2_EV", "DEFAULT_P", "BUNCHES"]
