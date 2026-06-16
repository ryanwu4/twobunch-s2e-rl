"""MBRL on the two-bunch flow surrogate.

BPTT/SHAC agents tune the 8 sweep knobs against a composite, feasibility-gated reward
(spacing target + per-bunch emittance + survival) computed from the differentiable
`TwoBunchFlow.observables()`. Mirrors the photoinjector-rl-clean flow-MBRL stack. Runs in the
`slac-rl` (torch) env. See lab-notebook 2026-06-15 MBRL plan.
"""
