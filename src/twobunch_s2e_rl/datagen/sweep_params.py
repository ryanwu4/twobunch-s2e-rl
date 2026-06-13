"""8-D sweep parameter table for the two-bunch LHS campaign.

All keys are setLattice() overrides applied on top of the golden two-bunch working
point (setLattice_configs/2024-10-14_twoBunch_baseline.yml). symmetricSextupoleStrengths
stays True (R-side sextupoles mirror the L-side values swept here).

Range provenance ("deck" = the 166-page "Road to two bunches" deck; page refs per
lab-notebook claude/analyses/2026-06-09_two-bunch-deck-analysis.md):

- L1PhaseSet: deck pp.165-166 narrow scans — ±2 deg around baseline is all clean
  spacing signal (~50-100 um/deg, spacing 150->400 um over -22 -> -18); breakdown only
  at the 5 deg scale. +0.5 deg margin adds early-degradation samples for the
  feasibility head.
- L2PhaseSet: the fragile axis — clean single witness spike only at -37/-36, LPS fold
  onset at -35, three-featured -34, S-folded -33. Deck recommends sampling [-38, -34]
  finer than 1 deg; range spans the usable band plus both failure modes.
- LxEnergyOffset: ±1% of each section's nominal energy gain (L1 210 MeV, L2 4.165 GeV,
  L3 5.5 GeV; setLinacsHelper), i.e. ±0.6%/±0.9%/±0.55% beam energy at BC11/BC14/BC20.
  Deck prescribes amplitudes ±1-2%; this is 3-4x the 0.25-0.3% gradient-jitter specs.
  With the baseline's energy asserts disabled these directly shift compressor energies.
- S1ELkG / S3ELkG: full hardware range (bounds.yml SCP limits 2024-08-01). Baseline
  S1EL sits at 81% of its ceiling — deck notes 100 um-spacing solutions rail sextupole
  limits, so the boundary is reachable territory. Includes sextupole-off, where the
  deck predicts large chromatic emittance growth (intentional feasibility signal).
- S2ELkG: ±50% of baseline. Hardware range (-21706, 0) is mostly off-manifold (baseline
  uses 18% of it); ±50% covers the official-lattice value (-2049.5) through 1.5x
  baseline, the documented re-optimization spread across spacing working points
  (deck pp.110-124).
- L3PhaseSet frozen at 0 (crest; operating point moved from the design -45 deg, energy
  trimmed via L3EnergyOffset instead) — user decision 2026-06-10, as is the exclusion
  of hidden RF jitter (knobs-only campaign).
"""

# key -> (low, high, baseline)  — baselines are the exact 2024-10-14 yml / defaults values
SWEEP_PARAMS = {
    "L1PhaseSet":     (-22.8,          -17.8,          -20.2889213421),
    "L2PhaseSet":     (-38.0,          -34.0,          -35.5447801603),
    "L1EnergyOffset": (-2.1e6,          2.1e6,          0.0),
    "L2EnergyOffset": (-48.5e6,         34.9e6,        -6.817565553821569e6),
    "L3EnergyOffset": (-43.3e6,         66.7e6,         1.1703527144314773e7),
    "S1ELkG":         (0.0,             2590.0,         2089.4846449653),
    "S2ELkG":         (-5931.561690604, -1977.187230201, -3954.374460403),
    "S3ELkG":         (-2625.0,         0.0,           -1087.9568814486),
}

PARAM_KEYS = list(SWEEP_PARAMS)
BOUNDS_LOW = [SWEEP_PARAMS[k][0] for k in PARAM_KEYS]
BOUNDS_HIGH = [SWEEP_PARAMS[k][1] for k in PARAM_KEYS]
BASELINE_KNOBS = {k: SWEEP_PARAMS[k][2] for k in PARAM_KEYS}
