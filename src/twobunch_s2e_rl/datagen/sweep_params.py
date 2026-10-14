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
    "L2PhaseSet":     (-40.0,          -34.0,          -35.5447801603), #changed bc close to rail on 300um
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


# ----------------------------------------------------------------------------------------
# Expanded 26-D set: the original 8 longitudinal knobs + the transverse final-focus system
# (6 FF quads, 4 FF kickers, 8 BC20 sextupole movers). These were held FIXED at the golden
# two-bunch working point in the original campaign; this set unfreezes them so the surrogate
# and RL controller become relevant to the transverse objectives (PENT matching / BMAG,
# driver-witness collinearity). The golden baseline is already transversely tuned (non-zero
# movers/kickers), so the baselines below are those tuned values, not zero.
#
# Range provenance:
# - FF quads (Q5FF..Q0FF, kG.m): symmetric about golden by the per-quad excursion from the
#   2026-06-19 PENT beta-matching solve (golden beta*=0.5 m -> ~7.6 cm round-waist floor;
#   results/ff_quad_range/), mirrored to both sides and clipped to the
#   EPICS bounds (none clip at present). baseline = golden. Dogleg Q0D/Q1D/Q2D excluded
#   (downstream of PENT). NOTE: independent box-LHS over these lands mostly off the matched
#   manifold -> mostly mismatched PENT beams; the pilot gates whether to switch to a
#   manifold-anchored sampler for the full run.
# - FF kickers (XC1/XC3/YC1/YC2 FF, kG.m): +/-0.01 (PROVISIONAL -- FACET had no firm number;
#   golden already uses ~+/-0.005, so this is ~2x golden ~ +/-90 um centroid at PENT;
#   hardware rails are +/-0.37). Confirm true range with Zack / Ryan L.
# - Sextupole movers (S1EL/S1ER/S2EL/S2ER x/y, meters): golden baseline +/- 1.5 mm (FACET
#   cited ~1.5 mm mover travel). Applied independently of symmetricSextupoleStrengths.
# Transverse-wake coefficients (first-order for these knobs' objectives) are still pending
# validation (2026-06-10 audit) -- results from this set are provisional until then.
# ----------------------------------------------------------------------------------------
SWEEP_PARAMS_EXPANDED_EXTRA = {
    # FF telescope quads (kG.m) -- symmetric about golden by the derived golden->7.6cm
    # excursion (clipped to EPICS bounds; none clip at present). baseline = golden.
    "Q5FFkG":       (-103.287,  -40.387,  -71.837),
    "Q4FFkG":       (-133.581,  -28.921,  -81.251),
    "Q3FFkG":       (  81.146,  117.304,   99.225),
    "Q2FFkG":       ( 106.539,  146.161,  126.350),
    "Q1FFkG":       (-242.185, -228.251, -235.218),
    "Q0FFkG":       ( 108.703,  144.003,  126.353),
    # FF steering kickers (kG.m) -- provisional +/-0.01 (baseline = golden tuned value)
    "XC1FFkG":      (-0.01, 0.01,  0.0023982219),
    "XC3FFkG":      (-0.01, 0.01,  0.0015344214),
    "YC1FFkG":      (-0.01, 0.01, -0.0053321184),
    "YC2FFkG":      (-0.01, 0.01, -0.0035512385),
    # BC20 sextupole movers (m) -- golden tuned value +/- 1.5 mm (FACET cited ~1.5 mm travel)
    "S1EL_xOffset": (-0.0005931732, 0.0024068268,  0.0009068268),
    "S1EL_yOffset": (-0.0013684246, 0.0016315754,  0.0001315754),
    "S2EL_xOffset": (-0.0018885830, 0.0011114170, -0.0003885830),
    "S2EL_yOffset": (-0.0013834225, 0.0016165775,  0.0001165775),
    "S2ER_xOffset": (-0.0016679111, 0.0013320889, -0.0001679111),
    "S2ER_yOffset": (-0.0029881668, 0.0000118332, -0.0014881668),
    "S1ER_xOffset": (-0.0001676698, 0.0028323302,  0.0013323302),
    "S1ER_yOffset": (-0.0026182646, 0.0003817354, -0.0011182646),
}

EXPANDED_PARAMS = {**SWEEP_PARAMS, **SWEEP_PARAMS_EXPANDED_EXTRA}


# ----------------------------------------------------------------------------------------
# Manifold-anchored 26-D set: same knobs as `expanded`, but the transverse knobs are sampled
# near the deliverable manifold (build_manifest + datagen/ff_manifold.py; rationale in
# lab-notebook/claude/analyses/2026-06-21_manifold-anchoring-strategy.md). The FF quads are
# drawn on the matched beta-curve (build_manifest overwrites them via MANIFOLD_SPECS), so the
# FF entries here keep the `expanded` envelope only for clipping/baseline. The movers/kickers
# are NARROWED about their golden tuned values: offset is controllable in a ~0.15 mm basin, so
# +/-1.5 mm buries the collinear region (pilot: median offset 935 um, 0% < 10 um). A ~30%
# wide-box stratification tail (the `expanded` ranges) is added by build_manifest so the
# feasibility head still sees scraped/mismatched beams.
_ANCHORED_OVERRIDE = {
    # FF quads: declared range = EPICS bounds (the hard clip the curve sampler uses; jitter can
    # push a hair past the symmetric envelope). baseline = golden. The anchor block overwrites
    # these with the matched-curve draws; the wide tail draws FF from the `expanded` envelope.
    "Q5FFkG": (-256.0, 0.0, -71.837),
    "Q4FFkG": (-446.0, 0.0, -81.251),
    "Q3FFkG": (0.0, 457.0, 99.225),
    "Q2FFkG": (0.0, 167.0, 126.350),
    "Q1FFkG": (-257.0, 0.0, -235.218),
    "Q0FFkG": (0.0, 167.0, 126.353),
    # BC20 sextupole movers (m) -- golden +/- 0.15 mm
    "S1EL_xOffset": (0.0007568268, 0.0010568268, 0.0009068268),
    "S1EL_yOffset": (-0.0000184246, 0.0002815754, 0.0001315754),
    "S2EL_xOffset": (-0.0005385830, -0.0002385830, -0.0003885830),
    "S2EL_yOffset": (-0.0000334225, 0.0002665775, 0.0001165775),
    "S2ER_xOffset": (-0.0003179111, -0.0000179111, -0.0001679111),
    "S2ER_yOffset": (-0.0016381668, -0.0013381668, -0.0014881668),
    "S1ER_xOffset": (0.0011823302, 0.0014823302, 0.0013323302),
    "S1ER_yOffset": (-0.0012682646, -0.0009682646, -0.0011182646),
    # FF steering kickers (kG.m) -- golden +/- 0.005
    "XC1FFkG": (-0.0026017781, 0.0073982219, 0.0023982219),
    "XC3FFkG": (-0.0034655786, 0.0065344214, 0.0015344214),
    "YC1FFkG": (-0.0103321184, -0.0003321184, -0.0053321184),
    "YC2FFkG": (-0.0085512385, 0.0014487615, -0.0035512385),
}
EXPANDED_ANCHORED_PARAMS = {**EXPANDED_PARAMS, **_ANCHORED_OVERRIDE}


# ----------------------------------------------------------------------------------------
# Tightened-box 26-D set: a plain box-LHS set (no manifold sampler) whose transverse-matching
# knobs are tightened to the matched region found by the 2026-06-21 beam-based beta* scan
# (analysis/2026-06-21_beam-based-matching-and-anchoring.md; curve = results/beam_matching/beam_matched_curve.csv,
# beta* 7.6-50 cm, witness BMAG ~1.0 throughout). This tests whether a simple tight box -- centered
# on the MATCHED region, not golden (golden is a mismatched witness) -- gives acceptable near-matched
# coverage, which would let us skip the anchoring machinery (the A/B vs `expanded_anchored`).
#
# - FF quads + S1/S2/S3 sextupole STRENGTHS: scan [min,max] over beta* + ~margin. Several are
#   offset from golden (Q0FF +12%, Q2FF -5%, S2EL -4%); the box is centered on the matched band.
#   The FF is loosely constrained (degenerate) so its band is wider; the sextupoles are the tight
#   beta*-determining lever.
# - FF kickers / sextupole movers: tight about golden (the matched config used ~golden values;
#   offset/collinearity is a separate objective handled by the movers + offset_floor).
# - Longitudinal phases/energies (L1/L2 phase, L1/L2/L3 energy): left at the `expanded` ranges so
#   the controller still explores spacing/energy. NOTE: the matched band was found at the golden
#   longitudinal point; if the wide longitudinal variation breaks the match, the pilot will show it
#   (then narrow longitudinal to isolate, or widen the transverse band).
# Baselines = the 15 cm matched point (FF + sextupole strengths) / golden (kickers, movers).
# ----------------------------------------------------------------------------------------
_TIGHTBOX_OVERRIDE = {
    # FF quads (kG.m) -- scan matched band + margin (baseline = 15 cm matched)
    "Q5FFkG": (-95.0, -63.0, -81.740),
    "Q4FFkG": (-90.0, -74.0, -83.006),
    "Q3FFkG": (96.0, 108.0, 101.154),
    "Q2FFkG": (113.0, 125.0, 121.004),
    "Q1FFkG": (-239.0, -230.0, -232.744),
    "Q0FFkG": (134.0, 150.0, 142.275),
    # BC20 sextupole strengths (kG.m) -- the tight beta*-determining lever (baseline = 15 cm matched)
    "S1ELkG": (2090.0, 2300.0, 2164.438),
    "S2ELkG": (-4280.0, -3960.0, -4161.751),
    "S3ELkG": (-1090.0, -925.0, -949.754),
    # FF kickers (kG.m) -- tight about golden (+/- 0.003)
    "XC1FFkG": (-0.0006018, 0.0053982, 0.0023982),
    "XC3FFkG": (-0.0014656, 0.0045344, 0.0015344),
    "YC1FFkG": (-0.0083321, -0.0023321, -0.0053321),
    "YC2FFkG": (-0.0065512, -0.0005512, -0.0035512),
    # BC20 sextupole movers (m) -- tight about golden (+/- 0.1 mm)
    "S1EL_xOffset": (0.0008068, 0.0010068, 0.0009068),
    "S1EL_yOffset": (0.0000316, 0.0002316, 0.0001316),
    "S2EL_xOffset": (-0.0004886, -0.0002886, -0.0003886),
    "S2EL_yOffset": (0.0000166, 0.0002166, 0.0001166),
    "S2ER_xOffset": (-0.0002679, -0.0000679, -0.0001679),
    "S2ER_yOffset": (-0.0015882, -0.0013882, -0.0014882),
    "S1ER_xOffset": (0.0012323, 0.0014323, 0.0013323),
    "S1ER_yOffset": (-0.0012183, -0.0010183, -0.0011183),
}
TIGHTBOX_PARAMS = {**EXPANDED_PARAMS, **_TIGHTBOX_OVERRIDE}

# ----------------------------------------------------------------------------------------
# XTCAV-entrance 23-D set (GPSR knob-delta fitting, Stage B — plan 2026-07-30): conditioning
# knobs for the surrogate retrained to emit at the XTCAVF handoff plane (saved at XTCAVF,
# element EXIT with V=0 ≡ entrance + exact 1.0334 m drift; back-drifted at fit time).
#
# Key differences vs the 26-D sets:
# - Q2FF/Q1FF/Q0FF are EXCLUDED: they sit DOWNSTREAM of XTCAVF (between it and PENT) and are
#   modeled exactly in the differentiable Cheetah segment (xtcav-gpsr-tuning), carrying zero
#   surrogate error. Same for Q0D/Q1D/Q2D, B5D36, XTCAV voltage.
# - Movers + kickers are IN (23 = 8 longitudinal + Q5/Q4/Q3FF + 4 kickers + 8 movers): mover
#   centering is a known major sim2real error class and FF kicker settings move eta_y(DTOTR)
#   by ~17-20% via spurious vertical dispersion — they must be fittable δ DOFs. The fit is
#   MAP-regularized toward recorded settings, so the box only needs to cover plausible
#   setting error + δ headroom, not an exploration range.
# - Data-grounded longitudinal box: covers the real E300/TEST L2 phase scans (2026-04-15)
#   that the old boxes failed to contain (L1 real −26.5…−19.3 vs old box [−22.8,−17.8];
#   L2 = L2_ref + 0…8° sweeps to −29.9 vs old edge −34.0).
#
# Range provenance:
# - L1PhaseSet [−27.5,−18.5]: real anchors+feedback deltas span −26.53…−19.27 + margin.
# - L2PhaseSet [−41.5,−29.5]: real L2_ref fits −38.4/−40.4/−37.9 + scan 0…8° + margin.
# - LxEnergyOffset: unchanged from the 8-D set (real values unrecorded; centered on baseline).
# - S1/S2/S3ELkG: golden ±10% (real scans ran golden; box = readback error + δ headroom).
# - Q5/Q4/Q3FFkG: recorded LGPS BACTs ±10% (E300_15671 machine_config.json, value_match
#   mapping — the machine ran ≈golden FF optics that day; baseline = recorded).
# - Kickers: golden tuned values ±0.005 kG·m (same envelope as expanded_anchored; the real
#   kicker settings were not recorded in the DAQ configs — golden is the sim assumption).
# - Movers: golden ±0.75 mm (plan band 0.5–1 mm: covers plausible centering error at several
#   σ_j of the MAP prior while staying denser than the ±1.5 mm expanded box).
# Frozen per campaign config via fixed_overrides (NOT dims): Q19851kG/Q19871kG at the
# recorded BACTs (−63.764/+42.500; the −5.4% Q19871 trim was a latent mismatch in the old
# campaign frame), trackEnd: PENT.
# ----------------------------------------------------------------------------------------
XTCAV23_PARAMS = {
    "L1PhaseSet":     (-27.5,           -18.5,           -20.2889213421),
    "L2PhaseSet":     (-41.5,           -29.5,           -35.5447801603),
    "L1EnergyOffset": (-2.1e6,           2.1e6,           0.0),
    "L2EnergyOffset": (-48.5e6,          34.9e6,         -6.817565553821569e6),
    "L3EnergyOffset": (-43.3e6,          66.7e6,          1.1703527144314773e7),
    "S1ELkG":         (1880.5361804688,  2298.4331094618, 2089.4846449653),
    "S2ELkG":         (-4349.8119064433, -3558.9370143627, -3954.374460403),
    "S3ELkG":         (-1196.7525695935, -979.1611933037, -1087.9568814486),
    "Q5FFkG":         (-79.4074249268,   -64.9697113037,  -72.1885681152),
    "Q4FFkG":         (-89.3678749084,   -73.1191703796,  -81.2435226440),
    "Q3FFkG":         ( 89.3107727051,   109.1576110840,   99.2341918945),
    "XC1FFkG":        (-0.0026017781,     0.0073982219,    0.0023982219),
    "XC3FFkG":        (-0.0034655786,     0.0065344214,    0.0015344214),
    "YC1FFkG":        (-0.0103321184,    -0.0003321184,   -0.0053321184),
    "YC2FFkG":        (-0.0085512385,     0.0014487615,   -0.0035512385),
    "S1EL_xOffset":   ( 0.0001568268,     0.0016568268,    0.0009068268),
    "S1EL_yOffset":   (-0.0006184246,     0.0008815754,    0.0001315754),
    "S2EL_xOffset":   (-0.0011385830,     0.0003614170,   -0.0003885830),
    "S2EL_yOffset":   (-0.0006334225,     0.0008665775,    0.0001165775),
    "S2ER_xOffset":   (-0.0009179111,     0.0005820889,   -0.0001679111),
    "S2ER_yOffset":   (-0.0022381668,    -0.0007381668,   -0.0014881668),
    "S1ER_xOffset":   ( 0.0005823302,     0.0020823302,    0.0013323302),
    "S1ER_yOffset":   (-0.0018682646,    -0.0003682646,   -0.0011182646),
}

# Knobs that sit DOWNSTREAM of a treaty point cannot affect the beam there, so a campaign that
# varied them can still train a model at that point which lacks them (preprocess knob-drop rule).
DOWNSTREAM_OF = {
    "XTCAVF": ["Q2FFkG", "Q1FFkG", "Q0FFkG"],
    "PENT": [],
}
# fixed settings that only affect the beam downstream of every treaty point used for training
DOWNSTREAM_SETTINGS = {"trackEnd"}

# Named sweep sets, selectable per campaign config via the `sweep_set` key. run_sweep
# defaults to "original8" so the existing 8-D campaign/configs/manifests reproduce exactly.
# `expanded_anchored` additionally triggers manifold sampling (ff_manifold.MANIFOLD_SPECS);
# `tightbox` is a plain box-LHS over the matched-region-tightened bounds (no manifold sampler).
# `xtcav23` is the 23-D XTCAV-entrance set (plain box-LHS; its keys differ from the 26-D
# family, so it cannot be "+"-unioned with them).
SWEEP_SETS = {
    "original8": SWEEP_PARAMS,
    "expanded":  EXPANDED_PARAMS,
    "expanded_anchored": EXPANDED_ANCHORED_PARAMS,
    "tightbox": TIGHTBOX_PARAMS,
    "xtcav23": XTCAV23_PARAMS,
}


# ----------------------------------------------------------------------------------------
# Injector (Impact-T) knobs — phase 5 injector campaign (results/injector_pilot_2026-10-14/PLAN.md).
# Keys are sim/injector.py settings keys (native lume-impact / distgen keys, or its convenience /
# derived keys), so a manifest row's injector dict goes straight into injector_beam(). They are
# surrogate inputs and RL hidden context (randomized per episode), never RL controls.
#
# Ranges (sensitivity study data/injector_sensitivity_2026-10-14/, absolute arrival mode):
# - "active" (move XTCAVF spacing / sigma_z / charge ratio by >= 1 um or the ratio itself):
#   box = 1.5x the per-episode DR magnitude, so DR draws at the DR edge are interior to the data.
#     laser_timing_ps    DR +-0.2 ps (DAQ; spec 200 fs)          -> +-0.30   (+-7.6 um spacing @0.2)
#     GUNF amplitude     DR +-0.25% (assumed)                    -> +-0.375% (-+4 um)
#     GUNF phase         DR +-0.25 deg (assumed)                 -> +-0.375 deg (+-1.3 um)
#     PSDL delay (d2 avg_t) DR +-0.2 ps (Ryan, 2026-10-14)       -> 9 +- 0.30 ps (+-4.6 um per 0.1 ps)
#     drive_fraction     DAQ run means 0.69/0.71/0.73 (screen, energy-gated), design 0.75; RL DR =
#                        per-episode nominal in 0.69-0.75 +- 0.02 -> box 0.66-0.80 (golden 0.75 inside)
#     total charge       DR 0.3% 1-sigma (DAQ)                   -> +-0.9% (3 sigma; -+0.4 um/0.3%)
# - "passive" (< 0.05 deg equivalent L1/L2 phase; <= 1% transverse emittance): swept at 1x their
#   (assumed) DR magnitude — costless in a nested LHS, lets the campaign measure the
#   "negligible" claim away from the golden linac point, and keeps them available as inputs.
#     L0AF phase +-0.25 deg, solenoid +-0.5%, spot sigma_xy +-2%, truncation radius +-2%,
#     MTE +-10%, laser sigma_t (both pulses) +-2%.
# Golden values = sim/injector.py GOLDEN_SETTINGS (the 2024-10-14 golden beam).
# ----------------------------------------------------------------------------------------
_GUN_AMP = 119e6 / 2.44885          # GUNF:rf_field_scale golden
_SOL = -0.39 / 1.6                  # SOL10111:solenoid_field_scale golden
INJECTOR_PARAMS = {
    # active
    "laser_timing_ps":                      (-0.3,              0.3,               0.0),
    "GUNF:rf_field_scale":                  (_GUN_AMP * (1 - 0.00375), _GUN_AMP * (1 + 0.00375), _GUN_AMP),
    "GUNF:theta0_deg":                      (-60.7 - 0.375,    -60.7 + 0.375,     -60.7),
    "distgen:t_dist:dists:d2:avg_t:value":  (8.70,              9.30,              9.0),
    "drive_fraction":                       (0.66,              0.80,              0.75),
    "distgen:total_charge:value":           (1600.0 * 0.991,    1600.0 * 1.009,    1600.0),
    # passive
    "L0AF_phase_offset_deg":                (-0.25,             0.25,              0.0),
    "SOL10111:solenoid_field_scale":        (_SOL * 1.005,      _SOL * 0.995,      _SOL),   # negative
    "distgen:r_dist:sigma_xy:value":        (1.49 * 0.98,       1.49 * 1.02,       1.49),
    "distgen:r_dist:truncation_radius:value": (2.52 * 0.98,     2.52 * 1.02,       2.52),
    "distgen:start:MTE:value":              (360.0,             440.0,             400.0),
    "sigma_t_ps":                           (1.17 * 0.98,       1.17 * 1.02,       1.17),
}
INJECTOR_ACTIVE_KEYS = list(INJECTOR_PARAMS)[:6]
INJECTOR_KEYS = list(INJECTOR_PARAMS)

# L0B (Bmad-side; the injector's last S-band section after L0AFEND) as PASSIVE linac knobs of the
# injector campaign (physics review: not context anywhere yet). Golden = the 2024-10-14 baseline yml
# (phase -15 deg "by eye"; energy offset -2.259 MeV so the beam leaves L0B at 125 MeV). Box: phase
# +-0.5 deg, amplitude +-0.25% of the 59.5 MeV L0B gain (+-149 keV). On the machine the DL10 energy
# feedback likely trims L0B amplitude, so the amplitude knob is a conservative (unfed-back) range.
_L0B_E = -2259313.0
L0B_PARAMS = {
    "L0BPhaseSet":     (-15.5,                 -14.5,                 -15.0),
    "L0BEnergyOffset": (_L0B_E - 0.0025 * 59.5e6, _L0B_E + 0.0025 * 59.5e6, _L0B_E),
}

# Kept OUT of SWEEP_SETS (whose membership is pinned by tests and by every existing artifact):
# injector-only sets, and concatenated linac + injector sets (keys = linac keys then injector keys).
INJECTOR_SWEEP_SETS = {"inj12": INJECTOR_PARAMS, "l0b2": L0B_PARAMS}
CONCAT_SWEEP_SETS = {"xtcav23_inj12": ("xtcav23", "inj12"),
                     "xtcav23_l0b2": ("xtcav23", "l0b2"),                       # campaign linac knobs
                     "xtcav23_l0b2_inj12": ("xtcav23", "l0b2", "inj12")}        # campaign frame


# Readback-error-scale 1-sigma per knob (physical units): the hidden per-episode context jitter of
# the delta controller (rl/delta_env.py) and the "controller region" linac draws of campaigns.
SECTION_GAIN_EV = {"L0BEnergyOffset": 59.5e6, "L1EnergyOffset": 210e6, "L2EnergyOffset": 4.165e9,
                   "L3EnergyOffset": 5.5e9}


def readback_sigma(key: str, baseline: float) -> float:
    """0.1% of the section energy gain for energy offsets, 0.1 mm for movers, 5e-4 kG.m for FF
    kickers, 0.3 deg for phases, 1% for other magnets."""
    if key in SECTION_GAIN_EV:
        return 1e-3 * SECTION_GAIN_EV[key]
    if "Offset" in key:
        return 1e-4
    if key.startswith(("XC", "YC")):
        return 5e-4
    if "Phase" in key:
        return 0.3
    return 0.01 * abs(baseline)


def injector_golden(keys=None) -> dict:
    """Golden value of each injector knob (what every golden-beam campaign was generated at)."""
    return {k: INJECTOR_PARAMS[k][2] for k in (keys or INJECTOR_KEYS)}


def resolve_sweep_set(name="original8"):
    """Return (keys, low, high, baseline) for a named sweep set.

    keys preserve dict-insertion order, so a given (set, seed) yields a stable LHS draw.
    A "+"-joined name (e.g. "tightbox+expanded") returns the *element-wise union* of the
    named sets' bounds -- the common normalization frame for a dataset combining draws from
    several boxes (used by preprocess when merging campaigns, and by the RL env so its action
    space maps onto the same frame). The sets must share identical keys/order; baseline is
    taken from the first set (its operating point).
    Raises KeyError on an unknown set name.
    """
    if "+" in name:
        parts = name.split("+")
        resolved = [resolve_sweep_set(p) for p in parts]
        keys0 = resolved[0][0]
        for k, _, _, _ in resolved[1:]:
            if k != keys0:
                raise ValueError(f"cannot union sweep sets with differing keys: {name}")
        low = [min(r[1][i] for r in resolved) for i in range(len(keys0))]
        high = [max(r[2][i] for r in resolved) for i in range(len(keys0))]
        return keys0, low, high, resolved[0][3]
    if name in CONCAT_SWEEP_SETS:
        parts = [resolve_sweep_set(p) for p in CONCAT_SWEEP_SETS[name]]
        keys = [k for p in parts for k in p[0]]
        if len(set(keys)) != len(keys):
            raise ValueError(f"concatenated sweep set {name} has duplicate keys")
        return (keys, [v for p in parts for v in p[1]], [v for p in parts for v in p[2]],
                {k: v for p in parts for k, v in p[3].items()})
    if name in INJECTOR_SWEEP_SETS:
        params = INJECTOR_SWEEP_SETS[name]
        keys = list(params)
        return (keys, [params[k][0] for k in keys], [params[k][1] for k in keys],
                {k: params[k][2] for k in keys})
    if name not in SWEEP_SETS:
        raise KeyError(f"unknown sweep_set {name!r}; choose from "
                       f"{list(SWEEP_SETS) + list(INJECTOR_SWEEP_SETS) + list(CONCAT_SWEEP_SETS)}")
    params = SWEEP_SETS[name]
    keys = list(params)
    low = [params[k][0] for k in keys]
    high = [params[k][1] for k in keys]
    baseline = {k: params[k][2] for k in keys}
    return keys, low, high, baseline
