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
#   FACET2-S2E/analysis/2026-06-19_ff-quad-range/), mirrored to both sides and clipped to the
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

# Named sweep sets, selectable per campaign config via the `sweep_set` key. run_sweep
# defaults to "original8" so the existing 8-D campaign/configs/manifests reproduce exactly.
# `expanded_anchored` additionally triggers manifold sampling (ff_manifold.MANIFOLD_SPECS);
# `tightbox` is a plain box-LHS over the matched-region-tightened bounds (no manifold sampler).
SWEEP_SETS = {
    "original8": SWEEP_PARAMS,
    "expanded":  EXPANDED_PARAMS,
    "expanded_anchored": EXPANDED_ANCHORED_PARAMS,
    "tightbox": TIGHTBOX_PARAMS,
}


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
    if name not in SWEEP_SETS:
        raise KeyError(f"unknown sweep_set {name!r}; choose from {list(SWEEP_SETS)}")
    params = SWEEP_SETS[name]
    keys = list(params)
    low = [params[k][0] for k in keys]
    high = [params[k][1] for k in keys]
    baseline = {k: params[k][2] for k in keys}
    return keys, low, high, baseline
