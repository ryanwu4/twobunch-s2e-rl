# Architecture plan: extensible knobs, targets, injector stage, and XTCAV endpoint

Status: **proposed** (2026-10-13/14, branch `dev/2026-10-13`; revised after two review rounds and a
lab-notebook pass). **First priority: the delta controller on L1/L2 phase (see below).**
Nothing here is implemented yet.

## Implementation status (overnight 2026-10-14; details in `docs/overnight_2026-10-14.md`)

| Phase | State |
|---|---|
| 0 compatibility guards | done (`tests/test_compat.py`, goldens in `tests/golden/`) |
| 1a core | done: `knobs.py`, `artifacts.py`, L2 frame fix, `resolve_ckpt` everywhere, loud norms cache |
| 1b goals | done: goals on any target |
| 2 XTCAV endpoint | done: `src/xtcav_segment/` on Cheetah 0.8.4, Stage A bit-identical, PENT gate passed; `sim/bmad.py` |
| 3 delta controller | done (deviation: separate `rl/delta_env.py` instead of options on `diff_env`); cloud + screen modes trained; Bmad-checked |
| 3a injector study | done (`sim/injector.py`, `scripts/injector_sensitivity.py`) |
| 4 cleanup | partly: surrogate yaml + run.json, campaign.json for new campaigns; backfill for old campaigns NOT done |
| 5 injector campaign | not started; `widen_conditioning` and preprocess embed / merge checks / knob-drop rule are in place |

## Goals and constraints

- Grow the **control/input variables** (first: injector / Impact-T) and the **control targets**
  (observables / objectives) without touching every module each time.
- **Backward compatible**: every existing sample json, manifest, processed h5, norm json, flow
  checkpoint, policy pickle, reward-norms cache, and RL `cfg.yaml` keeps loading and reproduces
  its **train-time** behaviour.
- **Readable and low-debt**: plain functions + one small dataclass. No framework, registry, or
  stage classes. A physicist should be able to read `track()` top to bottom.
- Conventions: one YAML config per stage (CLI only overrides), `run.json` provenance next to
  every output, tests guard every compatibility promise.

## Decisions (interview, 2026-10-13)

| Topic | Decision |
|---|---|
| Architecture | Lean shared core; keep the `datagen/ → surrogate/ → rl/` layout |
| Configs | YAML everywhere (surrogate gains one); plain yaml + dataclasses, no Hydra |
| Truth metric | `surrogate/properties.py` on saved clouds; `getBeamSpecs` (90% core) kept as a labeled secondary column |
| Frame authority | Each artifact's **stored** frame (ckpt buffers / norm json / manifest) is authoritative, never today's knob tables |
| Old 8-knob behaviour | Reproduce **train-time** behaviour (L2 ∈ [−38, −34], DR L2 drift std 0.125); one-line denormalize fixes in legacy modules |
| Old data | Embedded via each campaign's recorded fixed settings (`campaign.json`), checked in-frame |
| Injector variables | Gun/solenoid/L0A, laser delay + charge ratio, laser spot/MTE/pulse length — all surrogate inputs |
| Injector in RL | Hidden from the controller; randomized per episode |
| Impact-T code | Small wrapper in this repo (FACET2-S2E untouched) |
| Injector strategy | Benchmark Impact-T first, then nested design (preferred) vs chained surrogates |
| Default endpoint for new work | **XTCAVF surrogate (xtcav23) + Cheetah segment** to PENT and DTOTR; PENT-plane models stay supported |
| Segment code | Forward model only (segment, convert, screen) from xtcav-gpsr-tuning → **`src/xtcav_segment/`** here; GPSR fitting stays in xtcav-gpsr-tuning and re-exports from here |
| Cheetah | Pin released **cheetah-accelerator 0.8.4** (has `TransverseDeflectingCavity` + `drift_kick_drift`); re-run Stage-A validation on it |
| XTCAV in RL | Screen features are both **observed and objectives**; toggle `observable_source: cloud \| screen` (global default `cloud`; the delta controller trains on `screen`) |
| Screen realism | Configurable: PSF, pixel noise, per-shot centering optional; streak-calibration error on by default in screen mode (±20%); shot jitter + M-shot averaging |
| Screen features | Scalars: spacing, drive σ_z, charge ratio, ΔE, energy spread, each with a validity flag; witness σ_z monitor-only; **no peak current** (resolution-saturated on the real XTCAV); energy-gated split is the reference extraction |
| TCAV state | Passes follow the config: TCAV-on → DTOTR iff screen features are used, TCAV-off → PENT iff PENT objectives are |
| Segment controls | Q2/Q1/Q0FF only; TCAV V/phase, Q0D/Q1D/Q2D, B5D36 fixed from the segment spec |
| First priority | **Delta controller**: L1 + L2 phase deltas; xtcav23 + Cheetah plant; goals on spacing + drive σ_z; screen observables; hidden offset+gain misspecification; other knobs hidden, jittered at readback-error scale around golden |
| Not adopted (for now) | The 07-22 minimal-harness spec as a whole; pluggable-plant API / EPICS adapter |
| Legacy 8-knob code | Left in place; docstring "legacy" note; only the one-line frame fix |
| Scope | Core pipeline (not `results/` plot scripts, not XTCAV sim2real scripts) |
| Simplicity cuts | No multi-flow convention, no history window (error trend + last action instead), no stage tags, no union-with-differing-keys until needed, no coverage gate or CI |

## Gaps this fixes (from the audits)

1. Knob identity not first-class: 8-knob `PARAM_KEYS`/`BOUNDS_*`/`N_KNOB` hardcoded in ~8
   modules; normalize/denormalize reimplemented ~11x; ckpts lack `knob_keys`/`treaty_point`;
   policy runs don't record their flow ckpt.
2. **Live bug:** `SWEEP_PARAMS["L2PhaseSet"]` was widened in place (−38 → −40, commit `fafad75`),
   but all 8-knob data/norms/ckpts use [−38, −34]. `eval_bmad`, `bmad_env`, `particle_study`,
   `scripts/baseline_sanity.py`, and `rf_drift_std_vector` read the current table and so mis-map
   old 8-knob policies today (L2 drift std 0.083 instead of the trained 0.125).
3. Bmad tracking reimplemented ~8x; scripts import private `run_sweep` internals.
4. `sorted(glob)[-1]` ckpt resolution copied ~11x (lexicographic; also mis-sorts negative `val_loss`).
5. Eval fragility: `compare.py` broken; `eval_bmad` 8-knob only and drops `knob_box`; goal range
   misread for `objectives:` configs; 200 µm / goal lists hardcoded; rollout loop copied 4x.
6. Two "truth" definitions (full-RMS properties vs getBeamSpecs 90% core).
7. Silent overwrites (incl. the reward-norms cache), inconsistent output locations, merged-h5
   `idx` collisions, no git hash.

## Target layout

Only new/changed files shown.

```
src/
  twobunch_s2e_rl/
    knobs.py            NEW  KnobSpace dataclass (the single knob frame object)
    artifacts.py        NEW  resolve_ckpt, run.json read/write + no-overwrite, 2 small legacy tables
    datagen/
      sweep_params.py        knob tables, APPEND-ONLY once used; DOWNSTREAM_OF lists
      run_sweep.py           LHS + manifest + pool; calls sim.bmad.track(); writes campaign.json
    sim/                NEW
      bmad.py                track() — the ONE Bmad path
      injector.py            Impact-T wrapper: injector_beam(inj_knobs, workdir) -> ParticleGroup
    surrogate/
      model.py               + knob_keys / treaty_point hparams (default None); widen_conditioning()
      preprocess.py          campaign.json-based embed, merge checks, knob-drop rule, campaign column
      train.py               reads configs/surrogate/*.yaml (CLI overrides)
    rl/
      diff_env.py            controls vs hidden context by NAME; flow (+ optional segment) observables
      reward.py              `goal:` on any target; boundary term on controls only
      _train_utils.py        flow ckpt (+ segment spec) from yaml; run.json with parents
      rollout.py        NEW  one deterministic rollout from an explicit start point
      eval.py                fixes; shared goal helper; no hardcoded 200 µm
      validate.py       NEW  promoted transfer_setpoints + validate_bmad; any knob count; lazy pytao
  xtcav_segment/        NEW  forward model from xtcav-gpsr-tuning (see "XTCAV endpoint")
    segment.py               spec export (bmad-qpad-dev) + Cheetah build from spec
    convert.py               cloud_to_beam / beam_to_cloud
    screen.py                differentiable DTOTR screen + optional realism
    features.py         NEW  screen-feature scalars, torch; cloud-mode twins of the same names
configs/
  surrogate/*.yaml      NEW  one per NEW trained model
  segment/*.json        NEW  exported segment specs (copied from xtcav-gpsr-tuning/configs)
```

Untouched on purpose: `rl/diffrl/` (policy `.pt` files are pickles of its classes and must keep
importing from exactly those paths). Legacy 8-knob modules (`eval_bmad`, `bmad_env`,
`bmad_bridge`, `_bmad_worker`, `compare`, `particle_study`, `train_shac`) stay where they are with
a "legacy: 8-knob only, superseded by rl/validate.py" docstring note; the only code change is the
one-line switch from `BOUNDS_LOW/HIGH` to the policy's flow frame (train-time behaviour).

## Core pieces

### `KnobSpace` (`knobs.py`)

A frozen dataclass: `keys, low, high, baseline`. Methods: `normalize / denormalize` (the only
implementation of the min-max frame), `embed(values, fill)` (fill missing keys from `fill`),
`subset(keys)`, `union(other)` (same keys only, as today's `"a+b"`; differing keys added in
phase 5 if needed), `to_dict / from_dict`. No units machinery, no per-knob classes.

**Rule: knob tables are append-only once a campaign uses them.** A new range gets a new set
name; widening an existing entry in place (as happened to L2PhaseSet) is forbidden and caught by
a test that compares each norm json against `resolve_sweep_set(norm["sweep_set"])`.

### Frames for old artifacts (`artifacts.py`)

Bounds always come from the artifact itself (ckpt `knob_low/high` buffers, norm json, manifest).
Only the key names may be missing; a 3-row table fills them from `condition_dim` (verified):

| condition_dim | knob keys | treaty point |
|---|---|---|
| 8 | `original8` key order | PENT |
| 23 | `xtcav23` | XTCAVF |
| 26 | `expanded` key order (all 26-D sets share it) | PENT |

Old policy runs get their flow ckpt from a small hand-written map, **verified** by reproducing
each run's existing `logs/<run>/eval_best_goal*.json` (tightbox/matched/sigmaz may have used
`combined` or `combined_ft`).

### Provenance (`run.json`) and checkpoint resolution

Written next to every processed h5, model, policy run, and eval/validation output: `{kind,
created, git_hash, git_dirty, facet2_s2e_hash, config (resolved), parents, knob_space,
treaty_point, knob_box (resolved bounds)}`. Refuses to overwrite unless `--overwrite`.
`resolve_ckpt(pattern)` parses `val_loss` (including negative values; lower is better) or policy
reward (higher is better) and fails loudly on ambiguity.

**Reward-norms cache:** for an existing cache, a provenance mismatch is an error (or
compare-then-warn), never a silent recompute. The generic-goals change leaves `floor_desc`
byte-identical (goals are not part of it).

### One Bmad path (`sim/bmad.py`)

`run_sweep._track_and_collect`, made public. `setLattice` layers overrides on **`defaults.yml`**,
not on the baseline yml, so `track()` must always pass the full merged dict:

```python
def track(tao, settings: dict, points, save_beams_at, out_prefix, input_beam=None):
    # settings = {**baseline_yml, **fixed_overrides, **knobs}  (as run_sweep does today)
    if input_beam is not None:            # injector variation: swap the L0AFEND beam,
        write_input_beam(input_beam, tao)  # same downsample + time-centring as initializeTao
    setLattice(tao, **settings); trackBeam(tao, ..., **settings)
    # per point: save beam h5, per-bunch specs — same sample json as today
```

datagen, `rl/validate.py`, and closed-loop eval all call it. Sample-json schema unchanged (new
fields additive only).

### Truth = `properties.py`

Validation runs `properties.per_bunch` / `inter_bunch` on saved Bmad clouds — the same functions
the surrogate and reward use. `getBeamSpecs` values are a separate, labeled column. Datagen must
save beams at every point used as a target.

## Growing the knob space

1. Add new keys to a knob table **at the end of the key order** (append-only rule).
2. **Every campaign records what it held fixed.** `run_sweep` writes `data/<campaign>/campaign.json`:
   the swept frame plus a `fixed` block — baseline yml content hash, resolved values of all
   non-swept settings (`loadConfig(baseline)` incl. `defaults.yml`), `fixed_overrides`, input
   beam file, `csrTF`, `transverseWakes`, `num_macro_particles`, treaty points, FACET2-S2E hash.
   **Backfill for old campaigns** (one-off script): swept frame from `manifest.json` + norm json
   (not today's tables); fixed values from `loadConfig` with the FACET2-S2E hash recorded;
   cross-checked against the sample-json fields (`csrTF`, `transverseWakes`,
   `num_macro_particles`, `fixed_overrides`). Output checked by hand.
3. **Merge checks.** Preprocess compares the `fixed` blocks key by key and refuses to merge on a
   mismatch upstream of the treaty point (e.g. xtcav23's recorded Q19851/Q19871 vs the defaults
   used everywhere else).
4. **Embed.** Old samples fill missing knobs from their `campaign.json`, not from the new set's
   baseline (golden Q0FF = 126.35 lies outside `tightbox` [134, 150] but inside the
   `tightbox+expanded` union). Every filled value must lie inside the target frame, else refuse.
5. **Knob-drop rule.** A campaign can feed a model lacking one of its knobs only if that knob was
   fixed in the campaign or is in `DOWNSTREAM_OF[treaty_point]` (hand-written list next to the
   knob tables). Otherwise refuse.
6. **`campaign` column** in the processed h5. Old files: recovered from the `idx` reset at
   campaign boundaries (files are sorted per subdir), checked against sample counts, recorded in
   `run.json` — the h5 itself is not rewritten (`twobunch_combined.h5` already merges two).
7. **`widen_conditioning(old_ckpt, new_space)`.** Only `encoder[0]` (input `[knobs, one-hot(2)]`)
   and `feas_head[0]` see the knob vector. New columns are zero-initialized, inserted **before**
   the one-hot in `encoder[0]`. If a shared knob's bounds change, its old columns get the exact
   affine remap (`W' = W·s`, `b' = b + W·o`). Update `knob_low/high` buffers and the
   `condition_dim` / `knob_keys` hparams. Columns are matched by name. Then fine-tune on old + new.

## Growing the targets

- New function of the same clouds → one kernel in `properties.py` + one `objectives:` entry.
- **Goals on any target.** One goal buffer per objective that declares `goal: [lo, hi]`, appended
  to obs as `<key>_goal` in term order; the spacing goal keeps its existing name `spacing_goal`
  so goal-conditioned policies load unchanged. Each goal declares its sampling space (`linear` —
  default for identity-transform keys like spacing — or `log` for log10 keys like σ_z, BMAG).
  Eval's `"spacing_goal" in obs_keys` check becomes the generic form; the duplicated goal-range
  readers are removed.
- XTCAV screen features → see the next section.

## XTCAV endpoint (own section: `src/xtcav_segment/`)

**What comes in.** The forward model from xtcav-gpsr-tuning — `segment.py` (spec export +
Cheetah build), `convert.py` (`cloud_to_beam`), `screen.py` (differentiable DTOTR screen) — plus
their tests and the exported segment specs. GPSR fitting (`fit.py`, `targets*`, `stage_d.py`,
Stage C/D scripts) stays in xtcav-gpsr-tuning, whose `segment/convert/screen` become one-line
re-exports from `xtcav_segment` so its scripts keep working, and which loads the segment specs
from `configs/segment/` here (one copy). Package name differs from
`xtcav_gpsr_tuning` so the two can share a PYTHONPATH.

**Dependencies.** Pin `cheetah-accelerator==0.8.4` (released; replaces the 0.8.2-dev source clone
and its `sys.path` insertion). Vendor only the few PyTao→Cheetah bridge functions the segment
uses (`extract_elements`, `_make_element`), not the bridge's debug scripts. Gate: re-run Stage-A
`validate_segment.py` vs Bmad on 0.8.4 before anything else uses it. Keep the design rules that
make the segment correct: `drift_kick_drift` on every element (the FF is chromatically
hair-triggered), TDC `PHI0` passed through unconverted, exit-plane beams tracked with
`start_after="XTCAVF"`.

**The model chain** (one plain function, no classes):

```python
def observe(knobs):                                 # knobs: 23 flow + 3 segment (Q2/Q1/Q0FF)
    clouds = flow.sample(knobs[flow_keys])          # drive + witness at XTCAVF (V=0 exit)
    pent = segment(clouds, quads=knobs[seg_keys], tcav_on=False, end="PENT")
    out = properties(pent)                           # beam-quality objectives at PENT
    if observable_source == "screen":                # runs the TCAV-on DTOTR pass
        screen = render(segment(clouds, quads=..., tcav_on=True, end="DTOTR"), realism)
        out |= screen_features(screen)
    else:                                            # "cloud": ground truth, same feature names,
        out |= cloud_features(clouds)                # from the XTCAVF clouds the screen measures
    return out
```

Segment quad knobs are converted kG·m → k1 at the segment's reference energy, in `KnobSpace`
physical units like every other knob.

**Observable source toggle.** `observable_source: cloud | screen` (default `cloud`). Both produce
the **same feature names** — spacing, drive σ_z, charge ratio, ΔE, energy spread (+ witness σ_z
as a monitor only: 7–10 µm is below the ~10 µm resolution) — each with a **validity flag** (screen
mode: energy-gated split converged, readings above the ~50 µm separation floor, not at a scan
extreme), so objectives and obs are unchanged by the toggle. Invalid features are masked in the
reward and flagged in the obs. **No peak current**: real XTCAV I_peak is resolution-saturated at
7–10 kA (2026-07-23 sim2real), so it is never a beam property. ΔE is observed, not targeted (a
driven quantity, per Yiheng's AAC26 two-bunch talk). `cloud` computes them exactly from the bunches
(ground truth); `screen` extracts them from the rendered DTOTR image — what the machine actually
measures. **In the training env, screen features are computed in particle space at DTOTR** (z =
x / streak scale, soft energy gate on y, charge-weighted moments) — rendering the deployed 300×160
KDE image inside BPTT costs ~N×48k kernel values per env and is not feasible. Image rendering,
PSF, and pixel noise are used only in eval and tests, where the particle-space features are
checked against `xtcav_metrics.energy_gated_image_features` on rendered images. Resolution blur is
**included** in measured σ_z (quadrature, as the deployed analysis reports it), not deconvolved.

**Invalid features.** Validity flags and the split location are detached (non-differentiable by
nature); band membership uses a soft sigmoid. When a screen feature is invalid, the **reward uses
its cloud twin** (privileged, training-only, never in the obs) plus a fixed penalty, so the policy
keeps a gradient out of invalid regions (e.g. separation below the floor on the fragile L2 axis).
All abs / sqrt / log10 operations are clamped so masked branches cannot produce NaN gradients.

**Screen realism** (each a config switch): camera PSF, pixel noise, per-shot centering (default
off); **streak-calibration error** (default on in screen mode: per-episode hidden scale U(0.8, 1.2),
sized to the measured V_eff 27–30 MV vs 22 MV readback gap); **shot jitter + averaging** (per-shot
beam jitter, ~30 µm on spacing and ~9% on charge ratio, averaged over M shots per control step).
In training this is **one feature-level noise draw with σ/√M** per step (M configurable, default
50); explicit per-shot simulation is eval-only.

**TCAV state follows the config.** The TCAV-on DTOTR pass runs iff screen features are
configured; the TCAV-off PENT pass runs iff PENT objectives are (both → interleaved measurement
shots). The delta controller has no PENT objective, so it runs the DTOTR pass only.

**Validation gate before RL uses this chain.** Cheetah-vs-Bmad at PENT over a spread of knob
settings (not just two optics): spacing, per-bunch ε_n, BMAG, collinearity, σ_z. Data source:
new Bmad tracks via `rl/validate.py`, reading the campaign's fixed settings (Q19851/Q19871,
`trackEnd`) from its sample json / datagen yaml until `campaign.json` exists (xtcav23 samples saved beams only at XTCAVF; their PENT
`getBeamSpecs` give a quick secondary check). Old PENT-plane campaigns cannot train an XTCAVF
model (no XTCAVF beams); they remain PENT-model training/validation data.

## RL: controls vs hidden context

```yaml
diff_env:
  controls: [L1PhaseSet, ..., S1ER_yOffset, Q2FFkG, Q1FFkG, Q0FFkG]  # default: all non-injector knobs
  context:                                   # hidden from obs, resampled each episode
    default: uniform_in_box                  # every non-control knob, within knob_box if set
    GUNF:theta0_deg: {range: [-62, -59]}     # optional per-knob override (physical units)
```

Action dim = `len(controls)`; obs = controls + achieved observables (context excluded). The hidden
RF drift becomes the same per-episode sampler keyed by name. **The boundary/trust-region term
covers controls only** (a context knob near an edge isn't the policy's fault). Configs with no
`controls:` key keep today's behaviour exactly (all knobs controlled, boundary on all).

## First priority: delta controller on L1/L2 phase

The first controller built on the refactored pipeline. It never sees absolute knob readbacks, so
it can't lean on the simulator's knob calibration; it acts by deltas and reads the beam.

| Piece | Choice |
|---|---|
| Actions | Δ L1PhaseSet, Δ L2PhaseSet per step (clamped, scaled by `action_scale`) |
| Plant | xtcav23 flow + Cheetah segment (needs the phase-2 port and validation gate) |
| Goals | per-episode spacing goal **and** drive σ_z goal (generic goals), defined on the **measured** screen features (operator-realistic, includes the hidden streak-scale error and resolution blur); the **true** beam quantities are always logged alongside in eval |
| Observation | current beam features (screen source) + validity flags · error trend (feature − goal, change over the last step) · last action · cumulative Δ since episode start · goals. **No absolute knobs.** |
| Misspecification (hidden, per episode) | per-linac phase offset ~ N(0, 1°) and gain ~ U(0.8, 1.2), in **physical degrees** via `KnobSpace`, applied to the cumulative command then clamped: `true = start + gain·(cmd − start) + offset` |
| Other knobs (hidden) | the other 21 xtcav23 knobs + Q2/Q1/Q0FF, Gaussian around golden at readback-error scale (≈1% magnets, ≈0.1 mm movers, ≈0.3° other phases, ≈0.1% energy offsets; per-knob σ in the yaml) |
| Measurement | screen mode: streak-scale error U(0.8, 1.2), shot jitter, M-shot averaging |
| Start point | random L1/L2 phase within a `knob_box` **sub-box** of the xtcav23 range (L2 is fragile: clean witness only near −37/−36°); valid-feature fraction measured over the sub-box before training |
| Obs details | goal error in the reward's transform space (identity for spacing, log10 for σ_z), normalized by the reward norms; error trend = 0 at reset, 0 + flag when a feature is invalid; reward norms for screen-feature keys from their cloud twins on xtcav23 (new configs only; old caches byte-identical) |
| Optional (off by default) | within-episode OU drift on the RF phases, through the same name-keyed sampler |
| Injector noise | **considered from the start**: per-episode hidden offsets; representation decided by the sensitivity study below. Training starts with injector noise **off** and plugs it in when the study is done (off the critical path) |

**Reachable goals.** With two actions and two goals, only a 2-D family of (spacing, drive σ_z)
pairs is reachable, and it moves with the hidden context. Goals are therefore sampled as
*achievable* pairs: draw a random **true** L1/L2 point in the sub-box, evaluate the plant under
the episode's hidden context (noise-free, measured features), and use its features as the goal —
reachable at any gain/offset by construction. No hand-set goal box that might be
unreachable; the same procedure gives "targets spanning the reachable band" for evaluation.

**Injector noise in the early controller.** The xtcav23 surrogate has no injector inputs, so
injector noise can't enter through conditioning yet. Steps:
1. **Magnitudes from real DAQ data** (done 2026-10-14; 17 runs over 2026-04-15 and 2026-05-01,
   compared against Yakimenko 2019 / Emma 2021 / deck specs; approved). Per-episode domain
   randomization ranges:

   | Variable | Per-episode DR range | Basis |
   |---|---|---|
   | Laser-to-RF timing (common-mode) | **uniform ±0.2 ps** (≈ ±0.21° S-band) | facility spec 200 fs; measured ≤ 0.1 ps |
   | Bunch charge | 0.3% (1σ) | measured run-to-run ≈ 0.1% |
   | Charge ratio (drive fraction) | ±0.02 (≈ 10% in witness/drive ratio) | measured 0.69–0.73 between runs |
   | Laser spot / pulse length / PSDL | from specs (no PVs exist) | laser power ≈ 1% as a proxy |

   Notes: the large injector RF phase drifts seen on 2026-05-01 (L0A −30.1° → −23.7°) are likely
   readback drift (DL10 energy held to 0.001%) and are not used as noise magnitudes; energy/BLEN
   feedbacks absorb run-to-run injector offsets, which appear as commanded L1 phase.
2. **Sensitivity study with Impact-T**: perturb the injector variables (gun phase/amplitude,
   solenoid, L0A phase, laser delay + charge ratio, spot / MTE / pulse length, charge) one at a
   time and in a few combinations at the DAQ-derived magnitudes; Impact-T → L0AFEND → Bmad →
   XTCAVF; measure the shifts in spacing / drive σ_z / charge ratio / ΔE against the controller's
   own L1/L2 phase authority (measured with the **same low-N Bmad** setup, not the surrogate).
   Starts at **low particle count** (e.g. 1e4–5e4, grid ~32), raised if the effects are near the
   noise floor. **Common random numbers**: the production distgen input uses `random_type:
   hammersley` (deterministic, seedless), kept unchanged across a perturbation set; Impact-T runs
   at the Bmad particle count so the unseeded `random.sample` downsample never fires. **Noise
   floor**: a baseline pair with two different Hammersley `burnin` values (a bit-identical repeat
   would measure nothing). **Arrival time**: `runImpact` and `modifyAndSaveInputBeam` both
   subtract mean t, which erases injector timing shifts before Bmad — the study records ⟨t⟩ at
   L0AFEND before centring and reports its shift as an equivalent phase (~1.03°/ps at S-band).
   This needs the injector wrapper (per-run workdir, generation-time tagging, pinned L0A phase
   reference) early, and doubles as the first Impact-T runtime benchmark.
3. **Decide the representation** from the study: proxy now (timing → equivalent phase offset on
   the input side; charge / ratio effects on the output side), a small beam-level campaign to
   widen xtcav23, or negligible. Per-episode offsets only (constant within an episode). Replaced
   by true injector conditioning once the Impact-T surrogate exists.

**Box edges.** The env still clamps the true knobs to the box and applies the boundary penalty (on
controls only); the policy can't see the edges and learns them through the reward. If that is too
weak, expose "distance to hardware limit" (known reliably on the machine).

**What it needs from the refactor**: compatibility guards, `KnobSpace` + frame
fix, controls/context split with the name-keyed sampler, generic goals, the XTCAV endpoint
(segment port, screen features, realism, validation gate), one obs-building function with
`obs_mode: delta`, `rollout.py` from an explicit start point, and `rl/validate.py` for the Bmad
check; and, off the critical path, the minimal injector wrapper, DAQ magnitudes, and the
Impact-T sensitivity study. Existing absolute-obs policies keep `obs_mode: absolute` (default)
unchanged.

## Injector stage

**Wrapper (`sim/injector.py`)**, adapted from `FACET2_S2E.simulators.impact.runImpact`, calling
lume-impact + distgen directly:

- per-run working directory and output path (parallel-safe; never rewrites `ImpactT.yaml`);
- all chosen variables exposed by their lume-impact / distgen keys (gun phase + amplitude,
  SOL10111, L0A phase, `t_dist` delay + weights, `sigma_xy`, truncation radius, MTE, `sigma_t`);
- L0A phase reference pinned once (autophase at baseline only);
- drive/witness tagged **at generation** (two distgen components). Invariant enforced and
  tested: exactly two unique weights, drive weight strictly greater than witness weight, for any
  charge ratio / particle count (every consumer relies on it);
- beams cached by knob hash, so a nested design runs each injector point once;
- records ⟨t⟩ at L0AFEND before time-centring (centring itself reproduces the legacy unweighted
  mean exactly); equal particle counts per bunch, so the weight-tagging invariant holds.

**Verify before any campaign.** Baseline reproduction: at baseline settings the wrapper must
statistically reproduce `beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5` (inputs in its
`Source.zip`) — this is what lets old data stand in for "injector at baseline". L0B energy
re-correction for a different injector energy (`tune_to_P0Cs` / `edit_energy_based_on_beam_inj`).
Fidelity choice (production 5e5 particles, grid ≥ 60) from the benchmark.

## Testing strategy

Today: 104 tests (model math, properties, sweep tables, env/reward). Untested: surrogate
train/eval, preprocess end-to-end, Bmad paths, old-artifact loading, policy pickles.

1. **Compatibility guards** (phase 0, written with pre-refactor code, never deleted; auto-skip
   when the gitignored artifact is absent):
   - every ckpt / norm json / processed h5 / norms cache / sample-json schema variant /
     manifest loads;
   - flow goldens: CPU, fixed `z_drive`/`z_witness`, float32 tolerances, fixed knob points,
     stored under `tests/golden/`;
   - policy goldens: per-step actions and rewards for a fixed seed (covers obs order, since
     `obs_rms` is pickled with each actor), for every obs layout (15, 16, 36-D and the
     matched/sigmaz layouts);
   - norm json frame == its named sweep set (catches in-place range edits);
   - the 8-knob DR config reproduces L2 drift std 0.125.
2. **Unit tests with every new module**: `KnobSpace` (round-trip, `embed`, json); `resolve_ckpt`
   (negative `val_loss`, reward ranking, ambiguity error); `run.json` no-overwrite; norms-cache
   loud failure; merge checks, knob-drop rule, `campaign` recovery on a synthetic campaign;
   `widen_conditioning` (exact output match at several new-knob values and with a widened shared
   knob near its old edge); generic goals + `spacing_goal` name kept; controls-only boundary;
   context sampling within `knob_box`; tagging invariant; `rollout`; validation metrics ==
   `properties.py`; segment quads conversion; `cloud` vs `screen` features agree on an ideal
   screen; screen features vs `xtcav_metrics` reference; validity flags trip on the known failure
   cases (separation below floor, unconverged split); delta controller: obs contains no absolute
   knob values (perturbing absolute knobs at fixed beam features leaves obs unchanged),
   misspecification sampler statistics, achievable goals are reachable at zero misspecification,
   feature-level shot noise scales as 1/√M; finite-difference vs autograd d(spacing, drive σ_z) /
   d(L1, L2) through segment + features; no NaN gradients with masked features; streak-scale
   error affects screen mode only, with the right sign; hidden gain / offset / context do not leak
   into the step-0 obs; "scan extreme" validity case; delta rollout → `rl/validate.py` replay;
   xtcav-gpsr-tuning re-exports reproduce a Stage-A golden on Cheetah 0.8.4.
3. **Small end-to-end on synthetic data**: tiny campaign dir (one sample json per schema variant +
   tiny two-weight beam h5s) → preprocess → tiny flow for a few steps → env (with and without
   segment) → a few BPTT iterations → eval.
4. **Simulator integration (opt-in)**: in `bmad-qpad-dev`, `pytest.importorskip("pytao")` /
   `importorskip("impact")` + an `ImpactTexe` check: baseline Bmad track at low particle count vs
   the campaign baseline-repeat; injector wrapper at minimal fidelity (tagging, workdir
   isolation, cache hits); Cheetah-vs-Bmad segment agreement.

Plumbing: `tests/conftest.py` (tiny flow, synthetic campaign, paths); pytest markers (`slow`,
`data`, `bmad`, `impact`) in `pyproject.toml`; a `dev` extra with `pytest`, `pytest-cov` (coverage
as a guide, not a gate); one documented test command per conda env. Every bug fixed in the
refactor (L2 frame, compare.py, knob_box drop, goal-range misread, ckpt glob) gets a regression
test first.

## Phases

Ordered so the delta controller comes first. Each phase ends with tests green and a commit.

0. **Compatibility guards** (testing layer 1), on the untouched code.
1. **Core needed by the controller**: `knobs.py`, `artifacts.py` (`resolve_ckpt`, `run.json`,
   legacy tables), replace normalize / ckpt-glob copies, legacy one-line frame fix (L2 bug),
   norms-cache loud failure, generic goals.
2. **XTCAV endpoint**: `xtcav_segment/` port, Cheetah 0.8.4 pin + Stage-A re-validation,
   xtcav-gpsr-tuning re-exports, model chain, cloud/screen features with validity flags, realism +
   shot averaging, PENT validation gate (via `rl/validate.py`, so the one Bmad path `sim/bmad.py`
   lands here too).
3. **Delta controller**: injector wrapper (minimal) + DAQ magnitudes + Impact-T sensitivity study →
   injector-noise representation;
   controls/context split + name-keyed sampler (jitter, misspecification,
   optional OU), `obs_mode: delta`, achievable-goal sampling, controls-only boundary, `rollout.py`,
   `run.json` for policies; train BPTT on xtcav23 + segment with screen observables; evaluate on
   the surrogate and validate setpoint trajectories in Bmad.
4. **Remaining cleanup**: eval fixes not needed above, `campaign.json` (+ backfill), surrogate
   yaml, legacy docstring notes.
5. **Injector (Impact-T knobs first)**: wrapper hardening + baseline reproduction + full benchmark → nested
   (preferred) vs chained → campaign → embed / merge checks / knob-drop rule / `widen_conditioning`
   (land here, where first used) → fine-tune → RL with injector variables as hidden context.
   The cheap-knob dry run (L0B phase/energy) is optional, if it de-risks the widening step.

## Rejected alternatives (and when to revisit)

- **Stage/Chain classes** — handoffs are files (L0AFEND) or one function call (XTCAVF → segment).
  Revisit if a third code (e.g. QPAD after PENT) joins.
- **Artifact registry** — `run.json` + explicit paths. Revisit with several concurrent users.
- **Fully joint injector+linac LHS** — one Impact-T run per sample, no data reuse.
- **Chained injector→linac surrogates as default** — compounds error; needs varied-input linac
  data. Fallback if the benchmark rules out a nested design.
- **Whole xtcav-gpsr-tuning import** — the fitting is a separate study; only the forward model is
  shared infrastructure.
