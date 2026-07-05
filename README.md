# twobunch-s2e-rl

FACET-II **two-bunch** start-to-end (S2E) pipeline: an LHS **data campaign** over
compression/steering knobs (tracked L0AFEND→PENT with Bmad/Tao), feeding an eventual
conditional **normalizing-flow surrogate** and downstream **RL**. This is the two-bunch
S2E analog of `photoinjector-rl-clean` (which targets the PR10241 photoinjector).

## Status

| stage | state |
|---|---|
| `datagen` — LHS sweep (Bmad/Tao) | **working** (5000-sample campaign complete) |
| `analysis` — consolidation + summary plots | **working** |
| `surrogate` — conditional NF | **working** — RealNVP + per-bunch whitening + feasibility heads; viability AUC 0.999, inter-bunch R²>0.97 (witness emittance R²~0.5–0.7 is the v2 target) |
| `rl` — SHAC/BPTT on the surrogate | **working** — `TwoBunchFlowEnv` + composite reward (200 µm spacing target, floor-clamped emittance, >90% survival hinge); vendored diffrl SHAC/BPTT; DR hooks (random starts + off-by-default RF drift); particle-count study sets n=2048 |

## Two environments

Data generation and the ML stages run in **different conda envs** and cannot share one:

- **`bmad-qpad-dev`** — datagen + analysis. Has `FACET2_S2E` (Bmad/Tao via `pytao`)
  editable-installed, plus pandas/matplotlib/h5py.
- **`slac-rl`** — surrogate + RL (torch); used once those stages exist.

The package is used via `PYTHONPATH=$PWD/src` (not `pip install`), matching the sibling
repos.

## FACET2_S2E dependency

`datagen` imports `FACET2_S2E` and reads its Bmad lattice / beam / config files. The repo
root of the FACET2-S2E checkout is resolved by `datagen/paths.py::facet2_root()`:

1. `$FACET2_S2E_ROOT` if set, else
2. derived from the installed package (`Path(FACET2_S2E.__file__).parents[2]`).

With the default editable install at `/home/rwu4/photoinjector-rl/FACET2-S2E`, no env var
is needed. Set `FACET2_S2E_ROOT` only if you point at a different checkout.

## Run

**Data campaign** (`bmad-qpad-dev`):
```bash
cd twobunch-s2e-rl
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/smoke.yaml   # then pilot.yaml, full.yaml
```
Re-running resumes: completed samples are skipped via their `sample_*.json`; the LHS
manifest is seeded and written once, so indices are stable. Output → `data/<config name>/`.

**Analysis** (`bmad-qpad-dev`, or any env with pandas/matplotlib/h5py). Reusable per-campaign
tools live in `src/.../analysis_tools/`; one-off study reports live beside their figures in
`results/<study>/` and write there (run by file path):
```bash
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.analysis_tools.build_dataset                    # -> results/tables/dataset.pkl
PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis_tools.achievable_targets tightbox_v2_full  # -> results/tightbox_v2_full/
PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/dataset_overview/summary_plots.py
PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/combined_dataset/dataset_coverage.py
```

**Surrogate** (`slac-rl` torch env): conditional RealNVP over the 8 knobs → drive+witness 6D
beam at PENT + per-bunch feasibility. Reparameterized sampling, so `model.observables(knobs)`
is differentiable w.r.t. the knobs (for later MBRL). See `docs/surrogate_roadmap.md`.
```bash
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.preprocess --subdir full   # -> processed/twobunch_flow.h5
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.train --epochs 150         # -> trained/twobunch_flow/
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.eval  --ckpt "trained/twobunch_flow/checkpoints/best-*.ckpt"
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.diagnostics --ckpt "trained/twobunch_flow/checkpoints/best-*.ckpt"
PYTHONPATH=$PWD/src pytest tests/                                                  # 21 tests (data tests auto-skip if absent)
```
`eval` writes parity / phase-space / metrics to `results/surrogate/<model>/` (via `--out`);
`diagnostics` adds a `diagnostics/` subfolder there (full 6D corner plots per bunch, phase-space slices +
correlation matrices for representative samples, the knob→observable response surface,
feasibility calibration, and per-coordinate dispersion ratios).

## Layout

```
src/twobunch_s2e_rl/
  datagen/        sweep_params.py (authoritative knob table), run_sweep.py, paths.py (+ output-dir helpers)
    phase0/       run_baseline.py, run_convergence.py, bench_threads.py (pre-campaign study)
  analysis_io.py  shared loaders (load/derived/flatten_sample) for tools + report scripts
  analysis_tools/ reusable per-campaign tools: achievable_targets, build_dataset, beam_matching_solve, offset_floor
  surrogate/      preprocess, dataset, model (TwoBunchFlow), train, eval, diagnostics, plot_training
  rl/             reward, diff_env (TwoBunchFlowEnv), diffrl/ (vendored SHAC/BPTT),
                  train_{shac,bptt}, eval, compare, particle_study
configs/          datagen/ (smoke|pilot|full|tightbox*|wakes_gate); rl/ (shac|bptt*); surrogate/ (CLI-driven, README)
data/             campaign output + data/phase0/ baseline study output (gitignored)
results/          generated outputs (gitignored): <study>/ (report scripts + their figures/CSVs),
                  surrogate/<model>/ (metrics, parity, loss_curves, diagnostics, r2),
                  tables/ (dataset cache), rl/, presentation_figures/
trained/<model>/  checkpoints/ + csv/ + train.log  (models + logs only, no figures)
docs/             surrogate_roadmap.md
tests/            test_sweep_params.py + surrogate/rl tests
```

## Parameters & provenance

`src/twobunch_s2e_rl/datagen/sweep_params.py` is the single source of truth for the 8 swept
knobs (L1/L2 phase, L1/L2/L3 energy offsets, S1E/S2E/S3E sextupoles), their bounds, and the
golden-baseline values. Range provenance (the "Road to two bunches" deck) is documented in
that file's module docstring.
