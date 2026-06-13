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
| `surrogate` — conditional NF | stub + roadmap (`docs/surrogate_roadmap.md`) |
| `rl` — PPO/SHAC/BPTT on the surrogate | stub |

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
  -m twobunch_s2e_rl.datagen.run_sweep configs/smoke.yaml      # then pilot.yaml, full.yaml
```
Re-running resumes: completed samples are skipped via their `sample_*.json`; the LHS
manifest is seeded and written once, so indices are stable. Output → `data/<config name>/`.

**Analysis** (`bmad-qpad-dev`, or any env with pandas/matplotlib/h5py):
```bash
PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.analysis.build_dataset           # -> artifacts/dataset.pkl
PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.summary_plots
PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.twobunch_quality_plots
```

## Layout

```
src/twobunch_s2e_rl/
  datagen/    sweep_params.py (authoritative 8-knob table), run_sweep.py, paths.py
  analysis/   build_dataset.py, summary_plots.py, twobunch_quality_plots.py
  surrogate/  stub  (see docs/surrogate_roadmap.md)
  rl/         stub
configs/      smoke|pilot|full|wakes_gate.yaml   (output_dir -> data/<name>)
data/         campaign output (gitignored)
artifacts/    regenerated dataset cache + figures (gitignored)
docs/         surrogate_roadmap.md
tests/        test_sweep_params.py
```

## Parameters & provenance

`src/twobunch_s2e_rl/datagen/sweep_params.py` is the single source of truth for the 8 swept
knobs (L1/L2 phase, L1/L2/L3 energy offsets, S1E/S2E/S3E sextupoles), their bounds, and the
golden-baseline values. Range provenance (the "Road to two bunches" deck) is documented in
that file's module docstring.
