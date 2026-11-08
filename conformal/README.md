# Conformal prediction: calibration data

Calibration datasets for conformal prediction on the two-bunch CNF surrogates: the deployed model (facet2-twobunch-cnf, bundle `c_b3_ema`) and the research MFFF models. Each dataset is a campaign of independent FACET2-S2E simulations, run by `datagen/injector_campaign.py` with a `design.calibration` block.

## Layout

| Path | What |
|---|---|
| `datasets/<name>/campaign.yaml` | The campaign config |
| `datasets/<name>/design_check.json` | Pre-launch check: physics equal to the model's training campaign, knob box equal to the model bundle, checkpoint sha256, manifest sha256 |
| `datasets/<name>/SHA256SUMS`, `datasheet.json` | Written by `finalize_dataset.py` when the campaign is done |
| `datasets/repro_check_<date>/result.json` | Training rows re-run with the calibration code; must match bit for bit |
| `results.jsonl`, `RESULTS.md` | Results log (below) |
| `scripts/` | `check_design.py`, `launch_campaign.sh`, `repro_check.py`, `finalize_dataset.py` |
| `data/<name>/` | Simulation output (gitignored), at `/home/rwu4/photoinjector-rl/twobunch-s2e-rl/conformal/data/` |

## Making a dataset

1. Write `datasets/<name>/campaign.yaml`.
2. Run `check_design.py` and commit its `design_check.json`.
3. Launch the campaign and its reproduction check with `launch_campaign.sh`, which runs a worktree pinned to the given commit.
4. When the campaign is done, run `finalize_dataset.py`. It writes the checksums and the datasheet, which lists every failed row. `--freeze` then makes the data read-only, and `--verify` re-checks the checksums at any time.

## Rules

- **Never used for training or model selection.** Every row is tagged `purpose: calibration`, and `surrogate/preprocess.py` refuses tagged campaigns unless `--calibration` (and `--data-root conformal/data`) is given.
- **Failed simulations are listed, not dropped.** How they count is decided with the method.

## Results log

- `results.jsonl` gets one line per evaluation of a model on calibration data. Each line records:
  - time;
  - model checkpoint sha256;
  - dataset;
  - rows used;
  - score definition;
  - α;
  - result path;
  - git commit.
- `RESULTS.md` holds the curated headline results.

## Datasets

| Name | Box | Rows | Planes saved | Status |
|---|---|---|---|---|
| `narrow_uniform_2026-11-06` | deployed model, `c_b3_ema` (37 knobs), iid uniform | 1000 | MFFF, XTCAVF, PENT | done 11-08 04:37, frozen: 1000 succeeded, 0 failed; witness reaches XTCAVF in 994 |

`narrow_uniform_2026-11-06` ran in three launches, all from commit 1418d2e. The server crashed on 11-06 at ~21:20, and on 11-07 a reboot at ~22:18 stopped the campaign again. Before each resume, every injector beam with `injector_info.json` was checked: it loads, has more than 90 % of its particles and all values finite. Beams cut off mid-run were deleted and generated again. After the crash, one finished beam was re-run and matched bit for bit, apart from timings. The launch records are in `datasheet.json`. They differ only in pid and start time, except for `versions.ImpactTexe.path`: the first launch records the slac-rl copy and the later two the bmad-qpad-dev copy. The two paths are hard links to the same file (same sha256), and every launch ran the bmad-qpad-dev one. The path differs only because the record comes from `PATH`, while the run uses the copy beside `sys.executable`.
