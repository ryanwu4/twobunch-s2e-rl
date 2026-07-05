# Surrogate-fitting configs

Surrogate (flow) training is **CLI-driven** — there is no YAML config; all hyperparameters are
`train.py` flags, and the (de)norm frame comes from the preprocess `*_norm.json`. This folder is a
placeholder for that stage of the pipeline (`datagen/` → `surrogate/` → `rl/`).

Canonical commands:

```bash
# preprocess a campaign (or merge campaigns) into a training h5 + norm.json
python -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir tightbox_v2_full,expanded_full --sweep-set tightbox+expanded \
  --scaler intrabunch --P 1024 --out processed/twobunch_combined.h5

# fit the conditional flow (condition_dim auto-resolves from the norm's knob_keys)
python -m twobunch_s2e_rl.surrogate.train \
  --processed processed/twobunch_combined.h5 --out trained/twobunch_combined \
  --coupling rqs --epochs 300 --batch-size 64

# evaluate: per-observable R² + feasibility AUC
python -m twobunch_s2e_rl.surrogate.eval \
  --ckpt 'trained/twobunch_combined/checkpoints/best-*.ckpt' \
  --processed processed/twobunch_combined.h5 --out results/surrogate/combined
```

Outputs: checkpoints + CSV logs under `trained/<model>/`; eval figures/metrics under `results/surrogate/<model>/`.
