# Surrogate-fitting configs

Surrogate (flow) training takes a YAML config (`--config configs/surrogate/<model>.yaml`, keys =
`train.py` option names; see `example.yaml`), with CLI flags overriding it; the resolved settings and
provenance are written to `trained/<model>/run.json`. Models trained before 2026-10-14 were
CLI-driven; their settings live in the checkpoint hparams. The (de)norm frame always comes from the
preprocess `*_norm.json`, and new checkpoints also record `knob_keys` and `treaty_point`.

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
