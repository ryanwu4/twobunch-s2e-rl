# Overnight 2026-10-17 → 18

Ryan (18:30): add the deterministic centroid head; two RL directions, each constrained to realistic operation (drift rates, loop realism): (1) the best controller with **no BLEN at all** (both loops off; L1 via PREQ, L2 via multiknob; XTCAV + klystron readbacks only); (2) BLEN used realistically: **BC11 loop on for L1** (setpoint actuator, solved state-dependent loop), L2 via multiknob with BC14 off. Operating window = the controller box (L1 −26.5…−18.5, L2 −41…−33). Back ~09:00.

## Status

| Item | State |
|---|---|
| Centroid head (model + train + tests) | done; fine-tunes D1 (w_pair 3, GPU0) / D2 (w_pair 10, GPU1) running, ETA ~21:30 |
| DAQ realism analysis (timing, drift, jumps, loop gain) | agent running → `results/realism_2026-10-17/` |
| Env realism (continuous operation, absolute window, jumps, solved BC11 loop, hold) | in progress |
| RL direction 1 / 2 (S2, 2 seeds, 1200 it) | after the env changes + flow choice |
| Validation (eval, stress, DAQ replay, closed-loop Bmad) | morning |

## Log

- 18:45 centroid head (`centroid_head`, `w_centroid`): per-bunch MLP c_k(knobs) = the centroid (standardized), flow residual centred (frozen 512-sample buffer in the NLL; drawn samples recentred on their own mean, so the cloud mean is exactly c_k); c_k fitted by direct regression, spacing and pair losses on c_k (deterministic), NLL sees it detached. Fresh heads pre-fit to the train centroids (3000 steps: drive 3.1 / witness 5.8 µm median |Δz|; 20000 in the fine-tunes). Smoke: 1 epoch, val paired-Δ rms 7.48 µm (B2-cov 5.43; fresh head). Tests 4 new, 301 total pass.
- 18:50 fine-tunes D1 / D2 from B2-cov best, B2 recipe + centroid head (`scripts/launch_centroid_finetunes_2026-10-17.sh`).
