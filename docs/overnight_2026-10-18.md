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
- 19:00 **DAQ realism pass** (agent; `results/realism_2026-10-17/REPORT.md`): one controller step ≈ 20 s (30 valid images/sign ≈ 130 shots at 10 Hz + 6.5 s settle). L1 loop-off drift (upper bound, from the BC11 loop's correction): RW 0.035°/step, ramps rms 0.022°/step (max 0.046), bounded over ~1 h (0.3–0.4° rms). BC11 loop τ 0.9–3.4 s (≪ the 6.5 s settle → instant settling OK); loop-on PACT step scatter 0.07–0.16°. Setpoint gain 3.4 deg/ln across runs (both days), 3.8–7 shot level, state-dependent on 05-01. **Beam-side L2 drift −0.094 deg/min ≈ 1° per 32-step episode (BLEN_LI14 revisits, 15/16 pairs; invisible to the phase readbacks)**; L2 readback jumps +0.7…+1.8° (LI12/13, ~1%/update). The L2 scan moves the loop-on L1 readback by +0.034 deg/deg (unknown mechanism). TORO_LI14_890 dead in all runs.
- 19:00 env realism options in v4 (all default off): `window` (absolute knob limits), `goal_change` (continuous operation: new setpoint every U[lo, hi] steps around the current state), `jumps`, and the **`blen_loop` actuator** (solved BC11 loop: the action changes ln(setpoint); the loop drives the beam-effective L1 to s·P(φ) = target through the BLEN-model cubic → state-dependent setpoint gain; beam-side drift cancelled and visible in the knob/PACT readback; charge term). Tests: loop cancels drift and the readback equals the knob move; s·ΔP = Δln exactly; window, goal change, jumps.
- 19:02 configs `configs/rl/realistic_2026-10-17/real_covB2_screen_{noblen,bc11loop}.yaml` (`scripts/make_realistic_rl_configs_2026-10-17.py`): 128-step episodes, new setpoint every 16–48 steps, window = controller box, 16-step history, action cost 0.05, 1200 it; L1 RW 0.035 + ramp U(±0.04), L2 RW 0.05 + ramp U(±0.04) + jumps (p 0.02, σ 0.45°). noblen: direct PREQ (gain U(0.7, 1.3), klystron readback); bc11loop: blen_loop (s ~ U(0.8, 1.8) → 2.6–5.7 deg/ln at the operating point, PACT noise 0.10°, loop noise 0.05°, charge RW 0.3 %/step, sensitivity 0.4–2.5). No-action check over 40 steps: noblen beam L1 drifts 1.0° (knob fixed); bc11loop beam L1 held to 0.07° while the knob moves 0.8°; L2 0.8–1.0°.
- 19:02 first RL launch OOM (2 runs/GPU next to the fine-tunes, ~30 GB each); relaunched one per GPU (GPU0 noblen s0 → bc11loop s1, GPU1 bc11loop s0 → noblen s1), ETA ~03:00.
- 19:12 **centroid fine-tunes aborted and relaunched**: the fresh heads trained at the fine-tune lr 3e-5 (val paired-Δ rms 6.8 µm at epoch 5 vs B2-cov 5.43). Added `centroid_head_lr` (own param group, 1e-3), as for the T head.
- 19:20 continuous-operation eval (`rl/eval_continuous.py`: settled error before each setpoint change, time within 10 µm after 8 settle steps, steps to reach). Baseline S2 blen_dlog (flow A) under the new L2 drift/jumps + setpoint changes, 64 episodes: settled 10.8 µm (p90 66.5), 48% settled within 10 µm, 45% of the time within 10 µm, 75% of setpoints reached (median 6 steps).
