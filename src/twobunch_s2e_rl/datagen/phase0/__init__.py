"""Phase-0 two-bunch baseline study (pre-campaign).

One-off diagnostics that set the sweep's fidelity/threading choices:
- run_baseline   : reproduce the 2024-10-14 baseline + warm per-sample wall-time
- run_convergence: macroparticle convergence scan (10k..500k) + run-to-run scatter
- bench_threads  : OpenMP thread-scaling of one track

Run in the bmad-qpad-dev env (needs FACET2_S2E + pytao); outputs go to data/phase0/.
"""
