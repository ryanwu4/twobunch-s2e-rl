"""Phase 0: reproduce the 2024-10-14 two-bunch baseline and benchmark wall-time.

Runs the known-working two-bunch config (setLattice_configs/2024-10-14_twoBunch_baseline.yml)
L0AFEND -> PENT with CSR on, then re-sets the lattice and retracks once to measure the
marginal per-sample cost (the number that matters for sweep sizing, since a sweep keeps
one Tao instance alive per worker).

Outputs (in data/phase0/):
  - beam_{BEGBC20,MFFF,PENT}.h5   openPMD beams at the treaty points
  - specs.json                    per-bunch getBeamSpecs at each treaty point + transmission
  - timing.json                   init / setLattice / trackBeam wall-times
  - pent_lps.png                  longitudinal phase space at PENT

Run with the bmad-qpad-dev env, from the repo root:
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    -m twobunch_s2e_rl.datagen.phase0.run_baseline
"""

import os

# Single serial Bmad track -> use multiple OpenMP threads (Bmad saturates ~8-32). Default 32;
# override with BMAD_OMP_THREADS (=1 to time the parallel-sweep per-worker cost). Set before pytao.
_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import json
import time

import numpy as np

import FACET2_S2E as qs

from ..paths import facet2_root, repo_root

S2E_ROOT = facet2_root()
OUT_DIR = repo_root() / "data" / "phase0"
SCRATCH = OUT_DIR / "scratch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
NUM_MACRO_PARTICLES = 1e4
CSR_TF = True
TREATY_POINTS = ["BEGBC20", "MFFF", "PENT"]

EXPECTED = "Drive: 23 x 20 x 20 um; Witness: 20 x 20 x 9 um; 200 um spacing (baseline yml comment)"


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def collect_specs(tao, n_macro_in):
    """Per-bunch specs + transmission at each treaty point."""
    specs = {}
    for ele in TREATY_POINTS:
        P = qs.getBeamAtElement(tao, ele)
        specs[ele] = qs.getBeamSpecs(P, targetTwiss=ele)
        PDrive, PWitness = qs.getDriverAndWitness(P)
        specs[ele]["n_live_total"] = len(P.x)
        specs[ele]["n_live_drive"] = len(PDrive.x)
        specs[ele]["n_live_witness"] = len(PWitness.x)
        specs[ele]["transmission_total"] = len(P.x) / n_macro_in
    return specs


def main():
    config = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))
    timing = {}

    t0 = time.perf_counter()
    tao = qs.initializeTao(
        filePath=str(S2E_ROOT),
        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
        csrTF=CSR_TF,
        numMacroParticles=NUM_MACRO_PARTICLES,
        scratchPath=str(SCRATCH),
        randomizeFileNames=False,
    )
    timing["initializeTao_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qs.setLattice(tao, **config)
    timing["setLattice_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qs.trackBeam(tao, str(S2E_ROOT), **config)
    timing["trackBeam_s"] = time.perf_counter() - t0
    print(f"\n=== First track done in {timing['trackBeam_s']:.1f} s ===\n")

    specs = collect_specs(tao, int(NUM_MACRO_PARTICLES))

    for ele in TREATY_POINTS:
        P = qs.getBeamAtElement(tao, ele)
        P.write(str(OUT_DIR / f"beam_{ele}.h5"))

    P_pent = qs.getBeamAtElement(tao, "PENT")
    fig = qs.plotMod(P_pent, "z", "pz", bins=300)
    fig.savefig(OUT_DIR / "pent_lps.png", dpi=150)

    # Marginal per-sample cost: re-set lattice + retrack with the Tao instance warm
    t0 = time.perf_counter()
    qs.setLattice(tao, **config)
    timing["setLattice_warm_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    qs.trackBeam(tao, str(S2E_ROOT), **config)
    timing["trackBeam_warm_s"] = time.perf_counter() - t0
    timing["per_sample_warm_s"] = timing["setLattice_warm_s"] + timing["trackBeam_warm_s"]

    with open(OUT_DIR / "specs.json", "w") as f:
        json.dump(
            {
                "config_file": CONFIG_FILE,
                "num_macro_particles": int(NUM_MACRO_PARTICLES),
                "csr": CSR_TF,
                "expected": EXPECTED,
                "specs": specs,
            },
            f,
            indent=2,
            default=to_jsonable,
        )
    with open(OUT_DIR / "timing.json", "w") as f:
        json.dump(timing, f, indent=2, default=to_jsonable)

    # Console summary vs the recorded baseline numbers
    s = specs["PENT"]
    print("\n=== PENT summary (90% core values) ===")
    print(f"Expected: {EXPECTED}")
    for bunch in ["PDrive", "PWitness"]:
        print(
            f"{bunch}: emit_x={1e6 * s[f'{bunch}_norm_emit_x']:.1f} um, "
            f"emit_y={1e6 * s[f'{bunch}_norm_emit_y']:.1f} um, "
            f"sigma_z={1e6 * s[f'{bunch}_sigmaSI90_z']:.1f} um, "
            f"E={s[f'{bunch}_median_energy'] / 1e9:.3f} GeV, "
            f"q={1e3 * s[f'{bunch}_charge_nC']:.0f} pC, "
            f"BMAG=({s[f'{bunch}_BMAG_x']:.2f}, {s[f'{bunch}_BMAG_y']:.2f})"
        )
    print(
        f"bunchSpacing={1e6 * s['bunchSpacing']:.1f} um, "
        f"transverseOffset={1e6 * s['transverseCentroidOffset']:.1f} um, "
        f"transmission={s['transmission_total']:.3f}"
    )
    print("\n=== Timing ===")
    for k, v in timing.items():
        print(f"{k}: {v:.1f} s")


if __name__ == "__main__":
    main()
