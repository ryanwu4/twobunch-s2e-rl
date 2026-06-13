"""Macroparticle convergence scan for the 2024-10-14 two-bunch baseline.

Tracks the baseline at increasing macroparticle counts up to the full 500k in the
input file, reusing one Tao instance. N=10000 is run twice (different random
downsample + ISR fluctuation seeds) to measure the run-to-run scatter floor that
convergence differences must be judged against.

Outputs (in data/phase0/):
  - convergence.json            per-run PENT specs + wall time (written incrementally)
  - beam_PENT_{label}.h5        PENT beam per run

Run with the bmad-qpad-dev env, from the repo root:
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    -m twobunch_s2e_rl.datagen.phase0.run_convergence
"""

import json
import time

import numpy as np

import FACET2_S2E as qs
from FACET2_S2E.UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam

from ..paths import facet2_root, repo_root

S2E_ROOT = facet2_root()
OUT_DIR = repo_root() / "data" / "phase0"
SCRATCH = OUT_DIR / "scratch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
CSR_TF = True

# (label, numMacroParticles); None = use all 500k in the file
RUNS = [
    ("10k_a", 10_000),
    ("10k_b", 10_000),
    ("30k", 30_000),
    ("100k", 100_000),
    ("300k", 300_000),
    ("500k_full", None),
]


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def main():
    config = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))

    tao = qs.initializeTao(
        filePath=str(S2E_ROOT),
        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
        csrTF=CSR_TF,
        numMacroParticles=10_000,  # placeholder; each run below rewrites the active beam
        scratchPath=str(SCRATCH),
        randomizeFileNames=False,
    )
    qs.setLattice(tao, **config)

    results = {}
    for label, n in RUNS:
        modifyAndSaveInputBeam(
            tao.inputBeamFilePath,
            numMacroParticles=n,
            outputBeamFilePath=tao.activeFilePath,
        )

        t0 = time.perf_counter()
        qs.trackBeam(tao, str(S2E_ROOT), **config)
        track_s = time.perf_counter() - t0

        P = qs.getBeamAtElement(tao, "PENT")
        specs = qs.getBeamSpecs(P, targetTwiss="PENT")
        n_in = n if n else 500_000
        specs["n_macro_in"] = n_in
        specs["n_live_total"] = len(P.x)
        specs["transmission_total"] = len(P.x) / n_in
        specs["trackBeam_s"] = track_s
        results[label] = specs

        P.write(str(OUT_DIR / f"beam_PENT_{label}.h5"))
        with open(OUT_DIR / "convergence.json", "w") as f:
            json.dump(results, f, indent=2, default=to_jsonable)

        print(
            f"[{label}] {track_s:.0f} s | "
            f"drive sigma x/y/z = {1e6*specs['PDrive_sigmaSI90_x']:.1f}/{1e6*specs['PDrive_sigmaSI90_y']:.1f}/{1e6*specs['PDrive_sigmaSI90_z']:.1f} um | "
            f"witness sigma x/y/z = {1e6*specs['PWitness_sigmaSI90_x']:.1f}/{1e6*specs['PWitness_sigmaSI90_y']:.1f}/{1e6*specs['PWitness_sigmaSI90_z']:.1f} um | "
            f"spacing = {1e6*specs['bunchSpacing']:.1f} um",
            flush=True,
        )

    # Summary table vs the 500k reference
    ref = results["500k_full"]
    keys = [
        ("PDrive_norm_emit_x", 1e6), ("PDrive_norm_emit_y", 1e6),
        ("PWitness_norm_emit_x", 1e6), ("PWitness_norm_emit_y", 1e6),
        ("PDrive_sigmaSI90_x", 1e6), ("PDrive_sigmaSI90_y", 1e6), ("PDrive_sigmaSI90_z", 1e6),
        ("PWitness_sigmaSI90_x", 1e6), ("PWitness_sigmaSI90_y", 1e6), ("PWitness_sigmaSI90_z", 1e6),
        ("bunchSpacing", 1e6), ("PDrive_median_energy", 1e-9), ("PWitness_median_energy", 1e-9),
    ]
    print("\n=== Values (and % deviation from 500k_full) ===")
    header = "metric".ljust(28) + "".join(label.rjust(18) for label, _ in RUNS)
    print(header)
    for key, scale in keys:
        row = key.ljust(28)
        for label, _ in RUNS:
            v = results[label][key]
            dev = 100 * (v - ref[key]) / abs(ref[key]) if ref[key] else float("nan")
            row += f"{scale*v:10.2f}({dev:+5.1f}%)".rjust(18)
        print(row)
    print("\ntrack times [s]: " + ", ".join(f"{l}={results[l]['trackBeam_s']:.0f}" for l, _ in RUNS))


if __name__ == "__main__":
    main()
