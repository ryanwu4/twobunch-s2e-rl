"""Thread-scaling benchmark for the two-bunch baseline track.

The convergence-scan timings were taken with no OpenMP restriction and saturated all
128 threads. This measures whether that parallelism is real or spin-waiting, by timing
the same baseline track (10k and 100k macroparticles) under OMP_NUM_THREADS=<n>.
Appends to data/phase0/thread_scaling.json keyed by thread count.

Run with the bmad-qpad-dev env, from the repo root:
  OMP_NUM_THREADS=1 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    -m twobunch_s2e_rl.datagen.phase0.bench_threads
"""

import json
import os
import time

import FACET2_S2E as qs
from FACET2_S2E.UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam

from ..paths import facet2_root, repo_root

S2E_ROOT = facet2_root()
OUT_DIR = repo_root() / "data" / "phase0"
SCRATCH = OUT_DIR / "scratch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "thread_scaling.json"

CONFIG_FILE = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
COUNTS = [10_000, 100_000]


def main():
    n_threads = os.environ.get("OMP_NUM_THREADS", "unset")
    config = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))

    tao = qs.initializeTao(
        filePath=str(S2E_ROOT),
        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
        csrTF=True,
        numMacroParticles=COUNTS[0],
        scratchPath=str(SCRATCH),
        randomizeFileNames=True,
    )
    qs.setLattice(tao, **config)

    results = json.load(open(OUT_FILE)) if OUT_FILE.exists() else {}
    results.setdefault(n_threads, {})
    for n in COUNTS:
        modifyAndSaveInputBeam(
            tao.inputBeamFilePath,
            numMacroParticles=n,
            outputBeamFilePath=tao.activeFilePath,
        )
        t0 = time.perf_counter()
        qs.trackBeam(tao, str(S2E_ROOT), **config)
        dt = time.perf_counter() - t0
        results[n_threads][str(n)] = dt
        print(f"OMP_NUM_THREADS={n_threads}, N={n}: {dt:.1f} s", flush=True)
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    os.remove(tao.activeFilePath)


if __name__ == "__main__":
    main()
