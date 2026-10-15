"""Realization-noise check for the injector campaign (physics review #5): the golden injector at the
campaign fidelity with a DIFFERENT Hammersley burnin (a different quasi-random particle set), tracked
by Bmad at the convergence study's two linac points. Compared against the burnin -1 golden rows of
the convergence study (data/injector_convergence_2026-10-14/bmad/N100000_g64_golden_*.json).
Pass: spacing and drive sigma_z differ by <= 0.3 um.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/src nice -n 10 \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u scripts/injector_realization_noise.py
"""
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("conv", HERE / "injector_convergence.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
BURNIN = 1_000_003
N, GRID = 100_000, 64
conv.POINTS["golden_burnin2"] = {}
_orig = conv._settings


def _settings(point):
    s = _orig(point)
    if point == "golden_burnin2":
        s["hammersley_burnin"] = BURNIN
    return s


conv._settings = _settings


def bmad(lin):
    conv._TAO = None
    return conv.bmad_task(("golden_burnin2", N, GRID, lin))


if __name__ == "__main__":
    print(conv.impact_task(("golden_burnin2", N, GRID)), flush=True)
    with mp.get_context("spawn").Pool(2) as pool:
        print(pool.map(bmad, conv.LINAC), flush=True)
    for lin in conv.LINAC:
        a = json.loads((conv.OUT / "bmad" / f"N{N}_g{GRID}_golden_{lin}.json").read_text())["truth"]
        b = json.loads((conv.OUT / "bmad" / f"N{N}_g{GRID}_golden_burnin2_{lin}.json").read_text())["truth"]
        print(lin, {k: round(1e6 * (b[k] - a[k]), 3) for k in ("bunch_spacing", "drive_sigma_z", "witness_sigma_z")}, "um")
