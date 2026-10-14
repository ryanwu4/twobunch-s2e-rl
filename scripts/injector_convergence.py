"""Impact-T resolution convergence study for the injector campaign (particle count x space-charge grid).

For each (N, grid) the golden injector and three perturbations at their randomization magnitude
(laser timing +0.2 ps, drive-witness delay +0.2 ps, gun amplitude +0.25%) run cathode -> Impact-T
-> L0AFEND (sim/injector.py, Hammersley burnin -1 = common random numbers). Every L0AFEND beam is
then tracked by Bmad at FIXED fidelity (N_bmad = 1e5, CSR + wakes, xtcav23 baseline linac, to
XTCAVF), absolute arrival vs the same-resolution golden run, so only the Impact-T resolution varies.
Reference: the FACET2-S2E golden beam (runImpact gridCount=100, 5e5; beams/2024-10-14_Impact_TwoBunch).

Reported per (N, grid): absolute values at L0AFEND and XTCAVF vs the highest resolution and vs the
golden h5, and the injector EFFECTS (perturbed - golden at the same resolution), which is what the
surrogate has to learn.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/src nice -n 10 \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u scripts/injector_convergence.py [--workers 32]
  ... --analyze     (tables + figure from the finished runs)
Resumable: finished Impact-T beams (cached by settings hash) and Bmad jsons are skipped.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "injector_convergence_2026-10-14"
RES = REPO / "results" / "injector_convergence_2026-10-14"
CONFIGS = [(100_000, 32), (100_000, 48), (100_000, 64), (100_000, 100),
           (200_000, 64), (200_000, 100), (500_000, 64), (500_000, 100)]
POINTS = {"golden": {}, "timing+0.2ps": {"laser_timing_ps": 0.2},
          "delay+0.2ps": {"distgen:t_dist:dists:d2:avg_t:value": 9.2},
          "gunamp+0.25%": {"GUNF:rf_field_scale": "rel+0.0025"}}
N_BMAD = 100_000
CAMPAIGN_CFG = REPO / "configs" / "datagen" / "xtcav23_valgate.yaml"
GOLDEN_H5 = "/beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5"
_TAO = None


def _settings(point):
    from twobunch_s2e_rl.sim import injector as inj
    s = {"hammersley_burnin": -1}
    for k, v in POINTS[point].items():
        s[k] = inj.resolve_settings({})[k] * 1.0025 if v == "rel+0.0025" else v
    return s


def _l0af_ref():
    from twobunch_s2e_rl.sim import injector as inj
    return inj.pin_l0af_reference(OUT / "l0af_reference.json", OUT / "l0af_autophase")


def _beam_dir(point, n, grid, ref):
    from twobunch_s2e_rl.sim import injector as inj
    key = inj.cache_key(inj.resolve_settings(_settings(point)), n, grid, ref)
    return OUT / "beams" / f"N{n}_g{grid}" / key


def impact_task(args):
    point, n, grid = args
    from twobunch_s2e_rl.sim import injector as inj
    ref = _l0af_ref()
    wd = _beam_dir(point, n, grid, ref)
    t0 = time.perf_counter()
    try:
        _, info = inj.injector_beam(_settings(point), wd, n, grid=grid, l0af_reference=ref)
        return (point, n, grid), True, time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "error.txt").write_text(traceback.format_exc())
        return (point, n, grid), False, time.perf_counter() - t0


def _tao():
    global _TAO
    if _TAO is None:
        import FACET2_S2E as qs
        from twobunch_s2e_rl.datagen.paths import facet2_root
        from twobunch_s2e_rl.sim.bmad import make_tao
        camp = yaml.safe_load(CAMPAIGN_CFG.read_text())
        root = facet2_root()
        base = qs.loadConfig("/" + camp["baseline_config"], str(root))
        _TAO = (qs, root, base, camp, make_tao(qs, root, base, csr=True, wakes=True, n_particles=N_BMAD,
                                               scratch=OUT / "scratch" / f"pid{os.getpid()}"))
    return _TAO


def bmad_task(args):
    """Track one L0AFEND beam (or the golden h5) at fixed Bmad fidelity to XTCAVF."""
    point, n, grid = args
    tag = "golden_h5" if point == "golden_h5" else f"N{n}_g{grid}_{point}"
    jpath = OUT / "bmad" / f"{tag}.json"
    if jpath.exists():
        return tag, "skip"
    from FACET2_S2E.beam.manipulation import modifyAndSaveInputBeam
    from pmd_beamphysics import ParticleGroup
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
    from twobunch_s2e_rl.sim.bmad import cloud_metrics, merged_settings, track
    qs, root, base, camp, tao = _tao()
    rec = {"tag": tag, "point": point, "n_impact": n, "grid": grid, "n_bmad": N_BMAD}
    try:
        scratch = Path(tao.activeFilePath).parent
        if point == "golden_h5":
            src, centre = str(root) + GOLDEN_H5, True
        else:
            ref = _l0af_ref()
            wd = _beam_dir(point, n, grid, ref)
            info = json.loads((wd / "injector_info.json").read_text())
            gold = json.loads((_beam_dir("golden", n, grid, ref) / "injector_info.json").read_text())
            P = ParticleGroup(str(wd / "beam_L0AFEND.h5"))
            P.t = P.t + (info["mean_t_l0afend_s"] - gold["mean_t_l0afend_s"])     # absolute arrival
            src = str(scratch / f"in_{tag}.h5")
            P.write(src)
            centre = False
            rec["dt_l0afend_ps"] = 1e12 * (info["mean_t_l0afend_s"] - gold["mean_t_l0afend_s"])
            rec["impact_wall_s"] = info["wall_s"]
            rec["n_l0afend"] = info["n_l0afend"]
        random.seed(20261014)            # same downsample indices for every beam of the same N
        modifyAndSaveInputBeam(src, numMacroParticles=N_BMAD, timeCenterTF=centre,
                               outputBeamFilePath=tao.activeFilePath)
        knobs = resolve_sweep_set("xtcav23")[3]
        st = merged_settings(base, {**camp["fixed_overrides"], "trackEnd": "XTCAVF"}, knobs)
        t0 = time.perf_counter()
        out = track(qs, tao, root, st, ["XTCAVF"], n_in=N_BMAD, save_beams_at=("XTCAVF",),
                    out_prefix=str(OUT / "bmad" / tag))
        rec["bmad_wall_s"] = time.perf_counter() - t0
        rec["truth"] = cloud_metrics(str(OUT / "bmad" / f"{tag}_XTCAVF.h5"))
        rec["ok"] = True
    except Exception:  # noqa: BLE001
        rec["ok"], rec["error"] = False, traceback.format_exc()[-3000:]
    jpath.write_text(json.dumps(rec, indent=2, default=float))
    return tag, "ok" if rec["ok"] else "fail"


def l0afend_metrics(path):
    from pmd_beamphysics import ParticleGroup
    P = ParticleGroup(str(path))
    w = np.unique(P.weight)
    d, wt = P[P.weight == w[-1]], P[P.weight == w[0]]
    m = {}
    for name, b in (("drive", d), ("witness", wt)):
        m[f"{name}_emit_x_um"] = 1e6 * float(b["norm_emit_x"])
        m[f"{name}_emit_y_um"] = 1e6 * float(b["norm_emit_y"])
        m[f"{name}_sigma_t_ps"] = 1e12 * float(b["sigma_t"])
        m[f"{name}_energy_MeV"] = 1e-6 * float(b["mean_energy"])
        m[f"{name}_espread_keV"] = 1e-3 * float(b["sigma_energy"])
    m["dt_drive_witness_ps"] = 1e12 * (float(np.mean(d.t)) - float(np.mean(wt.t)))
    return m


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bmad").mkdir(exist_ok=True)
    _l0af_ref()                                       # pin once before the pool
    impact_jobs = [(p, n, g) for (n, g) in sorted(CONFIGS, key=lambda c: -c[0] * c[1] ** 2)
                   for p in POINTS]                    # longest first
    done_beams = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        pend = [pool.apply_async(bmad_task, (("golden_h5", 0, 0),))]
        for (p, n, g), ok, dt in pool.imap_unordered(impact_task, impact_jobs):
            print(f"impact {p} N={n} grid={g}: {'ok' if ok else 'FAIL'} {dt:.0f} s", flush=True)
            done_beams.setdefault((n, g), set()).add(p if ok else None)
            if done_beams[(n, g)] >= set(POINTS):     # all points of this resolution done
                pend += [pool.apply_async(bmad_task, ((q, n, g),)) for q in POINTS]
        for r in pend:
            print("bmad", *r.get(), flush=True)


def analyze():
    RES.mkdir(parents=True, exist_ok=True)
    ref = _l0af_ref()
    rows = []
    for n, g in CONFIGS:
        for p in POINTS:
            wd = _beam_dir(p, n, g, ref)
            j = OUT / "bmad" / f"N{n}_g{g}_{p}.json"
            if not (wd / "beam_L0AFEND.h5").exists() or not j.exists():
                continue
            b = json.loads(j.read_text())
            if not b.get("ok"):
                continue
            info = json.loads((wd / "injector_info.json").read_text())
            rows.append({"N": n, "grid": g, "point": p, "impact_wall_s": info["wall_s"],
                         **{f"L0_{k}": v for k, v in l0afend_metrics(wd / "beam_L0AFEND.h5").items()},
                         **{f"X_{k}": v for k, v in b["truth"].items()}})
    gj = OUT / "bmad" / "golden_h5.json"
    golden_h5 = json.loads(gj.read_text())["truth"] if gj.exists() else None
    (RES / "rows.json").write_text(json.dumps({"rows": rows, "golden_h5_xtcavf": golden_h5}, indent=1))
    print(f"{len(rows)} rows -> {RES / 'rows.json'}")
    return rows, golden_h5


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.workers)
