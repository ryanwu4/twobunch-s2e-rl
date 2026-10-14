"""Impact-T resolution convergence study for the injector campaign (particle count x space-charge grid).

For each (N, grid) the golden injector and three perturbations at their randomization magnitude
(laser timing +0.2 ps, drive-witness delay +0.2 ps, gun amplitude +0.25%) run cathode -> Impact-T
-> L0AFEND (sim/injector.py, Hammersley burnin -1 = common random numbers), then Bmad (CSR + wakes,
xtcav23 baseline linac, to XTCAVF) with the SAME particle count as Impact-T, exactly as a campaign
row runs it, absolute arrival vs the same-resolution golden run. The scan is at and BELOW the
planned campaign fidelity (1e5, grid 64), to see how far it can be cut; the high end is anchored by
one 1e5 / grid-100 point and the FACET2-S2E golden beam (runImpact gridCount=100, 5e5;
beams/2024-10-14_Impact_TwoBunch), tracked by Bmad at every N (isolates the Bmad count effect).

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
CONFIGS = [(10_000, 16), (10_000, 32), (20_000, 16), (20_000, 32), (20_000, 48),
           (50_000, 32), (50_000, 48), (50_000, 64), (100_000, 32), (100_000, 48), (100_000, 64),
           (100_000, 100), (100_000, (48, 48, 128))]   # last: anisotropic (finer Nz; physics review)
POINTS = {"golden": {}, "timing+0.2ps": {"laser_timing_ps": 0.2}, "dfrac0.80": {"drive_fraction": 0.80},
          "delay+0.2ps": {"distgen:t_dist:dists:d2:avg_t:value": 9.2},
          "gunamp+0.25%": {"GUNF:rf_field_scale": "rel+0.0025"}}
# the golden h5 (Impact-T 5e5 / grid 100) is also tracked at every N: isolates the Bmad count effect
CAMPAIGN_CFG = REPO / "configs" / "datagen" / "xtcav23_valgate.yaml"
# linac points: the xtcav23 baseline, and the strongly compressed pilot point "LHS B" (L1 -26.8,
# L2 -32.0; spacing ~60 um) where the pilot's injector response flipped sign and CSR noise is worst
PILOT_MANIFEST = REPO / "data" / "injector_pilot_2026-10-14" / "manifest.json"
LINAC = ("baseline", "compressed")


def _linac_knobs(name):
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
    if name == "baseline":
        return resolve_sweep_set("xtcav23")[3]
    rows = json.loads(PILOT_MANIFEST.read_text())
    return next(r["knobs"] for r in rows if r["block"] == "crossed" and r["linac_id"] == 2)
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


def _gtag(grid):
    return "x".join(map(str, grid)) if isinstance(grid, tuple) else str(grid)


def _beam_dir(point, n, grid, ref):
    from twobunch_s2e_rl.sim import injector as inj
    key = inj.cache_key(inj.resolve_settings(_settings(point)), n, grid, ref)
    return OUT / "beams" / f"N{n}_g{_gtag(grid)}" / key


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


def _tao(n):
    global _TAO
    if _TAO is None:
        import FACET2_S2E as qs
        from twobunch_s2e_rl.datagen.paths import facet2_root
        from twobunch_s2e_rl.sim.bmad import make_tao
        camp = yaml.safe_load(CAMPAIGN_CFG.read_text())
        root = facet2_root()
        base = qs.loadConfig("/" + camp["baseline_config"], str(root))
        _TAO = (qs, root, base, camp, make_tao(qs, root, base, csr=True, wakes=True, n_particles=n,
                                               scratch=OUT / "scratch" / f"pid{os.getpid()}"))
    return _TAO


def bmad_task(args):
    """Track one L0AFEND beam (or the golden h5) to XTCAVF, Bmad count = Impact-T count."""
    point, n, grid, linac = args
    n_bmad = n                       # golden_h5: the golden 5e5 beam downsampled to n
    tag = (f"golden_h5_N{n}" if point == "golden_h5" else f"N{n}_g{_gtag(grid)}_{point}") + f"_{linac}"
    jpath = OUT / "bmad" / f"{tag}.json"
    if jpath.exists():
        return tag, "skip"
    from FACET2_S2E.beam.manipulation import modifyAndSaveInputBeam
    from pmd_beamphysics import ParticleGroup
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
    from twobunch_s2e_rl.sim.bmad import cloud_metrics, merged_settings, track
    qs, root, base, camp, tao = _tao(n_bmad)
    rec = {"tag": tag, "point": point, "n_impact": n, "grid": _gtag(grid), "n_bmad": n_bmad, "linac": linac}
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
        modifyAndSaveInputBeam(src, numMacroParticles=n_bmad, timeCenterTF=centre,
                               outputBeamFilePath=tao.activeFilePath)
        knobs = _linac_knobs(linac)
        st = merged_settings(base, {**camp["fixed_overrides"], "trackEnd": "XTCAVF"}, knobs)
        t0 = time.perf_counter()
        out = track(qs, tao, root, st, ["XTCAVF"], n_in=n_bmad, save_beams_at=("XTCAVF",),
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
    cells = lambda g: g[0] * g[1] * g[2] if isinstance(g, tuple) else g ** 3
    impact_jobs = [(p, n, g) for (n, g) in sorted(CONFIGS, key=lambda c: -c[0] * cells(c[1]))
                   for p in POINTS]                    # longest first
    done_beams = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        pend = [pool.apply_async(bmad_task, (("golden_h5", n, 0, lin),))
                for n in sorted({c[0] for c in CONFIGS}) for lin in LINAC]
        for (p, n, g), ok, dt in pool.imap_unordered(impact_task, impact_jobs):
            print(f"impact {p} N={n} grid={_gtag(g)}: {'ok' if ok else 'FAIL'} {dt:.0f} s", flush=True)
            done_beams.setdefault((n, g), set()).add(p if ok else None)
            if done_beams[(n, g)] >= set(POINTS):     # all points of this resolution done
                pend += [pool.apply_async(bmad_task, ((q, n, g, lin),)) for q in POINTS for lin in LINAC]
        for r in pend:
            print("bmad", *r.get(), flush=True)


def analyze():
    RES.mkdir(parents=True, exist_ok=True)
    ref = _l0af_ref()
    rows = []
    for n, g in CONFIGS:
        for p in POINTS:
            wd = _beam_dir(p, n, g, ref)
            for lin in LINAC:
                j = OUT / "bmad" / f"N{n}_g{_gtag(g)}_{p}_{lin}.json"
                if not (wd / "beam_L0AFEND.h5").exists() or not j.exists():
                    continue
                b = json.loads(j.read_text())
                if not b.get("ok"):
                    continue
                info = json.loads((wd / "injector_info.json").read_text())
                rows.append({"N": n, "grid": _gtag(g), "point": p, "linac": lin, "impact_wall_s": info["wall_s"],
                             "bmad_wall_s": b.get("bmad_wall_s"),
                             **{f"L0_{k}": v for k, v in l0afend_metrics(wd / "beam_L0AFEND.h5").items()},
                             **{f"X_{k}": v for k, v in b["truth"].items()}})
    golden_h5 = {}
    for gj in sorted((OUT / "bmad").glob("golden_h5_N*.json")):
        g = json.loads(gj.read_text())
        if g.get("ok"):
            golden_h5[f"{g['n_bmad']}_{g.get('linac', 'baseline')}"] = {**g["truth"], "bmad_wall_s": g.get("bmad_wall_s")}
    (RES / "rows.json").write_text(json.dumps({"rows": rows, "golden_h5_xtcavf": golden_h5}, indent=1))
    print(f"{len(rows)} rows -> {RES / 'rows.json'}")
    return rows, golden_h5


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.workers)
