"""Impact-T injector sensitivity study: cathode -> Impact-T -> L0AFEND -> Bmad -> XTCAVF.

Usage (bmad-qpad-dev env, repo root):
  PYTHONPATH=$PWD/src nice -n 10 /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
      scripts/injector_sensitivity.py configs/injector/sensitivity_2026-10-14.yaml \
      [--only NAME[,NAME...]] [--n-particles N] [--workers W] [--list]
  ... --analyze          -> <output_dir>/effects.csv, runs.csv, summary.md

The yaml defines OAT perturbations, combinations, a Hammersley-burnin noise floor, and the
L1/L2 phase authority runs; expand_runs() turns it into named runs. Two stages, one process
pool (<= n_workers concurrent processes, each single-threaded):
  1. injector beams (sim.injector.injector_beam), cached under beams/<cache key>/;
  2. Bmad tracking with the xtcav23 campaign settings (same call pattern as
     datagen.run_sweep._track_and_collect), one persistent Tao per worker.
Per-run results: runs[_N<n>]/<name>.json (+ the XTCAVF cloud). Resume = rerun the command;
runs whose json has "success" are skipped.

Observables (truth = surrogate/properties.py on the saved cloud, split by weight with
drive = higher weight; getBeamSpecs kept as a secondary column): bunch spacing, drive /
witness sigma_z, charge ratio, energy difference, per-bunch energy spread + norm emittance,
plus the L0AFEND <t> shift vs the baseline (ps and S-band deg).
"""
from __future__ import annotations

import os

# Bmad OpenMP: one thread per worker (Phase-0 scaling result); must precede pytao import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from twobunch_s2e_rl.datagen.paths import facet2_root, repo_root
from twobunch_s2e_rl.sim import injector as inj

C_LIGHT = 299792458.0

# ---------------------------------------------------------------------------------------
# Config -> runs
# ---------------------------------------------------------------------------------------


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["output_dir"])
    cfg["output_dir"] = str(out if out.is_absolute() else repo_root() / out)
    return cfg


def _slug(key):
    return key.replace("distgen:", "").replace(":value", "").replace(":", ".")


def _apply_delta(settings, key, delta):
    """settings[key] = golden + abs or golden * (1 + rel). delta = {'abs': a} | {'rel': r}."""
    g = inj.golden_value(key)
    if "rel" in delta:
        settings[key] = g * (1.0 + float(delta["rel"]))
    else:
        settings[key] = g + float(delta["abs"])


def expand_runs(cfg):
    """All runs of the study, in a fixed order. Each run:
    {name, group, variable, delta, injector (settings), linac (Bmad overrides),
     beam: impact|golden, golden_seed, arrival: centred|absolute}."""
    runs = []

    def add(name, group, injector=None, linac=None, beam="impact", arrival="centred",
            variable=None, delta=None, golden_seed=None):
        runs.append({"name": name, "group": group, "variable": variable, "delta": delta,
                     "injector": injector or {}, "linac": linac or {}, "beam": beam,
                     "golden_seed": golden_seed, "arrival": arrival})

    add("baseline", "baseline")
    for b in cfg.get("noise_floor_burnins", []):
        add(f"noise__burnin{b}", "noise", injector={"hammersley_burnin": int(b)})

    modes = cfg.get("arrival_modes", ["centred"])
    for key, spec in cfg.get("oat", {}).items():
        for kind in ("abs", "rel"):
            for d in spec.get(kind, []):
                s = {}
                _apply_delta(s, key, {kind: d})
                for mode in modes:
                    sfx = "" if mode == "centred" else "__abs"
                    add(f"oat__{_slug(key)}__{kind}{d:+g}{sfx}", "oat", injector=s,
                        arrival=mode, variable=key, delta={kind: d})
    for cname, deltas in cfg.get("combos", {}).items():
        s = {}
        for key, d in deltas.items():
            _apply_delta(s, key, d)
        for mode in modes:
            sfx = "" if mode == "centred" else "__abs"
            add(f"combo__{cname}{sfx}", "combo", injector=s, arrival=mode, variable=cname,
                delta=deltas)

    auth = cfg.get("authority", {})
    base_knobs = _baseline_linac_values(cfg)
    seed0 = int(cfg["golden_beam_seed"])
    for beam in auth.get("beams", []):
        if beam == "golden":
            add("auth__golden__base", "authority", beam="golden", golden_seed=seed0)
            for s2 in auth.get("extra_golden_seeds", []):
                add(f"auth__golden_seed{s2}__base", "authority_noise", beam="golden",
                    golden_seed=int(s2))
        for knob in ("L1PhaseSet", "L2PhaseSet"):
            for d in auth.get(knob, []):
                add(f"auth__{beam}__{knob}__{d:+g}", "authority", beam=beam,
                    golden_seed=seed0 if beam == "golden" else None,
                    linac={knob: base_knobs[knob] + float(d)}, variable=knob,
                    delta={"abs": float(d)})
    return runs


def _baseline_linac_values(cfg):
    with open(facet2_root() / cfg["bmad"]["baseline_config"]) as f:
        base = yaml.safe_load(f)
    return {k: float(base[k]) for k in ("L1PhaseSet", "L2PhaseSet")}


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------


def _suffix(cfg, n):
    """'' at the config's (N, grid); else _N<n> and/or _grid<g>, so fidelities never mix."""
    s = "" if n == int(cfg["n_particles"]) else f"_N{n}"
    if cfg.get("_grid_override"):
        s += f"_grid{cfg['impact_grid']}"
    return s


def run_dir(cfg, n):
    d = Path(cfg["output_dir"]) / f"runs{_suffix(cfg, n)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def l0af_reference(cfg):
    out = Path(cfg["output_dir"])
    return inj.pin_l0af_reference(out / "l0af_reference.json", out / "l0af_autophase")


def beam_workdir(cfg, run, n, ref):
    key = inj.cache_key(inj.resolve_settings(run["injector"]), n, cfg["impact_grid"], ref)
    return Path(cfg["output_dir"]) / "beams" / key


def golden_downsample(cfg, n, seed):
    """Seeded version of modifyAndSaveInputBeam's random.sample downsample of the golden
    5e5 beam (weights rescaled by N0/N), written once."""
    from pmd_beamphysics import ParticleGroup

    path = Path(cfg["output_dir"]) / "beams" / f"golden_N{n}_seed{seed}.h5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        P = ParticleGroup(str(facet2_root() / "beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5"))
        n0 = len(P)
        P = P[random.Random(seed).sample(range(n0), n)]
        P.weight = P.weight * (n0 / n)
        tmp = path.with_suffix(".tmp.h5")
        P.write(str(tmp))
        os.replace(tmp, path)
    return path


def _is_done(path):
    try:
        with open(path) as f:
            return "success" in json.load(f)
    except (OSError, json.JSONDecodeError):
        return False


# ---------------------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------------------

_CFG = None
_TAO = None
_BASELINE = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg  # (nice is inherited from the parent's `nice -n 10`)


def impact_task(args):
    """Stage 1: one injector beam (cached)."""
    run, n = args
    ref = l0af_reference(_CFG)
    wd = beam_workdir(_CFG, run, n, ref)
    t0 = time.perf_counter()
    try:
        _, info = inj.injector_beam(run["injector"], wd, n, grid=_CFG["impact_grid"],
                                    l0af_reference=ref)
        return run["name"], True, time.perf_counter() - t0, info.get("impact_run_s")
    except Exception as e:
        with open(wd / "error.txt", "w") as f:
            f.write(traceback.format_exc())
        return run["name"], False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _get_tao(n):
    global _TAO, _BASELINE
    if _TAO is None:
        import FACET2_S2E as qs

        root = facet2_root()
        _BASELINE = qs.loadConfig("/" + _CFG["bmad"]["baseline_config"], str(root))
        # The campaign's baseline repeats also pass the sweep set's baseline knobs (for
        # xtcav23 this changes only Q5FF/Q4FF/Q3FF slightly vs the yml).
        set_name = _CFG["bmad"].get("sweep_set_baseline")
        if set_name:
            from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
            _BASELINE = {**_BASELINE, **dict(resolve_sweep_set(set_name)[3])}
        scratch = Path(_CFG["output_dir"]) / "scratch" / f"pid{os.getpid()}"
        _TAO = qs.initializeTao(
            filePath=str(root), inputBeamFilePathSuffix=_BASELINE["inputBeamFilePathSuffix"],
            csrTF=_CFG["bmad"]["csrTF"], transverseWakes=_CFG["bmad"]["transverseWakes"],
            numMacroParticles=n, scratchPath=str(scratch), randomizeFileNames=True)
    return _TAO


def observables(P, n_in_drive, n_in_witness):
    """Truth observables at the treaty point from properties.py (drive = higher weight)."""
    import torch
    from twobunch_s2e_rl.surrogate import properties as props

    w = np.unique(P.weight)
    if len(w) != 2:
        return {"n_unique_weights": int(len(w))}
    d, wt = P[P.weight == w[1]], P[P.weight == w[0]]

    def cloud(pg):
        a = np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], axis=1)
        return torch.as_tensor(a, dtype=torch.float64)[None]

    cd, cw = cloud(d), cloud(wt)
    pd, pw, ib = props.per_bunch(cd), props.per_bunch(cw), props.inter_bunch(cd, cw)
    f = lambda t: float(t.reshape(-1)[0])
    out = {
        "bunch_spacing_um": f(ib["bunch_spacing"]) * 1e6,
        "drive_sigma_z_um": f(pd["sigma_z"]) * 1e6,
        "witness_sigma_z_um": f(pw["sigma_z"]) * 1e6,
        "drive_charge_nC": float(d.charge) * 1e9,
        "witness_charge_nC": float(wt.charge) * 1e9,
        "charge_ratio_w_over_d": float(wt.charge / d.charge),
        "energy_difference_MeV": f(ib["energy_difference"]) / 1e6,
        "drive_mean_energy_GeV": f(pd["mean_energy"]) / 1e9,
        "witness_mean_energy_GeV": f(pw["mean_energy"]) / 1e9,
        "drive_energy_spread_pct": 100 * f(pd["energy_spread"]) / f(pd["mean_energy"]),
        "witness_energy_spread_pct": 100 * f(pw["energy_spread"]) / f(pw["mean_energy"]),
        "drive_norm_emit_x_um": f(pd["norm_emit_x"]) * 1e6,
        "drive_norm_emit_y_um": f(pd["norm_emit_y"]) * 1e6,
        "witness_norm_emit_x_um": f(pw["norm_emit_x"]) * 1e6,
        "witness_norm_emit_y_um": f(pw["norm_emit_y"]) * 1e6,
        "transmission_drive": len(d) / max(n_in_drive, 1),
        "transmission_witness": len(wt) / max(n_in_witness, 1),
    }
    return out


def bmad_task(args):
    """Stage 2: track one run's input beam L0AFEND -> trackEnd, measure at the treaty point."""
    run, n = args
    import FACET2_S2E as qs
    from FACET2_S2E.beam.manipulation import modifyAndSaveInputBeam
    from pmd_beamphysics import ParticleGroup

    out_json = run_dir(_CFG, n) / f"{run['name']}.json"
    result = {**run, "n_particles": n, "pid": os.getpid(), "timestamp": time.time()}
    try:
        tao = _get_tao(n)
        scratch = Path(tao.activeFilePath).parent
        # Input beam and arrival-time bookkeeping
        if run["beam"] == "golden":
            src = golden_downsample(_CFG, n, run["golden_seed"])
            info = None
        else:
            ref = l0af_reference(_CFG)
            wd = beam_workdir(_CFG, run, n, ref)
            src = wd / "beam_L0AFEND.h5"
            with open(wd / "injector_info.json") as f:
                info = json.load(f)
            base = {**run, "injector": {}}
            with open(beam_workdir(_CFG, base, n, ref) / "injector_info.json") as f:
                info_base = json.load(f)
            dt = info["mean_t_l0afend_s"] - info_base["mean_t_l0afend_s"]
            result["injector_info"] = info
            result["dt_l0afend_ps"] = dt * 1e12
            result["dt_l0afend_sband_deg"] = dt * 1e12 * inj.S_BAND_DEG_PER_PS
            if run["arrival"] == "absolute":
                P_in = ParticleGroup(str(src))
                P_in.t = P_in.t + dt          # keep the shift vs the baseline arrival
                src = scratch / f"in_{run['name']}.h5"
                P_in.write(str(src))
        P_in = ParticleGroup(str(src))
        n_in = len(P_in)
        w_in = np.unique(P_in.weight)
        n_in_d, n_in_w = int(np.sum(P_in.weight == w_in[-1])), int(np.sum(P_in.weight == w_in[0]))
        # Exactly initializeTao's input path: equal counts -> random.sample is a permutation.
        modifyAndSaveInputBeam(str(src), numMacroParticles=n,
                                  timeCenterTF=(run["arrival"] == "centred"),
                                  outputBeamFilePath=tao.activeFilePath)
        merged = {**_BASELINE, **_CFG["bmad"].get("fixed_overrides", {}), **run["linac"]}
        t0 = time.perf_counter()
        qs.setLattice(tao, **merged)
        qs.trackBeam(tao, str(facet2_root()), **merged)
        result["bmad_wall_s"] = time.perf_counter() - t0
        ele = _CFG["bmad"]["treaty_point"]
        P = qs.getBeamAtElement(tao, ele)
        if len(P.x):
            P.write(str(run_dir(_CFG, n) / f"{run['name']}_{ele}.h5"))
        result["n_in"] = n_in
        result["obs"] = observables(P, n_in_d, n_in_w)
        try:
            result["specs"] = qs.getBeamSpecs(P, targetTwiss=None) or {}
        except Exception as e:
            result["specs"] = {"specs_error": f"{type(e).__name__}: {e}"}
        result["success"] = True
    except Exception as e:
        result["success"] = False
        result["error"] = traceback.format_exc()
        global _TAO
        _TAO = None   # rebuild Tao on the next task
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return run["name"], result["success"], result.get("bmad_wall_s")


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------


def run_study(cfg, runs, n, workers):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ref = l0af_reference(cfg)   # autophase once, in the parent, before the pool
    print(f"L0AF reference: {ref}", flush=True)

    todo = [r for r in runs if not _is_done(run_dir(cfg, n) / f"{r['name']}.json")]
    # Stage-1 tasks: unique injector beams needed by the todo runs (+ the baseline beam,
    # which every absolute-mode run and the impact authority runs reference).
    need, seen = [], set()
    for r in [{"name": "baseline", "injector": {}}] + [r for r in todo if r["beam"] == "impact"]:
        wd = beam_workdir(cfg, r, n, ref)
        if wd in seen or ((wd / "injector_info.json").exists() and (wd / "beam_L0AFEND.h5").exists()):
            continue
        seen.add(wd)
        need.append(r)
    golden_runs = [r for r in todo if r["beam"] == "golden"]
    for r in golden_runs:
        golden_downsample(cfg, n, r["golden_seed"])
    print(f"{len(runs)} runs, {len(todo)} to do; {len(need)} injector beams to generate; "
          f"N={n}, grid={cfg['impact_grid']}, {workers} workers", flush=True)

    t_start = time.perf_counter()
    # spawn, not fork: the parent has used polars (lume-impact's particle parser) and pytao,
    # whose thread pools deadlock forked children (observed: workers stuck in futex).
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,),
                              maxtasksperchild=None) as pool:
        # Golden-beam Bmad runs don't wait for Impact-T: start them alongside stage 1.
        golden_async = pool.map_async(bmad_task, [(r, n) for r in golden_runs], chunksize=1)
        for name, ok, wall, extra in pool.imap_unordered(impact_task, [(r, n) for r in need],
                                                         chunksize=1):
            print(f"  [impact] {name} {'ok' if ok else 'FAIL'} {wall:.0f}s {extra if not ok else ''}"
                  f" | {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
        stage2 = [(r, n) for r in todo if r["beam"] == "impact"]
        for name, ok, wall in pool.imap_unordered(bmad_task, stage2, chunksize=1):
            print(f"  [bmad] {name} {'ok' if ok else 'FAIL'} {wall and f'{wall:.0f}s' or ''}"
                  f" | {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
        for name, ok, wall in golden_async.get():
            print(f"  [bmad golden] {name} {'ok' if ok else 'FAIL'} {wall and f'{wall:.0f}s' or ''}",
                  flush=True)
    print(f"Done in {(time.perf_counter() - t_start) / 60:.1f} min. Output: {run_dir(cfg, n)}",
          flush=True)


# ---------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------

KEY_OBS = ["bunch_spacing_um", "drive_sigma_z_um", "witness_sigma_z_um",
           "charge_ratio_w_over_d", "energy_difference_MeV", "drive_energy_spread_pct",
           "witness_energy_spread_pct", "drive_norm_emit_x_um", "witness_norm_emit_x_um"]
PROXY_FIT = ["bunch_spacing_um", "drive_sigma_z_um"]       # the controller's two goals
PROXY_CHECK = ["energy_difference_MeV", "drive_energy_spread_pct"]   # predicted by the 2x2 proxy?
# Observables expressed in "authority degrees" (|delta| / max(|dO/dL1|, |dO/dL2|) per deg):
# the controller's goals plus the energy difference (all have real L1/L2 authority). Excluded:
# witness sigma_z (quadratic in L1 around golden, so a linear authority is meaningless) and
# the energy spreads / emittances (weak L1/L2 authority -> "equivalent degrees" exaggerate
# tiny absolute changes); those are reported as relative changes instead.
AUTH_OBS = ["bunch_spacing_um", "drive_sigma_z_um", "energy_difference_MeV"]
PROXY_GATE = ["energy_difference_MeV"]   # residuals that decide input-side proxy vs campaign
AUTH_THRESH_DEG = 0.05   # below this equivalent phase an effect is negligible vs the controller
SPECS_COLS = {"bunchSpacing": 1e6, "PDrive_sigmaSI90_z": 1e6, "PWitness_sigmaSI90_z": 1e6,
              "PDrive_charge_nC": 1.0, "PWitness_charge_nC": 1.0}


def load_results(cfg, n):
    rows = []
    for p in sorted(run_dir(cfg, n).glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        if not r.get("success"):
            rows.append({"name": r["name"], "group": r["group"], "success": False})
            continue
        row = {k: r.get(k) for k in ("name", "group", "variable", "beam", "arrival")}
        row["success"] = True
        row["delta"] = json.dumps(r.get("delta"))
        row["bmad_wall_s"] = r.get("bmad_wall_s")
        row["dt_l0afend_ps"] = r.get("dt_l0afend_ps")
        row["dt_l0afend_sband_deg"] = r.get("dt_l0afend_sband_deg")
        info = r.get("injector_info") or {}
        row["impact_run_s"] = info.get("impact_run_s")
        row["l0afend_mean_energy_MeV"] = (info.get("mean_energy_eV") or np.nan) / 1e6
        row.update(r.get("obs", {}))
        for k, s in SPECS_COLS.items():
            v = r.get("specs", {}).get(k)
            row[f"specs_{k}"] = v * s if v is not None else np.nan
        rows.append(row)
    return rows


def analyze(cfg, n):
    import pandas as pd

    out = Path(cfg["output_dir"])
    df = pd.DataFrame(load_results(cfg, n))
    df.to_csv(out / f"runs{_suffix(cfg, n)}.csv", index=False)
    ok = df[df.success == True].set_index("name")  # noqa: E712
    if "baseline" not in ok.index:
        print("baseline missing; nothing to analyze")
        return
    obs = [c for c in KEY_OBS if c in ok.columns]
    base = ok.loc["baseline", obs].astype(float)

    # Noise floor: std over the Hammersley realisations (baseline + burnins), ddof=1
    noise_names = ["baseline"] + [k for k in ok.index if k.startswith("noise__")]
    floor = ok.loc[noise_names, obs].astype(float).std(ddof=1) if len(noise_names) > 1 else base * np.nan
    gnames = [k for k in ok.index if k.startswith("auth__golden") and k.endswith("__base")]
    gfloor = ok.loc[gnames, obs].astype(float).std(ddof=1) if len(gnames) > 1 else base * np.nan

    # Authority: linear slope per deg (fit over 0 and the +-0.5/+-1 deg runs), per beam
    slopes = {}
    for beam, b0 in (("impact", "baseline"), ("golden", "auth__golden__base")):
        if b0 not in ok.index:
            continue
        for knob in ("L1PhaseSet", "L2PhaseSet"):
            names = [k for k in ok.index if k.startswith(f"auth__{beam}__{knob}__")]
            if not names:
                continue
            x = [0.0] + [json.loads(ok.loc[k, "delta"])["abs"] for k in names]
            y = ok.loc[[b0] + names, obs].astype(float).values
            slopes[(beam, knob)] = pd.Series(np.polyfit(x, y, 1)[0], index=obs)
    S = pd.DataFrame({f"{b}:{k}": s for (b, k), s in slopes.items()}).T

    # Effects of every injector run vs the baseline
    eff_rows = []
    J = A = None
    if ("impact", "L1PhaseSet") in slopes and ("impact", "L2PhaseSet") in slopes:
        A = np.maximum(slopes[("impact", "L1PhaseSet")].abs(), slopes[("impact", "L2PhaseSet")].abs())
    if ("impact", "L1PhaseSet") in slopes and ("impact", "L2PhaseSet") in slopes:
        J = np.array([[slopes[("impact", k)][o] for k in ("L1PhaseSet", "L2PhaseSet")]
                      for o in PROXY_FIT + PROXY_CHECK])
    for name in ok.index:
        r = ok.loc[name]
        if r["group"] not in ("oat", "combo", "noise"):
            continue
        d = ok.loc[name, obs].astype(float) - base
        row = {"name": name, "group": r["group"], "variable": r["variable"],
               "delta": r["delta"], "arrival": r["arrival"],
               "dt_l0afend_ps": r["dt_l0afend_ps"], "dt_sband_deg": r["dt_l0afend_sband_deg"]}
        for o in obs:
            row[f"d_{o}"] = d[o]
            row[f"z_{o}"] = d[o] / floor[o] if floor[o] > 0 else np.nan
            row[f"zg_{o}"] = d[o] / gfloor[o] if gfloor[o] > 0 else np.nan
            row[f"rel_{o}"] = d[o] / base[o] if base[o] else np.nan
            if A is not None and A[o] > 0:
                row[f"authdeg_{o}"] = abs(d[o]) / A[o]
            for knob in ("L1PhaseSet", "L2PhaseSet"):
                s = slopes.get(("impact", knob))
                row[f"eqdeg_{knob[:2]}_{o}"] = d[o] / s[o] if s is not None and s[o] != 0 else np.nan
        if J is not None:
            dfit = np.array([d[o] for o in PROXY_FIT])
            dphi = np.linalg.solve(J[:2], dfit)
            row["proxy_dL1_deg"], row["proxy_dL2_deg"] = dphi
            pred = J[2:] @ dphi
            for o, pr in zip(PROXY_CHECK, pred):
                row[f"proxy_resid_z_{o}"] = (d[o] - pr) / floor[o] if floor[o] > 0 else np.nan
                row[f"proxy_resid_deg_{o}"] = abs(d[o] - pr) / A[o] if A[o] > 0 else np.nan
        eff_rows.append(row)
    E = pd.DataFrame(eff_rows)
    E.to_csv(out / f"effects{_suffix(cfg, n)}.csv", index=False)
    write_summary(cfg, n, ok, obs, base, floor, gfloor, S, E)


def _recommend(sub, obs):
    """Representation recommendation for one variable from its rows (both signs).
    Returns (text, max authority-deg, proxy range)."""
    zq = sub["z_charge_ratio_w_over_d"].abs().max() if "z_charge_ratio_w_over_d" in sub else 0
    ecols = [f"authdeg_{o}" for o in AUTH_OBS if f"authdeg_{o}" in sub]
    emax = sub[ecols].max().max() if ecols else np.nan
    zmax = sub[[f"z_{o}" for o in obs if o != "charge_ratio_w_over_d"]].abs().max().max()
    pr = sub[["proxy_dL1_deg", "proxy_dL2_deg"]].abs().max().max() if "proxy_dL1_deg" in sub else np.nan
    rcols = [f"proxy_resid_deg_{o}" for o in PROXY_GATE if f"proxy_resid_deg_{o}" in sub]
    resid = sub[rcols].max().max() if rcols else np.nan
    info = f"max eq. phase {emax:.3f} deg; proxy |dL|<={pr:.3f} deg, residual {resid:.3f} deg"
    if np.isfinite(zq) and zq >= 2:
        return f"output-side feature perturbation (charge ratio moves, z={zq:.0f}); {info}"
    if np.isfinite(zmax) and zmax < 2:
        return f"negligible (below the Hammersley noise floor); {info}"
    if np.isfinite(emax) and emax < AUTH_THRESH_DEG:
        return f"negligible vs authority (< {AUTH_THRESH_DEG} deg); {info}"
    if np.isfinite(resid) and resid < AUTH_THRESH_DEG:
        return f"input-side proxy (equivalent L1/L2 offset); {info}"
    return f"small campaign (L1/L2 proxy misses the energy difference); {info}"


def write_summary(cfg, n, ok, obs, base, floor, gfloor, S, E):
    out = Path(cfg["output_dir"])
    L = [f"# Injector sensitivity study ({cfg['name']}), N = {n}, Impact grid {cfg['impact_grid']}",
         "",
         "Generated by `scripts/injector_sensitivity.py --analyze`. Observables at "
         f"{cfg['bmad']['treaty_point']} from `surrogate/properties.py` (drive = higher weight). "
         "Deltas are vs the study's own baseline injector beam (common Hammersley sequence). "
         "`z` = delta / noise floor (std over Hammersley-burnin realisations; with common "
         "random numbers this floor is tiny, so also compare with the golden-resample noise "
         "column, the noise a random-downsampled campaign beam has at the same N). "
         "`eq L1/L2` = delta / (d obs / d phase) from the authority runs on the same baseline beam. "
         "Arrival `centred` = legacy (injector <t> shift erased before Bmad); `absolute` = the "
         "<t> shift is kept, so downstream RF sees it.",
         "", "## Baseline, noise floor, authority", "",
         "| observable | baseline | noise floor (impact, burnin) | noise (golden, resample) | "
         "d/dL1 [per deg] | d/dL2 [per deg] | d/dL1 golden | d/dL2 golden |",
         "|---|---|---|---|---|---|---|---|"]
    for o in obs:
        g = lambda k: S.loc[k, o] if k in S.index else np.nan
        L.append(f"| {o} | {base[o]:.4g} | {floor[o]:.3g} | {gfloor[o]:.3g} | "
                 f"{g('impact:L1PhaseSet'):.3g} | {g('impact:L2PhaseSet'):.3g} | "
                 f"{g('golden:L1PhaseSet'):.3g} | {g('golden:L2PhaseSet'):.3g} |")
    show = ["bunch_spacing_um", "drive_sigma_z_um", "witness_sigma_z_um",
            "charge_ratio_w_over_d", "energy_difference_MeV"]
    show = [o for o in show if o in obs]
    L += ["", "## Effects per run", "",
          "Each cell: delta (z = delta/floor; eq L1 deg). dt = L0AFEND <t> shift [ps] "
          "(S-band deg).", "",
          "| run | dt ps (deg) | " + " | ".join(show) + " | proxy (dL1, dL2) deg |",
          "|---|---|" + "---|" * len(show) + "---|"]
    for _, r in E.iterrows():
        cells = [f"{r[f'd_{o}']:+.3g} (z {r[f'z_{o}']:+.1f}; {r[f'eqdeg_L1_{o}']:+.2f})"
                 for o in show]
        dt = r["dt_l0afend_ps"]
        dts = f"{dt:+.3f} ({r['dt_sband_deg']:+.3f})" if dt is not None and np.isfinite(dt) else ""
        prox = (f"{r['proxy_dL1_deg']:+.3f}, {r['proxy_dL2_deg']:+.3f}"
                if "proxy_dL1_deg" in r and np.isfinite(r["proxy_dL1_deg"]) else "")
        L.append(f"| {r['name']} | {dts} | " + " | ".join(cells) + f" | {prox} |")
    L += ["", "## Recommendation per variable", "",
          "Rule (both signs of a variable together): output-side if the charge ratio moves "
          "(|z| >= 2); negligible if every other |z| < 2 (Hammersley floor); negligible vs "
          f"authority if every effect is < {AUTH_THRESH_DEG} deg of equivalent L1/L2 phase "
          "(|delta| / max(|dO/dL1|, |dO/dL2|), over spacing, drive sigma_z, energy difference); "
          "input-side proxy if the 2x2 (dL1, dL2) that reproduces (spacing, drive sigma_z) also "
          f"predicts the energy difference to < {AUTH_THRESH_DEG} authority-deg; "
          "else small campaign. Evaluated separately for both arrival modes "
          "(absolute = physically complete; centred = what every existing campaign did).", "",
          "| variable | centred | absolute |", "|---|---|---|"]
    for var, sub in E[E.group.isin(["oat", "combo"])].groupby("variable", sort=False):
        rec = {m: _recommend(sub[sub.arrival == m], obs) if (sub.arrival == m).any() else ""
               for m in ("centred", "absolute")}
        L.append(f"| {var} | {rec['centred']} | {rec['absolute']} |")
    with open(out / f"summary{_suffix(cfg, n)}.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def _beam_stats(P):
    """L0AFEND statistics for the baseline check (whole beam + per bunch, unweighted t)."""
    d, w = inj.split_drive_witness(P)
    s = {"n": len(P), "charge_nC": P.charge * 1e9, "charge_ratio_w_over_d": w.charge / d.charge,
         "count_frac_witness": len(w) / len(P),
         "dt_drive_minus_witness_ps": (np.mean(d.t) - np.mean(w.t)) * 1e12}
    for tag, pg in (("all", P), ("drive", d), ("witness", w)):
        s[f"{tag}_mean_energy_MeV"] = pg["mean_energy"] / 1e6
        s[f"{tag}_sigma_energy_keV"] = pg["sigma_energy"] / 1e3
        s[f"{tag}_sigma_t_ps"] = pg["sigma_t"] * 1e12
        s[f"{tag}_sigma_x_um"] = pg["sigma_x"] * 1e6
        s[f"{tag}_sigma_y_um"] = pg["sigma_y"] * 1e6
        s[f"{tag}_norm_emit_x_um"] = pg["norm_emit_x"] * 1e6
        s[f"{tag}_norm_emit_y_um"] = pg["norm_emit_y"] * 1e6
    return s


def baseline_check(cfg, n):
    """Wrapper baseline beam (N, grid) vs the golden 5e5 / grid-100 L0AFEND beam."""
    from pmd_beamphysics import ParticleGroup

    out = Path(cfg["output_dir"])
    ref = l0af_reference(cfg)
    wd = beam_workdir(cfg, {"injector": {}}, n, ref)
    P_ours = ParticleGroup(str(wd / "beam_L0AFEND.h5"))
    with open(wd / "injector_info.json") as f:
        info = json.load(f)
    P_gold = ParticleGroup(str(facet2_root() / "beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5"))
    a, b = _beam_stats(P_ours), _beam_stats(P_gold)
    a["mean_t_abs_ns"], b["mean_t_abs_ns"] = info["mean_t_l0afend_s"] * 1e9, float(np.mean(P_gold.t)) * 1e9
    L = [f"# Baseline check: wrapper (N={n}, grid {cfg['impact_grid']}) vs golden 2024-10-14 "
         "(N=5e5, grid 100)", "",
         f"L0AF reference (pinned autophase): {ref}. Golden ImpactT.yaml: 359.86189124 deg / "
         "32599504.356.", "",
         "| quantity | wrapper | golden | diff | rel diff |", "|---|---|---|---|---|"]
    for k in a:
        diff = a[k] - b[k]
        rel = diff / b[k] if b[k] else np.nan
        L.append(f"| {k} | {a[k]:.6g} | {b[k]:.6g} | {diff:+.4g} | {rel:+.2%} |")
    text = "\n".join(L) + "\n"
    with open(out / f"baseline_check_N{n}_grid{cfg['impact_grid']}.md", "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated run names (plus prefixes*)")
    ap.add_argument("--n-particles", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the expanded run names")
    ap.add_argument("--baseline-check", action="store_true",
                    help="compare the cached baseline injector beam with the golden beam")
    ap.add_argument("--grid", type=int, default=None, help="override impact_grid")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    n = int(a.n_particles or cfg["n_particles"])
    if a.grid and int(a.grid) != int(cfg["impact_grid"]):
        cfg["impact_grid"] = int(a.grid)
        cfg["_grid_override"] = True
    if a.baseline_check:
        baseline_check(cfg, n)
        return
    if a.analyze:
        analyze(cfg, n)
        return
    runs = expand_runs(cfg)
    if a.only:
        pats = a.only.split(",")
        runs = [r for r in runs if any(r["name"] == p or (p.endswith("*") and r["name"].startswith(p[:-1]))
                                       for p in pats)]
    if a.list:
        for r in runs:
            print(r["name"])
        return
    workers = min(int(a.workers or cfg["n_workers"]), 24)   # hard cap (shared machine)
    run_study(cfg, runs, n, workers)


if __name__ == "__main__":
    sys.exit(main())
