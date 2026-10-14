"""Injector + linac campaign driver: cathode -> Impact-T -> L0AFEND -> Bmad -> treaty points.

Usage (bmad-qpad-dev env, repo root):
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nice -n 10 \\
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \\
    -m twobunch_s2e_rl.datagen.injector_campaign configs/datagen/injector_pilot_2026-10-14.yaml
  ... --dry-run     build + print the manifest only (no Impact-T / Bmad)

Nested design (docs/architecture_plan.md "Injector strategy"): each injector point is run through
Impact-T ONCE (sim/injector.py, cached by settings hash under <out>/beams/<key>/) and reused for
several linac knob settings in Bmad. One spawn-context pool of <= n_workers single-threaded
processes runs both stages; a row's Bmad task is submitted as soon as its injector beam (and the
golden-injector reference beam) exist, so the two stages overlap.

Manifest (built once, deterministic by seed; see build_manifest): rows of
  {idx, block, beam: impact|golden_h5, injector: {12 injector knobs}, knobs: {linac knobs},
   linac_id}
Rows sharing a linac_id have identical linac knobs (paired / crossed rows: the injector effect at
fixed linac knobs is an exact difference). beam = golden_h5 rows ("bridge" rows) track the golden
2024-10-14 h5 beam instead of Impact-T, at the same linac knobs: they measure the offset between
old golden-beam campaigns and this campaign's injector-at-golden point. They are written as
bridge_<idx>.json (NOT sample_*.json), so preprocess never trains on them.

Arrival time: "absolute" (the only supported mode for campaigns): the Impact-T beam's <t> shift at
L0AFEND relative to the campaign's golden-injector beam is KEPT (t -> t - <t>_golden), so every
downstream cavity sees the injector timing error. At injector = golden this is exactly the legacy
centred beam, which is what lets golden-beam campaigns embed at injector = golden.

Per row: sample_<idx>.json (+ sample_<idx>_<ELE>.h5 for save_beams_at) with the run_sweep schema
(idx, knobs, specs, success, ...) where knobs = linac + injector knobs (so preprocess with the
concatenated sweep set, e.g. xtcav23_inj12, reads all of them), plus injector_settings, beam,
linac_id, injected_charge_nC {drive, witness}, injector {cache_key, dt_l0afend_ps, ...}.
campaign.json records the injector fidelity block (fixed.injector) and the golden-equivalent beam.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"          # Bmad: one thread per worker (Phase-0 result)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import multiprocessing
import queue
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from .paths import facet2_root, repo_root
from .sweep_params import INJECTOR_KEYS, injector_golden, resolve_sweep_set

GOLDEN_BEAM_SUFFIX = "/beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5"
TREATY_POINTS = ["BEGBC20", "MFFF", "XTCAVF", "PENT"]
MAX_WORKERS = 64


# ---------------------------------------------------------------------------------------
# Config + manifest (pure: no Impact-T / Bmad imports)
# ---------------------------------------------------------------------------------------

def load_cfg(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["output_dir"])
    cfg["output_dir"] = str(out if out.is_absolute() else repo_root() / out)
    if cfg["injector"].get("arrival", "absolute") != "absolute":
        raise ValueError("injector campaigns must keep absolute arrival time "
                         "(centring erases the injector timing effect, ~6x too small)")
    return cfg


def _lhs(n, low, high, seed):
    from scipy.stats import qmc
    if n <= 0:
        return np.zeros((0, len(low)))
    return qmc.scale(qmc.LatinHypercube(d=len(low), seed=seed).random(n=n), low, high)


def _injector_point(delta: dict, golden: dict) -> dict:
    """Full injector dict: golden overridden by `delta` ({key: {abs: a} | {rel: r} | value})."""
    out = dict(golden)
    for k, d in (delta or {}).items():
        if k not in golden:
            raise KeyError(f"unknown injector knob {k!r}")
        if isinstance(d, dict):
            out[k] = golden[k] + float(d["abs"]) if "abs" in d else golden[k] * (1 + float(d["rel"]))
        else:
            out[k] = float(d)
    return out


def build_manifest(cfg) -> list[dict]:
    """Rows from cfg['design'] (written once to <out>/manifest.json and reused on resume).

    design:
      seed: int
      injector: {golden: bool, oat: [{key: {abs|rel}}...], lhs: n, box_scale: 1.0}
      linac:    {shared: {baseline: bool, lhs: n}, per_injector_lhs: k}
      pair_golden_frac: f     # fraction of per-injector rows re-run at injector = golden
      bridge: bool            # golden-h5 beam rows at the shared linac points
    Every injector point x every shared linac point (crossed), plus k independent linac LHS
    draws per LHS injector point (nested; one LHS over the linac box of size n_lhs*k, so the
    linac marginals stay stratified over the whole block)."""
    path = Path(cfg["output_dir"]) / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = make_rows(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    return rows


def make_rows(cfg) -> list[dict]:
    des = cfg["design"]
    seed = int(des["seed"])
    lkeys, llo, lhi, lbase = resolve_sweep_set(cfg["linac_sweep_set"])
    ikeys, ilo, ihi, _ = resolve_sweep_set(cfg["injector_sweep_set"])
    golden = injector_golden(ikeys)
    scale = float(des["injector"].get("box_scale", 1.0))
    ilo_s = [g - scale * (g - lo) for g, lo in zip(golden.values(), ilo)]
    ihi_s = [g + scale * (hi - g) for g, hi in zip(golden.values(), ihi)]

    inj_pts = []   # (tag, injector dict)
    if des["injector"].get("golden", True):
        inj_pts.append(("golden", dict(golden)))
    for j, d in enumerate(des["injector"].get("oat", []) or []):
        inj_pts.append((f"oat{j}", _injector_point(d, golden)))
    n_lhs = int(des["injector"].get("lhs", 0))
    for j, r in enumerate(_lhs(n_lhs, ilo_s, ihi_s, seed)):
        inj_pts.append((f"lhs{j}", dict(zip(ikeys, map(float, r)))))

    shared = []
    sh = des["linac"].get("shared", {}) or {}
    if sh.get("baseline", False):
        shared.append(dict(lbase))
    for r in _lhs(int(sh.get("lhs", 0)), llo, lhi, seed + 1):
        shared.append(dict(zip(lkeys, map(float, r))))

    rows, lid = [], 0

    def add(block, beam, inj, knobs, linac_id, tag):
        rows.append({"idx": len(rows), "block": block, "beam": beam, "injector_point": tag,
                     "injector": {k: float(v) for k, v in inj.items()},
                     "knobs": {k: float(v) for k, v in knobs.items()}, "linac_id": linac_id})

    # crossed: every injector point x every shared linac point
    for s_i, knobs in enumerate(shared):
        for tag, inj in inj_pts:
            add("crossed", "impact", inj, knobs, s_i, tag)
    lid = len(shared)
    # nested: k independent linac draws per LHS injector point
    k = int(des["linac"].get("per_injector_lhs", 0))
    lhs_pts = [(t, p) for t, p in inj_pts if t.startswith("lhs")]
    draws = _lhs(len(lhs_pts) * k, llo, lhi, seed + 2)
    if len(draws):
        draws = draws[np.random.default_rng(seed + 3).permutation(len(draws))]
    nested = []
    for j, (tag, inj) in enumerate(lhs_pts):
        for m in range(k):
            knobs = dict(zip(lkeys, map(float, draws[j * k + m])))
            add("nested", "impact", inj, knobs, lid, tag)
            nested.append(rows[-1])
            lid += 1
    # paired: a fraction of nested rows re-run at injector = golden (same linac knobs)
    f = float(des.get("pair_golden_frac", 0.0))
    if f > 0 and nested:
        pick = np.random.default_rng(seed + 4).choice(len(nested), round(f * len(nested)), replace=False)
        for j in sorted(pick):
            r = nested[j]
            add("paired_golden", "impact", golden, r["knobs"], r["linac_id"], "golden")
    # bridge: the golden h5 beam at the shared linac points
    if des.get("bridge", False):
        for s_i, knobs in enumerate(shared):
            add("bridge", "golden_h5", golden, knobs, s_i, "golden_h5")
    return rows


def injector_points(rows) -> dict:
    """{cache-independent key: injector dict} of the Impact-T beams the manifest needs."""
    out = {}
    for r in rows:
        if r["beam"] == "impact":
            out[json.dumps(r["injector"], sort_keys=True)] = r["injector"]
    return out


def absolute_time_shift(info: dict, ref_info: dict) -> float:
    """Seconds to ADD to a centred injector beam so its <t> keeps the shift vs the golden beam."""
    return float(info["mean_t_l0afend_s"]) - float(ref_info["mean_t_l0afend_s"])


def row_json_path(out_dir, row) -> Path:
    stem = "bridge" if row["beam"] == "golden_h5" else "sample"
    return Path(out_dir) / f"{stem}_{row['idx']:05d}.json"


def _is_done(path) -> bool:
    try:
        return "success" in json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False


# ---------------------------------------------------------------------------------------
# campaign.json
# ---------------------------------------------------------------------------------------

def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def campaign_record(cfg, baseline_settings: dict, l0af_reference: dict, ref_info: dict,
                    golden_beam_sha256: str | None, file_shas: dict | None = None,
                    provenance: dict | None = None) -> dict:
    """campaign.json record (pure). file_shas: {baseline_config_sha256, defaults_yml_sha256}."""
    lkeys, llo, lhi, _ = resolve_sweep_set(cfg["linac_sweep_set"])
    ikeys, ilo, ihi, _ = resolve_sweep_set(cfg["injector_sweep_set"])
    held = {k: v for k, v in injector_golden().items() if k not in ikeys}
    inj = cfg["injector"]
    return {
        "name": cfg["name"], "config": cfg,
        "swept": {"sweep_set": cfg["sweep_set"], "knob_keys": lkeys + ikeys,
                  "knob_low": llo + ilo, "knob_high": lhi + ihi,
                  "linac_sweep_set": cfg["linac_sweep_set"],
                  "injector_sweep_set": cfg["injector_sweep_set"]},
        "fixed": {
            "baseline_config": cfg["baseline_config"],
            "baseline_config_sha256": (file_shas or {}).get("baseline_config_sha256"),
            "defaults_yml_sha256": (file_shas or {}).get("defaults_yml_sha256"),
            "baseline_settings": baseline_settings,
            "fixed_overrides": cfg.get("fixed_overrides", {}),
            "input_beam": "impact-t (sim/injector.py), per sample",
            "input_beam_sha256": None,
            "csrTF": cfg["csrTF"], "transverseWakes": cfg.get("transverseWakes", False),
            "num_macro_particles": int(inj["n_particles"]),
            "treaty_points": cfg.get("treaty_points", TREATY_POINTS),
            "injector_settings": held,                     # injector knobs NOT swept here
            "injector": {
                "source": "impact",
                "n_particles": int(inj["n_particles"]), "grid": int(inj["grid"]),
                "code_version": ref_info.get("code_version"),
                "arrival": "absolute",
                "hammersley_burnin": int(inj.get("hammersley_burnin", -1)),
                "l0af_reference": l0af_reference,
                "reference_cache_key": ref_info.get("cache_key"),
                "reference_mean_t_l0afend_s": ref_info.get("mean_t_l0afend_s"),
                "golden_equivalent_input_beam": GOLDEN_BEAM_SUFFIX,
                "golden_equivalent_input_beam_sha256": golden_beam_sha256,
            },
        },
        **(provenance or {}),
    }


def write_campaign_json(cfg, l0af_reference, ref_info) -> Path:
    out = Path(cfg["output_dir"]) / "campaign.json"
    if out.exists():
        return out
    import FACET2_S2E as qs
    from ..artifacts import facet2_s2e_hash, git_state
    root = facet2_root()
    baseline = qs.loadConfig("/" + cfg["baseline_config"], str(root))
    rec = campaign_record(cfg, baseline, l0af_reference, ref_info,
                          _sha(str(root) + GOLDEN_BEAM_SUFFIX),
                          {"baseline_config_sha256": _sha(root / cfg["baseline_config"]),
                           "defaults_yml_sha256": _sha(root / "setLattice_configs" / "defaults.yml")},
                          {"facet2_s2e_hash": facet2_s2e_hash(), **git_state()})
    out.write_text(json.dumps(rec, indent=2, default=float))
    return out


# ---------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------

_CFG = None
_TAO = None
_BASELINE = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _settings(inj: dict) -> dict:
    """Injector knobs + campaign-held injector settings -> sim.injector settings dict."""
    return {**inj, "hammersley_burnin": int(_CFG["injector"].get("hammersley_burnin", -1))}


def _beam_dir(inj: dict, ref: dict) -> Path:
    from ..sim import injector as sim_inj
    c = _CFG["injector"]
    key = sim_inj.cache_key(sim_inj.resolve_settings(_settings(inj)), int(c["n_particles"]),
                            int(c["grid"]), ref)
    return Path(_CFG["output_dir"]) / "beams" / key


def impact_task(args):
    inj, ref = args
    from ..sim import injector as sim_inj
    c = _CFG["injector"]
    wd = _beam_dir(inj, ref)
    t0 = time.perf_counter()
    try:
        _, info = sim_inj.injector_beam(_settings(inj), wd, int(c["n_particles"]), grid=int(c["grid"]),
                                        l0af_reference=ref)
        return json.dumps(inj, sort_keys=True), True, time.perf_counter() - t0, info
    except Exception:
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "error.txt").write_text(traceback.format_exc())
        return json.dumps(inj, sort_keys=True), False, time.perf_counter() - t0, traceback.format_exc()


def _get_tao():
    global _TAO, _BASELINE
    if _TAO is None:
        import FACET2_S2E as qs
        from ..sim.bmad import make_tao
        root = facet2_root()
        _BASELINE = qs.loadConfig("/" + _CFG["baseline_config"], str(root))
        _TAO = make_tao(qs, root, _BASELINE, csr=_CFG["csrTF"], wakes=_CFG.get("transverseWakes", False),
                        n_particles=int(_CFG["injector"]["n_particles"]),
                        scratch=Path(_CFG["output_dir"]) / "scratch" / f"pid{os.getpid()}")
    return _TAO


def golden_downsample(n: int, seed: int) -> Path:
    """Seeded version of modifyAndSaveInputBeam's random downsample of the golden 5e5 beam."""
    from pmd_beamphysics import ParticleGroup
    path = Path(_CFG["output_dir"]) / "beams" / f"golden_h5_N{n}_seed{seed}.h5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        P = ParticleGroup(str(facet2_root()) + GOLDEN_BEAM_SUFFIX)
        n0 = len(P)
        P = P[random.Random(seed).sample(range(n0), n)]
        P.weight = P.weight * (n0 / n)
        tmp = path.with_suffix(f".tmp{os.getpid()}.h5")
        P.write(str(tmp))
        os.replace(tmp, path)
    return path


def bmad_task(args):
    """Track one manifest row; retry once with a fresh Tao. Failures are recorded, not dropped."""
    row, ref, ref_info = args
    global _TAO
    import FACET2_S2E as qs
    from FACET2_S2E.beam.manipulation import modifyAndSaveInputBeam
    from pmd_beamphysics import ParticleGroup
    from ..sim.bmad import merged_settings, track

    out_dir = Path(_CFG["output_dir"])
    n = int(_CFG["injector"]["n_particles"])
    jpath = row_json_path(out_dir, row)
    result = {"idx": row["idx"], "block": row["block"], "beam": row["beam"],
              "injector_point": row["injector_point"], "linac_id": row["linac_id"],
              "knobs": {**row["knobs"], **row["injector"]},
              "linac_knobs": row["knobs"], "injector_settings": row["injector"],
              "fixed_overrides": _CFG.get("fixed_overrides", {}),
              "num_macro_particles": n, "csrTF": _CFG["csrTF"],
              "transverseWakes": _CFG.get("transverseWakes", False),
              "pid": os.getpid(), "timestamp": time.time()}
    for attempt in range(2):
        try:
            tao = _get_tao()
            scratch = Path(tao.activeFilePath).parent
            if row["beam"] == "golden_h5":
                src = golden_downsample(n, int(_CFG["design"]["seed"]))
                centre = True
                result["injector"] = {"source": "golden_h5"}
            else:
                wd = _beam_dir(row["injector"], ref)
                info = json.loads((wd / "injector_info.json").read_text())
                dt = absolute_time_shift(info, ref_info)
                P_in = ParticleGroup(str(wd / "beam_L0AFEND.h5"))
                P_in.t = P_in.t + dt
                src = scratch / f"in_{row['idx']:05d}.h5"
                P_in.write(str(src))
                centre = False
                result["injector"] = {
                    "source": "impact", "cache_key": info["cache_key"],
                    "dt_l0afend_ps": dt * 1e12, "mean_t_l0afend_s": info["mean_t_l0afend_s"],
                    "mean_energy_eV": info["mean_energy_eV"], "charge_C": info["charge_C"],
                    "n_drive": info["n_drive"], "n_witness": info["n_witness"],
                    "n_ambiguous": info["n_ambiguous"], "impact_wall_s": info["wall_s"]}
            P_in = ParticleGroup(str(src))
            w = np.unique(P_in.weight)
            result["injected_charge_nC"] = {
                "drive": float(np.sum(P_in.weight[P_in.weight == w[-1]])) * 1e9,
                "witness": float(np.sum(P_in.weight[P_in.weight == w[0]])) * 1e9}
            # equal counts -> random.sample is a pure permutation (no downsample)
            modifyAndSaveInputBeam(str(src), numMacroParticles=n, timeCenterTF=centre,
                                   outputBeamFilePath=tao.activeFilePath)
            if row["beam"] != "golden_h5":
                os.remove(src)
            merged = merged_settings(_BASELINE, _CFG.get("fixed_overrides", {}), row["knobs"])
            t0 = time.perf_counter()
            stem = jpath.with_suffix("")
            rec = track(qs, tao, facet2_root(), merged, _CFG.get("treaty_points", TREATY_POINTS),
                        n_in=n, save_beams_at=_CFG.get("save_beams_at", ["XTCAVF"]),
                        out_prefix=str(stem))
            result["wall_s"] = time.perf_counter() - t0
            result.update(rec)
            result["success"] = True
            result.pop("error", None)
            break
        except Exception:
            result["success"] = False
            result["error"] = f"attempt {attempt}: {traceback.format_exc()}"
            _TAO = None
    jpath.write_text(json.dumps(result, indent=2, default=float))
    return row["idx"], result["success"], result.get("wall_s")


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

def run(cfg, workers: int):
    from ..sim import injector as sim_inj
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(cfg)
    ref = sim_inj.pin_l0af_reference(out / "l0af_reference.json", out / "l0af_autophase")
    golden_key = json.dumps(injector_golden(resolve_sweep_set(cfg["injector_sweep_set"])[0]),
                            sort_keys=True)
    todo = [r for r in rows if not _is_done(row_json_path(out, r))]
    points = injector_points(rows)
    points.setdefault(golden_key, json.loads(golden_key))     # reference beam always needed
    _init_worker(cfg)                                          # parent uses _beam_dir too
    need = {k: p for k, p in points.items()
            if not (_beam_dir(p, ref) / "injector_info.json").exists()}
    print(f"[{cfg['name']}] {len(rows)} rows, {len(todo)} to run; {len(points)} injector beams "
          f"({len(need)} to generate); N={cfg['injector']['n_particles']}, "
          f"grid={cfg['injector']['grid']}, {workers} workers", flush=True)

    events: queue.Queue = queue.Queue()
    t_start = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")   # fork deadlocks (polars / pytao thread pools)
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        ready = {k for k in points if k not in need}
        failed = set()
        for k, p in need.items():
            pool.apply_async(impact_task, ((p, ref),), callback=lambda r: events.put(("impact", r)),
                             error_callback=lambda e: events.put(("crash", e)))
        pending = {r["idx"]: r for r in todo}
        submitted = set()
        n_done = n_fail = 0

        def submit_ready():
            if golden_key not in ready:
                ref_info = None
            else:
                ref_info = json.loads((_beam_dir(points[golden_key], ref) / "injector_info.json").read_text())
            for idx, r in list(pending.items()):
                if idx in submitted:
                    continue
                key = json.dumps(r["injector"], sort_keys=True)
                if r["beam"] == "golden_h5" or (ref_info is not None and key in ready):
                    submitted.add(idx)
                    pool.apply_async(bmad_task, ((r, ref, ref_info),),
                                     callback=lambda res: events.put(("bmad", res)),
                                     error_callback=lambda e: events.put(("crash", e)))
                elif key in failed or (golden_key in failed):
                    submitted.add(idx)
                    pending.pop(idx)
                    print(f"  [skip] idx={idx:05d}: injector beam failed", flush=True)

        submit_ready()
        while pending:
            kind, res = events.get()
            el = (time.perf_counter() - t_start) / 60
            if kind == "crash":
                print(f"  [crash] {res!r}", flush=True)
                continue
            if kind == "impact":
                key, ok, wall, info = res
                (ready if ok else failed).add(key)
                print(f"  [impact] {'ok' if ok else 'FAIL'} {wall:.0f}s "
                      f"{'' if ok else str(info)[-300:]} | {el:.1f} min", flush=True)
                if ok and key == golden_key:
                    write_campaign_json(cfg, ref, info)
                submit_ready()
                continue
            idx, ok, wall = res
            pending.pop(idx, None)
            n_done += 1
            n_fail += not ok
            print(f"  [bmad {n_done}/{len(todo)}] idx={idx:05d} {'ok' if ok else 'FAIL'} "
                  f"{wall and f'{wall:.0f}s' or ''} | {el:.1f} min", flush=True)
    if golden_key in ready:
        info = json.loads((_beam_dir(points[golden_key], ref) / "injector_info.json").read_text())
        write_campaign_json(cfg, ref, info)
    print(f"Done: {n_done - n_fail} ok, {n_fail} failed, "
          f"{(time.perf_counter() - t_start) / 3600:.2f} h wall. Output: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    workers = min(int(a.workers or cfg["n_workers"]), MAX_WORKERS)
    if a.dry_run:
        rows = make_rows(cfg)
        from collections import Counter
        print(f"{len(rows)} rows; blocks {dict(Counter(r['block'] for r in rows))}; "
              f"{len(injector_points(rows))} Impact-T beams")
        return
    run(cfg, workers)


if __name__ == "__main__":
    sys.exit(main())
