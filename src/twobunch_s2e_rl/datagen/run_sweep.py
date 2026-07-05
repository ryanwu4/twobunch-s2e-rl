"""LHS sweep driver for the two-bunch NF-surrogate data campaign.

Usage (bmad-qpad-dev env, from the repo root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
    -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/smoke.yaml

Per sample: setLattice(baseline + LHS knobs) -> trackBeam (L0AFEND->end, CSR per config)
-> per-bunch specs at BEGBC20/MFFF/PENT -> sample_{idx:05d}.json + PENT beam h5.
One persistent Tao per worker (init ~1 s, but reuse avoids beam-file churn); resume by
re-running the same command — completed samples are skipped via their json files.

The FACET2-S2E checkout (Bmad lattice / beams / setLattice configs) is located via
paths.facet2_root(); a relative output_dir in the config is resolved against the repo root
(paths.repo_root()), i.e. data/<name>.
"""

import os

# Phase-0 thread-scaling result: Bmad's OpenMP saturates ~8-32 threads and unrestricted
# workers would oversubscribe n_workers x 128 threads. Must be set before pytao loads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from .sweep_params import resolve_sweep_set
from .ff_manifold import MANIFOLD_SPECS, sample_anchored_ff
from .paths import facet2_root, repo_root

TREATY_POINTS = ["BEGBC20", "MFFF", "PENT"]

# Per-worker globals (set in _init_worker)
_TAO = None
_BASELINE = None
_CFG = None
_S2E_ROOT = None  # FACET2-S2E checkout root; resolved per worker (defers FACET2_S2E import)


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    cfg["output_dir"] = str(out_dir)
    return cfg


def build_manifest(cfg):
    """LHS over the configured sweep set's bounds + optional baseline repeats. Deterministic
    by seed; written once and reused on resume so indices never reshuffle. The set is
    cfg["sweep_set"] (default "original8"). A set listed in ff_manifold.MANIFOLD_SPECS instead
    uses manifold-anchored sampling for its FF quads (see _manifold_rows)."""
    manifest_path = Path(cfg["output_dir"]) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    set_name = cfg.get("sweep_set", "original8")
    keys, low, high, baseline = resolve_sweep_set(set_name)
    n = cfg["n_samples"]
    spec = MANIFOLD_SPECS.get(set_name)

    if n <= 0:
        rows = []  # baseline-repeats-only config (e.g. the transverse-wakes gate)
    elif spec is None:
        sampler = qmc.LatinHypercube(d=len(keys), seed=cfg["seed"])
        scaled = qmc.scale(sampler.random(n=n), low, high)
        rows = [{"knobs": dict(zip(keys, map(float, r)))} for r in scaled]
    else:
        rows = _manifold_rows(cfg["seed"], n, keys, low, high, spec)

    manifest = [{"idx": i, "is_baseline_repeat": False, **row} for i, row in enumerate(rows)]
    for j in range(cfg.get("n_baseline_repeats", 0)):
        manifest.append(
            {"idx": n + j, "knobs": dict(baseline), "is_baseline_repeat": True}
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _manifold_rows(seed, n, keys, low, high, spec):
    """Stratified manifold-anchored rows. ~(1-tail) drawn near the deliverable manifold (FF on
    the matched beta-curve + jitter, transverse knobs in their narrowed ranges) and ~tail from
    the wide transverse box (spec['wide_set']) for feasibility/recovery coverage. Deterministic
    by seed. Each row carries a "block" tag; anchor rows also carry their FF target beta."""
    n_tail = round(spec["stratify_tail_frac"] * n)
    n_anchor = n - n_tail
    ff_idx = [keys.index(k) for k in spec["ff_keys"]]
    rng = np.random.default_rng(seed)
    rows = []

    if n_anchor > 0:
        a = qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed).random(n=n_anchor), low, high)
        beta, ff = sample_anchored_ff(rng, n_anchor, spec["jitter_frac"],
                                      spec["beta_lo"], spec["beta_hi"])
        a[:, ff_idx] = ff
        rows += [{"knobs": dict(zip(keys, map(float, r))), "block": "anchor",
                  "ff_target_beta_m": float(b)} for r, b in zip(a, beta)]

    if n_tail > 0:
        wkeys, wlow, whigh, _ = resolve_sweep_set(spec["wide_set"])
        wlow = [wlow[wkeys.index(k)] for k in keys]   # realign to this set's key order
        whigh = [whigh[wkeys.index(k)] for k in keys]
        t = qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed + 1).random(n=n_tail), wlow, whigh)
        rows += [{"knobs": dict(zip(keys, map(float, r))), "block": "tail",
                  "ff_target_beta_m": None} for r in t]

    return rows


def _sample_json_path(out_dir, idx):
    return Path(out_dir) / f"sample_{idx:05d}.json"


def _is_done(out_dir, idx):
    path = _sample_json_path(out_dir, idx)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            return "success" in json.load(f)
    except (json.JSONDecodeError, OSError):
        return False  # corrupt partial write -> redo


def _init_worker(cfg):
    global _TAO, _BASELINE, _CFG, _S2E_ROOT
    _CFG = cfg
    _S2E_ROOT = facet2_root()
    import FACET2_S2E as qs  # heavyweight import deferred to workers

    _BASELINE = qs.loadConfig("/" + cfg["baseline_config"], str(_S2E_ROOT))
    _TAO = _make_tao(qs)


def _make_tao(qs):
    return qs.initializeTao(
        filePath=str(_S2E_ROOT),
        inputBeamFilePathSuffix=_BASELINE["inputBeamFilePathSuffix"],
        csrTF=_CFG["csrTF"],
        transverseWakes=_CFG.get("transverseWakes", False),
        numMacroParticles=_CFG["num_macro_particles"],
        scratchPath=str(Path(_CFG["output_dir"]) / "scratch"),
        randomizeFileNames=True,
    )


def _track_and_collect(qs, item):
    merged = {**_BASELINE, **item["knobs"]}
    t0 = time.perf_counter()
    qs.setLattice(_TAO, **merged)
    qs.trackBeam(_TAO, str(_S2E_ROOT), **merged)
    wall_s = time.perf_counter() - t0

    n_in = int(_CFG["num_macro_particles"])
    record = {"wall_s": wall_s, "specs": {}}
    for ele in TREATY_POINTS:
        P = qs.getBeamAtElement(_TAO, ele)

        # Save the beam before computing specs: degenerate beams (a fully-scraped
        # bunch) can crash getBeamSpecs, and the cloud itself is feasibility data.
        if ele in _CFG.get("save_beams_at", ["PENT"]) and len(P.x):
            P.write(str(Path(_CFG["output_dir"]) / f"sample_{item['idx']:05d}_{ele}.h5"))

        specs = {"n_live_total": len(P.x), "transmission_total": len(P.x) / n_in}
        if len(np.unique(P.weight)) == 2:
            PDrive, PWitness = qs.getDriverAndWitness(P)
            specs["n_live_drive"] = len(PDrive.x)
            specs["n_live_witness"] = len(PWitness.x)
        try:
            specs.update(qs.getBeamSpecs(P, targetTwiss=ele) or {})
        except Exception as e:
            specs["specs_error"] = f"{type(e).__name__}: {e}"
        record["specs"][ele] = specs
    return record


def run_sample(item):
    """Track one knob setting. Retries once with a fresh Tao on any error; a failed
    sample is recorded, not discarded (failures are feasibility-head data)."""
    global _TAO
    import FACET2_S2E as qs

    result = {
        "idx": item["idx"],
        "knobs": item["knobs"],
        "is_baseline_repeat": item["is_baseline_repeat"],
        "num_macro_particles": int(_CFG["num_macro_particles"]),
        "csrTF": _CFG["csrTF"],
        "transverseWakes": _CFG.get("transverseWakes", False),
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    for attempt in range(2):
        try:
            record = _track_and_collect(qs, item)
            result.update(record)
            result["success"] = True
            break
        except Exception as e:
            result["success"] = False
            result["error"] = f"attempt {attempt}: {type(e).__name__}: {e}"
            if attempt == 0:
                try:
                    _TAO = _make_tao(qs)  # Tao may be in a bad state; rebuild
                except Exception as e2:
                    result["error"] += f" | tao reinit failed: {e2}"
                    break

    with open(_sample_json_path(_CFG["output_dir"], item["idx"]), "w") as f:
        json.dump(result, f, indent=2, default=to_jsonable)
    return item["idx"], result["success"], result.get("wall_s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a sweep yaml (configs/datagen/*.yaml)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(cfg)
    todo = [item for item in manifest if not _is_done(out_dir, item["idx"])]
    n_total, n_todo = len(manifest), len(todo)
    print(f"[{cfg['name']}] {n_total} samples in manifest, {n_total - n_todo} done, {n_todo} to run "
          f"({cfg['n_workers']} workers, {int(cfg['num_macro_particles'])} particles, "
          f"CSR={'on' if cfg['csrTF'] else 'off'}, wakes={'on' if cfg.get('transverseWakes') else 'off'})")
    if not todo:
        print("All samples complete.")
        return

    t_start = time.perf_counter()
    n_done = n_fail = 0
    with multiprocessing.Pool(cfg["n_workers"], initializer=_init_worker, initargs=(cfg,)) as pool:
        for idx, success, wall_s in pool.imap_unordered(run_sample, todo, chunksize=1):
            n_done += 1
            n_fail += not success
            elapsed = time.perf_counter() - t_start
            rate = n_done / elapsed
            eta_h = (n_todo - n_done) / rate / 3600 if rate > 0 else float("inf")
            print(f"  [{n_done}/{n_todo}] idx={idx:05d} "
                  f"{'ok' if success else 'FAIL'} {wall_s and f'{wall_s:.0f}s' or ''} "
                  f"| {rate * 3600:.0f} samples/h, ETA {eta_h:.1f} h", flush=True)

    print(f"\nDone: {n_done - n_fail} ok, {n_fail} failed, "
          f"{(time.perf_counter() - t_start) / 3600:.2f} h wall. Output: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
