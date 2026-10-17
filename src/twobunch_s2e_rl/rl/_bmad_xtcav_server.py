"""Bmad tracking server for the closed-loop delta-controller check (runs under bmad-qpad-dev; the client is
rl/closed_loop_bmad.py in slac-rl). One JSON request per stdin line; each answer is one stdout line
`BMADXT_RESULT <json>` (Tao's own stdout noise is ignored by the client).

  {"cmd": "prepare", "points": [{"key": k, "injector": {...inj12 knobs...}}]}
      per injector point: its cached Impact-T beam at L0AFEND (injector campaign `--campaign-cfg`, same
      cache key rule), absolute arrival kept (shift vs the golden beam, as the campaign does), ONE seeded
      downsample to --num-macro (weights rescaled) -> {key: input h5 path}. Fixed per point, so every
      step of an episode tracks the same macroparticles (no unseeded per-track downsample).
  {"cmd": "track", "tasks": [{"id": i, "knobs": {...linac knobs...}, "beam": h5, "seed": s, "out": prefix}]}
      setLattice + trackBeam to XTCAVF with the campaign physics (baseline config, fixed overrides, CSR,
      wakes), Tao random seed `seed` (fixed per episode: common random numbers across its steps); saves
      <prefix>_XTCAVF.h5 -> [{id, ok, h5, n_in_drive, n_in_witness, n_live_drive, n_live_witness, wall_s,
      error}]
  {"cmd": "quit"}
Workers: --workers single-thread processes (spawn; OMP_NUM_THREADS=1 in the environment).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

READY = "BMADXT_READY"
RESULT = "BMADXT_RESULT "
_W: dict = {}


def _say(obj) -> None:
    sys.stdout.write(RESULT + json.dumps(obj, default=float) + "\n")
    sys.stdout.flush()


def _init(cfg, n_macro, scratch):
    os.environ["OMP_NUM_THREADS"] = "1"
    import FACET2_S2E as qs
    from ..datagen.paths import facet2_root
    from ..sim.bmad import make_tao
    root = facet2_root()
    base = qs.loadConfig("/" + cfg["baseline_config"], str(root))
    tao = make_tao(qs, root, base, csr=cfg["csrTF"], wakes=cfg.get("transverseWakes", False),
                   n_particles=n_macro, scratch=Path(scratch) / f"pid{os.getpid()}")
    _W.update(qs=qs, root=root, base=base, cfg=cfg, tao=tao)


def _counts(P):
    w = np.unique(P.weight)
    if len(w) < 2:
        return len(P.x), 0
    return int((P.weight == w[-1]).sum()), int((P.weight == w[0]).sum())


def _track(task):
    from FACET2_S2E.beam.manipulation import modifyAndSaveInputBeam
    from pmd_beamphysics import ParticleGroup
    from ..sim.bmad import merged_settings, track
    qs, tao, cfg = _W["qs"], _W["tao"], _W["cfg"]
    out = {"id": task["id"], "ok": False}
    t0 = time.perf_counter()
    try:
        tao.cmd(f"set global random_seed = {int(task['seed'])}")
        P_in = ParticleGroup(task["beam"])
        out["n_in_drive"], out["n_in_witness"] = _counts(P_in)
        modifyAndSaveInputBeam(task["beam"], numMacroParticles=None, timeCenterTF=False,
                               outputBeamFilePath=tao.activeFilePath)
        settings = merged_settings(_W["base"], cfg.get("fixed_overrides", {}), task["knobs"])
        rec = track(qs, tao, _W["root"], settings, ["XTCAVF"], n_in=len(P_in.x),
                    save_beams_at=("XTCAVF",), out_prefix=task["out"])
        sp = rec["specs"]["XTCAVF"]
        out["n_live_drive"] = int(sp.get("n_live_drive", sp.get("n_live_total", 0)))
        out["n_live_witness"] = int(sp.get("n_live_witness", 0))
        h5 = task["out"] + "_XTCAVF.h5"
        out["h5"] = h5 if os.path.exists(h5) else None
        out["ok"] = out["h5"] is not None
    except Exception:  # noqa: BLE001 - reported to the client
        out["error"] = traceback.format_exc()[-2000:]
    out["wall_s"] = time.perf_counter() - t0
    return out


def _prepare(points, cfg, n_macro, scratch):
    from pmd_beamphysics import ParticleGroup
    from ..datagen import injector_campaign as ic
    from ..datagen.sweep_params import injector_golden, resolve_sweep_set
    from ..sim import injector as sim_inj
    out_dir = Path(cfg["output_dir"])
    ic._init_worker(cfg)
    ref = sim_inj.pin_l0af_reference(out_dir / "l0af_reference.json", out_dir / "l0af_autophase")
    golden = injector_golden(resolve_sweep_set(cfg["injector_sweep_set"])[0])
    ref_info = json.loads((ic._beam_dir(golden, ref) / "injector_info.json").read_text())
    paths = {}
    for p in points:
        dst = Path(scratch) / "inputs" / f"{p['key']}_N{n_macro}.h5"
        if not dst.exists():
            wd = ic._beam_dir(p["injector"], ref)
            info = json.loads((wd / "injector_info.json").read_text())
            P = ParticleGroup(str(wd / "beam_L0AFEND.h5"))
            P.t = P.t + ic.absolute_time_shift(info, ref_info)
            n0 = len(P.x)
            if n0 > n_macro:
                idx = np.sort(np.random.default_rng(12345).choice(n0, n_macro, replace=False))
                P = P[idx]
                P.weight = P.weight * (n0 / n_macro)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.h5")
            P.write(str(tmp))
            os.replace(tmp, dst)
        paths[p["key"]] = str(dst)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-cfg", required=True, help="injector campaign yaml (physics + beam cache)")
    ap.add_argument("--num-macro", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    from ..datagen import injector_campaign as ic
    cfg = ic.load_cfg(a.campaign_cfg)
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(cfg, a.num_macro, a.scratch)) as pool:
        sys.stdout.write(READY + "\n")
        sys.stdout.flush()
        for line in sys.stdin:
            req = json.loads(line)
            if req["cmd"] == "quit":
                break
            if req["cmd"] == "prepare":
                try:
                    _say(_prepare(req["points"], cfg, a.num_macro, a.scratch))
                except Exception:  # noqa: BLE001
                    _say({"error": traceback.format_exc()[-2000:]})
            elif req["cmd"] == "track":
                _say(pool.map(_track, req["tasks"], chunksize=1))
            else:
                _say({"error": f"unknown cmd {req['cmd']!r}"})


if __name__ == "__main__":
    main()
