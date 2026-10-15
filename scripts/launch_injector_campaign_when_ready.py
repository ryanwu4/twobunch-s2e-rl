"""Wait for the Impact-T convergence study and the realization-noise check, apply the pre-agreed
pass rules, pick the campaign fidelity, write + commit the launch config, and launch the campaign.

Pass rules (physics review #5 / round 2), all at BOTH linac points (baseline, compressed):
  realization noise (golden, burnin -1 vs burnin 1000003, 1e5/g64): |d spacing|, |d drive sigma_z|
      <= 0.3 um, else NO launch.
  grid: reference = 1e5 / grid 100. A candidate grid passes if
      absolute: |spacing - ref| <= 0.5 um, |drive sigma_z / ref - 1| <= 1.5%
      effects (timing, delay, gun amplitude, drive fraction 0.80; perturbed - golden at the same
      resolution): spacing within max(10%, 0.3 um), drive sigma_z within max(10%, 0.15 um).
      Order: grid 64, then (48, 48, 128); none passes -> NO launch.
  particles: 5e4 at the chosen grid replaces 1e5 if it passes the same rules against 1e5 at that
      grid (only grid 64 was run at 5e4; an anisotropic choice keeps 1e5).
Writes results/injector_convergence_2026-10-14/decision.json; on a pass writes
configs/datagen/injector_full_launch.yaml (= injector_full.yaml + chosen fidelity), commits it,
and launches the campaign (logs/injector_full/run.log, run.pid).

  nohup python scripts/launch_injector_campaign_when_ready.py > logs/injector_full/launcher.log &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BMAD = REPO / "data" / "injector_convergence_2026-10-14" / "bmad"
RES = REPO / "results" / "injector_convergence_2026-10-14"
CFG = REPO / "configs" / "datagen" / "injector_full.yaml"
LAUNCH_CFG = REPO / "configs" / "datagen" / "injector_full_launch.yaml"
LOG = REPO / "logs" / "injector_full"
PY = "/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python"
EFFECTS = ("timing+0.2ps", "delay+0.2ps", "gunamp+0.25%", "dfrac0.80")
LINAC = ("baseline", "compressed")


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def truth(n, g, point, lin):
    p = BMAD / f"N{n}_g{g}_{point}_{lin}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["truth"] if d.get("ok") and not d["truth"].get("witness_lost") else None


def compare(cand, ref):
    """(pass, details) of resolution `cand` = (n, gtag) against `ref`."""
    det, ok = {}, True
    for lin in LINAC:
        c0, r0 = truth(*cand, "golden", lin), truth(*ref, "golden", lin)
        if c0 is None or r0 is None:
            return False, {"missing": f"golden {lin}"}
        ds = 1e6 * (c0["bunch_spacing"] - r0["bunch_spacing"])
        dz = c0["drive_sigma_z"] / r0["drive_sigma_z"] - 1
        det[f"{lin}/abs"] = {"d_spacing_um": round(ds, 3), "d_drive_sigma_z_rel": round(dz, 4)}
        ok &= abs(ds) <= 0.5 and abs(dz) <= 0.015
        for e in EFFECTS:
            c1, r1 = truth(*cand, e, lin), truth(*ref, e, lin)
            if c1 is None or r1 is None:
                return False, {"missing": f"{e} {lin}"}
            es_c, es_r = 1e6 * (c1["bunch_spacing"] - c0["bunch_spacing"]), 1e6 * (r1["bunch_spacing"] - r0["bunch_spacing"])
            ez_c, ez_r = 1e6 * (c1["drive_sigma_z"] - c0["drive_sigma_z"]), 1e6 * (r1["drive_sigma_z"] - r0["drive_sigma_z"])
            ps = abs(es_c - es_r) <= max(0.1 * abs(es_r), 0.3)
            pz = abs(ez_c - ez_r) <= max(0.1 * abs(ez_r), 0.15)
            det[f"{lin}/{e}"] = {"spacing_um": [round(es_c, 3), round(es_r, 3)], "drive_sz_um": [round(ez_c, 3), round(ez_r, 3)],
                                 "pass": bool(ps and pz)}
            ok &= ps and pz
    return bool(ok), det


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    while running("injector_convergence.py") or running("injector_realization_noise.py"):
        time.sleep(300)
    RES.mkdir(parents=True, exist_ok=True)
    dec = {"time": time.time()}
    # realization noise
    noise_ok = True
    for lin in LINAC:
        a, b = truth(100000, 64, "golden", lin), truth(100000, 64, "golden_burnin2", lin)
        if a is None or b is None:
            noise_ok = False
            dec[f"noise/{lin}"] = "missing"
            continue
        d = {k: round(1e6 * (b[k] - a[k]), 3) for k in ("bunch_spacing", "drive_sigma_z")}
        dec[f"noise/{lin}"] = d
        noise_ok &= abs(d["bunch_spacing"]) <= 0.3 and abs(d["drive_sigma_z"]) <= 0.3
    dec["noise_pass"] = noise_ok
    ref = (100000, 100)
    choice = None
    for gtag, grid in (("64", 64), ("48x48x128", [48, 48, 128])):
        ok, det = compare((100000, gtag), ref)
        dec[f"grid_{gtag}"] = {"pass": ok, **det}
        if ok:
            choice = {"n_particles": 100000, "grid": grid, "gtag": gtag}
            break
    if choice and choice["gtag"] == "64":
        ok, det = compare((50000, "64"), (100000, "64"))
        dec["n_5e4_vs_1e5"] = {"pass": ok, **det}
        if ok:
            choice["n_particles"] = 50000
    dec["choice"] = choice
    launch = bool(noise_ok and choice)
    dec["launch"] = launch
    (RES / "decision.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1), flush=True)
    if not launch:
        print("NOT launching: a pass rule failed (see decision.json)", flush=True)
        return 1
    cfg = yaml.safe_load(CFG.read_text())
    cfg["injector"]["n_particles"] = choice["n_particles"]
    cfg["injector"]["grid"] = choice["grid"]
    cfg["output_dir"] = str(REPO / "data" / "injector_full")      # absolute: the run is from a worktree
    head = ("# GENERATED by scripts/launch_injector_campaign_when_ready.py from injector_full.yaml with the\n"
            f"# fidelity chosen by the convergence pass rules (results/injector_convergence_2026-10-14/decision.json):\n"
            f"# n_particles {choice['n_particles']}, grid {choice['grid']}.\n")
    LAUNCH_CFG.write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run(["git", "add", str(LAUNCH_CFG), str(RES / "decision.json")], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"injector_full launch config: n_particles {choice['n_particles']}, grid {choice['grid']} "
                    "(convergence pass rules)\n\nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"],
                   cwd=REPO, check=True)
    # run from a worktree pinned to this commit, so later edits in the main checkout can't change
    # the code mid-run (and git_hash / git_dirty in the provenance describe exactly what ran)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.strip()
    wt = REPO / ".worktrees" / f"injector_full_{sha[:10]}"
    excl = REPO / ".git" / "info" / "exclude"
    if ".worktrees/" not in excl.read_text():
        excl.write_text(excl.read_text() + "\n.worktrees/\n")
    if not wt.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=REPO, check=True)
    env = {**os.environ, "PYTHONPATH": str(wt / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    with open(LOG / "run.log", "a") as f:
        p = subprocess.Popen(["nice", "-n", "10", PY, "-u", "-m", "twobunch_s2e_rl.datagen.injector_campaign",
                              str(wt / "configs" / "datagen" / LAUNCH_CFG.name)], cwd=wt, env=env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
    (LOG / "run.pid").write_text(str(p.pid))
    print(f"launched pid {p.pid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
