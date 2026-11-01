"""Upstream (L0AFEND -> XTCAVF) quad-to-beam misalignment in Bmad: does it move the XTCAVF
longitudinal features (spacing, drive sigma_z, energy difference, transmissions) or their L2-phase
Jacobian beyond what the existing hidden context jitter already spans?

Random x/y offsets N(0, sigma) on every quadrupole upstream of BEGBC20 (L1/L2/L3 + BC11/BC14 chicane quads) (uncorrected orbit -> an upper
bound for the post-steering residual), 3 seeds, at 3 valgate operating points, at L2 and L2 + 0.5 deg.
bmad-qpad-dev env:  PYTHONPATH=$PWD/src python results/misalignment_2026-10-14/linac_quad_misalignment.py
"""
import json, multiprocessing, os, sys
from pathlib import Path
import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
CAMP = Path("data/xtcav23_valgate_2026-10-14")
POINTS, SEEDS, SIGMAS, DL2 = (160, 91, 10), (1, 2, 3), (5e-6, 10e-6), 0.5
N = 20000
W = {}

def init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import FACET2_S2E as qs
    from twobunch_s2e_rl.datagen.paths import facet2_root
    from twobunch_s2e_rl.sim.bmad import make_tao
    camp = yaml.safe_load(open("configs/datagen/xtcav23_valgate.yaml"))
    root = facet2_root(); base = qs.loadConfig("/" + camp["baseline_config"], str(root))
    tao = make_tao(qs, root, base, csr=True, wakes=True, n_particles=N, scratch=OUT / "scratch")
    names = tao.lat_list("quadrupole::*", "ele.name"); s = tao.lat_list("quadrupole::*", "ele.s")
    s_x = float(tao.lat_list("BEGBC20", "ele.s")[0])   # linac + BC11/BC14 only; BC20/FF overlap the mover/corrector knobs
    W.update(qs=qs, root=root, base=base, camp=camp, tao=tao,
             quads=sorted({n for n, si in zip(names, s)            # linac FODO quads that are not
                          if si < s_x and n.startswith("Q") and not n.startswith("QM") and "#" not in n}))   # superimposed on structures   # lords, not split slaves

def job(task):
    i, seed, sigma, dl2 = task
    tag = f"p{i}_s{seed}_sig{int(sigma*1e6)}_dl2{dl2:+.1f}"
    res = OUT / f"{tag}.json"
    if res.exists():
        return tag, "skip"
    from twobunch_s2e_rl.sim.bmad import cloud_metrics, merged_settings
    qs, tao = W["qs"], W["tao"]
    knobs = json.load(open(CAMP / f"sample_{i:05d}.json"))["knobs"]
    knobs = dict(knobs, L2PhaseSet=knobs["L2PhaseSet"] + dl2)
    st = merged_settings(W["base"], {**W["camp"]["fixed_overrides"], "trackEnd": "XTCAVF"}, knobs)
    qs.setLattice(tao, **st)
    rng = np.random.default_rng(seed)
    for q in W["quads"]:
        dx, dy = (rng.normal(0, sigma, 2) if seed else (0.0, 0.0))
        tao.cmd(f"set ele {q} x_offset = {dx}"); tao.cmd(f"set ele {q} y_offset = {dy}")
    try:
        qs.trackBeam(tao, str(W["root"]), **st)
        P = qs.getBeamAtElement(tao, "XTCAVF")
        h5 = OUT / "scratch" / f"{tag}.h5"; P.write(str(h5))
        rec = {"ok": True, "n_live": len(P.x), "truth": cloud_metrics(h5),
               "orbit_x_mm": 1e3 * float(np.mean(P.x)), "orbit_y_mm": 1e3 * float(np.mean(P.y))}
        h5.unlink()
    except Exception as e:  # noqa: BLE001
        rec = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    rec.update(point=i, seed=seed, sigma=sigma, dl2=dl2, n_quads=len(W["quads"]))
    res.write_text(json.dumps(rec, indent=1, default=float))
    return tag, "ok" if rec["ok"] else rec["error"][:80]

if __name__ == "__main__":
    (OUT / "scratch").mkdir(exist_ok=True)
    tasks = [(i, 0, 0.0, d) for i in POINTS for d in (0.0, DL2)]
    tasks += [(i, s, sg, d) for i in POINTS for s in SEEDS for sg in SIGMAS for d in (0.0, DL2)]
    with multiprocessing.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 21, initializer=init) as pool:
        for tag, st in pool.imap_unordered(job, tasks):
            print(tag, st, flush=True)
