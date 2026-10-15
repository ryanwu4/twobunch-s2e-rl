import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup
from xtcav_segment.convert import particlegroup_to_beam
from xtcav_segment.segment import back_drift, build_segment_from_spec, load_spec, set_tdc
PX = 30.5e-6
spec = load_spec("configs/segment/segment_baseline.json")
e_ref, L = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
eta = float(spec["meta"]["R_entrance_to_exit"][2][5])
seg, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
def trk(P, V):
    set_tdc(seg, V, 0.0); b, _ = particlegroup_to_beam(P, e_ref)
    o = seg.track(back_drift(b, L)); return o.x.numpy().copy(), o.y.numpy().copy()
fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
for r, i in enumerate((160, 91)):
    P = ParticleGroup(f"data/xtcav23_valgate_2026-10-14/sample_{i:05d}_XTCAVF.h5")
    q = np.asarray(P.weight); isw = q == np.unique(q)[0]
    x0, y0 = trk(P, 0.0); xp, yp = trk(P, 22e6)
    S = np.polyfit(P.z, xp - x0, 1)[0]
    ry0, ryp = np.floor(y0 / PX).astype(int), np.floor(yp / PX).astype(int)
    u, inv = np.unique(ry0, return_inverse=True)
    cent = dict(zip(u, np.bincount(inv, q * x0) / np.bincount(inv, q)))
    shift = np.array([cent.get(k, 0.0) for k in ryp]) - np.average(x0, weights=q)
    d = (P.energy - e_ref) / e_ref
    zr = lambda x: 1e6 * (x - np.average(x, weights=q)) / S
    yd = 1e2 * (yp - np.average(yp, weights=q)) / eta + 1e2 * np.average(d, weights=q)
    panels = [("true LPS at XTCAVF", 1e6 * (P.z - np.average(P.z, weights=q)), 1e2 * d),
              ("DTOTR +22 MV, raw  (x/S, y/η)", zr(xp), yd),
              ("DTOTR, off-shot row correction applied", zr(xp - shift), yd)]
    for c, (t, zz, dd) in enumerate(panels):
        a = ax[r, c]
        for m, col in ((~isw, "#1f77b4"), (isw, "#ff7f0e")):
            a.scatter(zz[m][::4], dd[m][::4], s=0.2, c=col, alpha=0.35, rasterized=True)
        a.set(title=f"sample {i}: {t}", xlabel="z [µm] (head +)", ylabel="δ [%]", xlim=(-260, 160))
    # blur budget bars
    a = ax[r, 3]
    lab, vals = [], []
    for nm, m in (("drive", ~isw), ("witness", isw)):
        A = np.stack([d[m] ** k for k in range(4)], 1)
        resid = x0[m] - A @ np.linalg.lstsq(A, x0[m], rcond=None)[0]
        lab += [f"{nm}\nσz true", f"{nm}\nx-spot raw", f"{nm}\nx-spot\nrow-corr"]
        vals += [1e6 * P.z[m].std(), 1e6 * x0[m].std() / abs(S), 1e6 * resid.std() / abs(S)]
    a.bar(range(6), vals, color=["#1f77b4"] * 3 + ["#ff7f0e"] * 3, alpha=0.8)
    a.axhspan(17, 21, color="gray", alpha=0.3, label="real σ_eff (sim2real, 17–21 µm)")
    a.set_xticks(range(6), lab, fontsize=7); a.set_ylabel("µm (z-equivalent)"); a.legend(fontsize=7)
    a.set_title("bunch length vs screen resolution")
fig.tight_layout(); fig.savefig("results/xtcav_split_2026-10-14/screen_blur_budget.png", dpi=120)
