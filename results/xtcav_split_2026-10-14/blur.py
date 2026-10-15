import json, numpy as np, torch
from pmd_beamphysics import ParticleGroup
from xtcav_segment.convert import particlegroup_to_beam
from xtcav_segment.segment import back_drift, build_segment_from_spec, load_spec, set_tdc
spec = load_spec("configs/segment/segment_baseline.json")
e_ref, L = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
eta = float(spec["meta"]["R_entrance_to_exit"][2][5])
seg, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
def trk(P, V):
    set_tdc(seg, V, 0.0); b, _ = particlegroup_to_beam(P, e_ref)
    o = seg.track(back_drift(b, L)); return o.x.numpy(), o.y.numpy()
def core(v, w, f=0.9):   # std of the central 90% (robust to halo)
    lo, hi = np.percentile(v, [50*(1-f), 100-50*(1-f)]); m = (v>=lo)&(v<=hi); return v[m].std()
for i in (160, 91):
    P = ParticleGroup(f"data/xtcav23_valgate_2026-10-14/sample_{i:05d}_XTCAVF.h5")
    w = np.unique(P.weight); isw = P.weight == w[0]
    x0, y0 = trk(P, 0.0); xp, yp = trk(P, 22e6)
    z = P.z; d = (P.energy - e_ref)/e_ref
    S = np.polyfit(z, xp - x0, 1)[0]
    print(f"sample {i}: streak S = {S:.2f} um/um, eta_y = {1e3*eta:.1f} mm")
    for nm, m in (("drive", ~isw), ("witness", isw)):
        rz = (x0[m]).std()/abs(S); rz90 = core(x0[m], None)/abs(S)
        ry = (y0[m] - eta*d[m]).std()/abs(eta); 
        print(f"  {nm:8s} sig_z true {1e6*z[m].std():5.1f} um | z-res from TCAV-off x spot {1e6*rz:5.1f} um (90% core {1e6*rz90:4.1f}) "
              f"| sig_d true {100*d[m].std():.3f}% | y spot (betatron) as delta {100*ry:.3f}% "
              f"| x halo frac >3 sigma_core {np.mean(np.abs(x0[m]-np.median(x0[m]))>3*core(x0[m],None)):.3f}")
print("--- decomposition of the TCAV-off x spot (z-equivalent, um) ---")
for i in (160, 91):
    P = ParticleGroup(f"data/xtcav23_valgate_2026-10-14/sample_{i:05d}_XTCAVF.h5")
    w = np.unique(P.weight); isw = P.weight == w[0]
    x0, y0 = trk(P, 0.0); xp, _ = trk(P, 22e6); S = abs(np.polyfit(P.z, xp - x0, 1)[0])
    d = (P.energy - e_ref)/e_ref; z = P.z
    for nm, m in (("drive", ~isw), ("witness", isw)):
        x = x0[m]
        # what the off-shot row-distortion correction removes: mean x per screen row (y pixel)
        rows = np.floor(y0[m]/30.5e-6).astype(int)
        u, inv = np.unique(rows, return_inverse=True)
        rowmean = np.bincount(inv, x)/np.bincount(inv)
        r_row = x - rowmean[inv]
        r_lin = x - np.polyval(np.polyfit(d[m], x, 1), d[m])
        r_zfit = x - np.polyval(np.polyfit(z[m], x, 3), z[m])
        # betatron-only: remove correlation with both delta and z (poly)
        A = np.stack([d[m]**k for k in range(4)] + [z[m]**k for k in range(1,4)], 1)
        r_both = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        print(f"  {i} {nm:8s} total {1e6*x.std()/S:5.1f} | minus linear x(delta) {1e6*r_lin.std()/S:5.1f} | "
              f"minus per-row mean (deployed row correction) {1e6*r_row.std()/S:5.1f} | minus x(z) cubic {1e6*r_zfit.std()/S:5.1f} | "
              f"minus x(delta,z) {1e6*r_both.std()/S:5.1f}")
