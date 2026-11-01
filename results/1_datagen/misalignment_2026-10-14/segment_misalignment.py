"""Post-XTCAVF element misalignments (100 um) -> change of the v2 screen features, on Bmad beams."""
import json, numpy as np, torch
from pmd_beamphysics import ParticleGroup
from xtcav_segment.convert import particlegroup_to_beam
from xtcav_segment.segment import back_drift, build_segment_from_spec, load_spec, set_tdc, element_by_name
from xtcav_segment.features import screen_features
spec = load_spec("configs/segment/segment_baseline.json")
e_ref, L = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
eta = float(spec["meta"]["R_entrance_to_exit"][2][5])
seg, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
S = -12.96
KEYS = ("bunch_spacing", "drive_sigma_z", "drive_charge_fraction", "energy_difference")
UNIT = {"bunch_spacing": 1e6, "drive_sigma_z": 1e6, "drive_charge_fraction": 1.0, "energy_difference": 1e-6}
def feats(P, isw, q):
    out = []
    for V in (22e6, -22e6):
        set_tdc(seg, V); b, _ = particlegroup_to_beam(P, e_ref); o = seg.track(back_drift(b, L)); out.append(o)
    xp, xm, y = out[0].x.clone(), out[1].x.clone(), out[0].y.clone()
    d, w = torch.as_tensor(~isw), torch.as_tensor(isw)
    T = lambda a: a[None]
    f = screen_features(T(xp[d]), T(xm[d]), T(y[d]), T(xp[w]), T(xm[w]), T(y[w]),
                        torch.tensor([q[~isw].sum()]), torch.tensor([q[isw].sum()]), streak=S, eta_y=eta,
                        ref_energy_eV=e_ref, split="energy", threshold="valley", row_correct=True, resolution_m=15e-6)
    return {k: float(f[k]) * UNIT[k] for k in KEYS} | {"y_mean_mm": 1e3 * float(y.mean()), "x_off_mm": 1e3 * float(0.5 * (xp + xm).mean())}
beams = {}
for i in (160, 91, 10, 50):
    P = ParticleGroup(f"data/xtcav23_valgate_2026-10-14/sample_{i:05d}_XTCAVF.h5")
    q = np.asarray(P.weight); beams[i] = (P, q == np.unique(q)[0], q)
base = {i: feats(*b) for i, b in beams.items()}
print("aligned:", {i: {k: round(v, 3) for k, v in f.items()} for i, f in base.items()})
for name in ("XTCAVF", "Q2FF", "Q1FF", "Q0FF", "Q0D", "Q1D", "Q2D"):
    e = element_by_name(seg, name)
    for ax, vec in (("x", (1e-4, 0.0)), ("y", (0.0, 1e-4))):
        e.misalignment = torch.tensor(vec, dtype=torch.float64)
        rows = []
        for i, b in beams.items():
            f = feats(*b); rows.append({k: f[k] - base[i][k] for k in f})
        e.misalignment = torch.zeros(2, dtype=torch.float64)
        txt = "  ".join(f"{k[:14]} " + "/".join(f"{r[k]:+.3g}" for r in rows) for k in rows[0])
        print(f"{name:6s} {ax} 100um: {txt}")
