"""Stage-A re-validation of the XTCAV segment on cheetah-accelerator 0.8.4 (plan phase 2 gate).

Three parts, all comparing the ``xtcav_segment`` port (0.8.4) against Bmad ground truth, and the
first two also against the ORIGINAL xtcav-gpsr-tuning code on the 0.8.2-dev Cheetah clone (same
inputs, run in a subprocess because two cheetahs cannot share one process):

  stageA     the five Stage-A tests of xtcav-gpsr-tuning/scripts/validate_segment.py, per optics
             (baseline, configurator), on xtcav-gpsr-tuning/data/bmad_reference_<optics>/.
  valtracks  the 24 held-out xtcav23 tracks (data/xtcav23_full_valtracks): XTCAVF exit (V=0) ->
             DTOTR at V = 0, +V, -V; DTOTR centroids, sizes, streak, energy axis (dy/d delta).
  pent       Cheetah XTCAVF -> PENT (TCAV off, Q2/Q1/Q0FF from each sample's knobs) vs Bmad PENT
             beams, per-bunch, with twobunch_s2e_rl.surrogate.properties on both sides.
             Sources: data/xtcav23_valgate_2026-10-14 (beams at XTCAVF AND PENT; whatever samples
             exist), plus a secondary getBeamSpecs-style check on data/xtcav23_full (XTCAVF beams +
             the campaign's PENT getBeamSpecs in each sample json; no PENT beams were saved).

Outputs: results/segment_validation_084/*.json (assumptions in NOTES.md there).

Run (slac-rl env, cheetah-accelerator==0.8.4):
  cd twobunch-s2e-rl
  PYTHONPATH=src nice -n 10 /home/rwu4/miniconda3/envs/slac-rl/bin/python \\
      scripts/validate_segment_084.py --part all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
GPSR_REPO = WORKSPACE / "xtcav-gpsr-tuning"
GPSR_COMMIT = "c1f68ce"  # Stage-A code + results (2026-07-30)
CHEETAH_CLONE = WORKSPACE / "prev_work" / "cheetah_analysis" / "cheetah"
BRIDGE_DIR = WORKSPACE / "prev_work" / "cheetah_analysis" / "bmad_cheetah_bridge"
OUT_DIR = REPO / "results" / "segment_validation_084"
VALTRACKS = REPO / "data" / "xtcav23_full_valtracks"
VALGATE = REPO / "data" / "xtcav23_valgate_2026-10-14"
XTCAV23_FULL = REPO / "data" / "xtcav23_full"
C_LIGHT = 299792458.0
GOLDEN_SEG_KG = {"Q2FF": 126.35, "Q1FF": -235.218, "Q0FF": 126.353}  # defaults.yml


# --------------------------------------------------------------------------- #
# Backends: the port on 0.8.4, or the original package on the clone
# --------------------------------------------------------------------------- #
def load_backend(name: str) -> SimpleNamespace:
    if name == "084":
        import cheetah
        from xtcav_segment import convert, segment
    elif name == "clone":
        # Original modules from git (independent of the xtcav-gpsr-tuning working tree), the
        # clone and the bridge first on sys.path -- exactly what paths.ensure_import_roots did.
        tmp = Path(tempfile.mkdtemp(prefix="xtcav_clone_"))
        pkg = tmp / "xtcav_gpsr_tuning"
        pkg.mkdir()
        for mod in ("__init__", "paths", "segment", "convert"):
            src = subprocess.run(
                ["git", "-C", str(GPSR_REPO), "show", f"{GPSR_COMMIT}:src/xtcav_gpsr_tuning/{mod}.py"],
                check=True, capture_output=True, text=True).stdout
            (pkg / f"{mod}.py").write_text(src)
        for p in (tmp, BRIDGE_DIR, CHEETAH_CLONE):
            sys.path.insert(0, str(p))
        import cheetah
        from xtcav_gpsr_tuning import convert, segment
        assert Path(cheetah.__file__).resolve().is_relative_to(CHEETAH_CLONE.resolve()), cheetah.__file__
    else:
        raise ValueError(name)
    return SimpleNamespace(name=name, cheetah=cheetah, seg=segment, conv=convert,
                           version=getattr(cheetah, "__version__", "?"),
                           cheetah_file=cheetah.__file__)


def load_pg(path):
    from pmd_beamphysics import ParticleGroup

    return ParticleGroup(str(path))


def wmean(beam, i):
    w = beam.particle_charges / beam.particle_charges.sum()
    return float((w * beam.particles[..., i]).sum())


def wstd(beam, i):
    w = beam.particle_charges / beam.particle_charges.sum()
    v = beam.particles[..., i]
    m = (w * v).sum()
    return float((w * (v - m) ** 2).sum().sqrt())


def sigma_x_weighted(beam):
    w = beam.particle_charges / beam.particle_charges.sum()
    m = (w * beam.x).sum()
    return ((w * (beam.x - m) ** 2).sum()).sqrt()


def summarize_compare(cmp: dict) -> dict:
    out = {}
    for coord, d in cmp.items():
        out[coord] = {"rms_over_std": d["rms"] / max(abs(d["std_b"]), 1e-30), "rms": d["rms"],
                      "max": d["max"], "mean_offset": d["mean_offset"],
                      "std_cheetah": d["std_a"], "std_bmad": d["std_b"]}
    return out


# --------------------------------------------------------------------------- #
# Part 1: Stage A (faithful to xtcav-gpsr-tuning/scripts/validate_segment.py)
# --------------------------------------------------------------------------- #
def stage_a(B, optics: str, dump: dict) -> dict:
    import torch

    spec_path = GPSR_REPO / "configs" / f"segment_{optics}.json"
    if B.name == "084":  # the port reads its own copy; assert it is the same spec
        own = B.seg.spec_path(optics)
        assert own.read_bytes() == spec_path.read_bytes(), f"{own} differs from {spec_path}"
        spec_path = own
    ref_dir = GPSR_REPO / "data" / f"bmad_reference_{optics}"
    spec = B.seg.load_spec(spec_path)
    meta = spec["meta"]
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    E_ref, L_tdc = meta["ref_energy_eV"], meta["tdc_length_m"]
    V, phi_test = manifest["voltage_V"], manifest["phi0_test_deg"]
    dtype = torch.float64
    res = {"optics": optics, "spec": str(spec_path), "ref_dir": str(ref_dir), "tests": {}}

    seg_exit, rep_exit = B.seg.build_segment_from_spec(spec, start_after="XTCAVF", dtype=dtype)
    seg_full, rep_full = B.seg.build_segment_from_spec(spec, dtype=dtype)
    res["segment"] = {"exit_plane_elements": rep_exit.n_in, "full_elements": rep_full.n_in,
                      "cheetah_length_exit_m": rep_exit.cheetah_length,
                      "approximated": [list(x) for x in rep_full.approximated]}
    p2b = lambda P: B.conv.particlegroup_to_beam(P, E_ref, None, dtype=dtype)[0]  # noqa: E731

    # 1: pure optics exit -> exit, V=0
    Px_off = load_pg(ref_dir / "beam_XTCAVF_off.h5")
    beam_off = p2b(Px_off)
    out_off = seg_exit.track(beam_off)
    ref_off = p2b(load_pg(ref_dir / "beam_DTOTR_off.h5"))
    res["tests"]["1_pure_optics_exit_to_exit"] = summarize_compare(
        B.conv.compare_beams(out_off, ref_off, center=("tau",)))
    dump[f"{optics}/off"] = out_off.particles[..., :6].detach().numpy()

    # 2: TDC bracket
    entrance = B.seg.back_drift(beam_off, L_tdc)
    B.seg.set_tdc(seg_full, V, 0.0)
    out_plus = seg_full.track(entrance)
    Pd_plus = load_pg(ref_dir / "beam_DTOTR_plus.h5")
    res["tests"]["2_tdc_bracket_plus"] = summarize_compare(
        B.conv.compare_beams(out_plus, p2b(Pd_plus), center=("tau",)))
    dump[f"{optics}/plus"] = out_plus.particles[..., :6].detach().numpy()
    B.seg.set_tdc(seg_full, 0.0, 0.0)
    rt = B.conv.compare_beams(seg_full.track(entrance), out_off, center=())
    res["tests"]["2b_v0_roundtrip_identity"] = {c: {"max": d["max"]} for c, d in rt.items()}

    # 3: PHI0 = +/- phi_test
    for case, sign in (("phi0p", +1.0), ("phi0m", -1.0)):
        f = ref_dir / f"beam_DTOTR_{case}.h5"
        if not f.exists():
            continue
        B.seg.set_tdc(seg_full, V, sign * phi_test)
        out_phi = seg_full.track(entrance)
        ref_phi = p2b(load_pg(f))
        res["tests"][f"3_phi0_{case}"] = {
            **summarize_compare(B.conv.compare_beams(out_phi, ref_phi, center=("tau",))),
            "x_centroid_cheetah_mm": wmean(out_phi, 0) * 1e3,
            "x_centroid_bmad_mm": wmean(ref_phi, 0) * 1e3,
        }
        dump[f"{optics}/{case}"] = out_phi.particles[..., :6].detach().numpy()

    # 4: R-matrix, empirical dispersion, streak S
    B.seg.set_tdc(seg_full, 0.0, 0.0)
    R_exit = B.seg.numerical_rmatrix(seg_exit, E_ref)
    R_entr = B.seg.numerical_rmatrix(seg_full, E_ref)
    dump[f"{optics}/R_exit"], dump[f"{optics}/R_entrance"] = R_exit, R_entr

    def r_cmp(Rc, Rb, elems=((0, 1), (0, 5), (2, 5), (0, 0), (1, 1), (2, 3))):
        return {f"R{i+1}{j+1}": {"cheetah": float(Rc[i, j]), "bmad": float(Rb[i, j]),
                                 "rel_err": float(abs(Rc[i, j] - Rb[i, j]) / max(abs(Rb[i, j]), 1e-12))}
                for i, j in elems}

    res["tests"]["4_rmatrix_exit"] = r_cmp(R_exit, np.array(meta["R_exit_to_exit"]))
    res["tests"]["4_rmatrix_entrance"] = r_cmp(R_entr, np.array(meta["R_entrance_to_exit"]))
    dc, db = disp_slopes(out_off), disp_slopes(ref_off)
    res["tests"]["4_empirical_dispersion_off"] = {
        "cheetah": dc, "bmad": db,
        "rel_err_dy": abs(dc["dy_ddelta"] - db["dy_ddelta"]) / max(abs(db["dy_ddelta"]), 1e-12)}
    B.seg.set_tdc(seg_full, V, 0.0)
    out_S = seg_full.track(entrance)
    z1 = -C_LIGHT * (np.asarray(Px_off.t) - np.average(Px_off.t, weights=Px_off.weight))
    S_c = float(np.polyfit(z1, out_S.x.detach().numpy(), 1)[0])
    Px_plus = load_pg(ref_dir / "beam_XTCAVF_plus.h5")
    z1p = -C_LIGHT * (np.asarray(Px_plus.t) - np.average(Px_plus.t, weights=Px_plus.weight))
    S_b = float(np.polyfit(z1p, np.asarray(Pd_plus.x), 1)[0])
    k_rf = 2 * np.pi * 11.424e9 / C_LIGHT
    res["tests"]["4_streak_S"] = {
        "cheetah_fit": S_c, "bmad_fit": S_b,
        "thin_kick_analytic": float(np.array(meta["R_exit_to_exit"])[0, 1] * k_rf * V / E_ref),
        "rel_err_cheetah_vs_bmad": abs(S_c - S_b) / abs(S_b)}

    # 5: gradients + batching
    tdc = B.seg.element_by_name(seg_full, "XTCAVF")
    q1ff = B.seg.element_by_name(seg_full, "Q1FF")
    B.seg.set_tdc(seg_full, V, 0.0)
    tdc.voltage.requires_grad_(True)
    q1ff.k1.requires_grad_(True)
    sigma_x_weighted(seg_full.track(entrance)).backward()
    g_V, g_k1 = float(tdc.voltage.grad), float(q1ff.k1.grad)

    def fd(param, h):
        with torch.no_grad():
            orig = param.detach().clone()
            param.copy_(orig + h)
            f_p = float(sigma_x_weighted(seg_full.track(entrance)))
            param.copy_(orig - h)
            f_m = float(sigma_x_weighted(seg_full.track(entrance)))
            param.copy_(orig)
        return (f_p - f_m) / (2 * h)

    fd_V, fd_k1 = fd(tdc.voltage, 1e3), fd(q1ff.k1, 1e-6)
    res["tests"]["5_gradients"] = {
        "dsigmax_dV": {"autograd": g_V, "fd": fd_V, "rel_err": abs(g_V - fd_V) / max(abs(fd_V), 1e-30)},
        "dsigmax_dk1_Q1FF": {"autograd": g_k1, "fd": fd_k1,
                             "rel_err": abs(g_k1 - fd_k1) / max(abs(fd_k1), 1e-30)}}
    tdc.voltage.requires_grad_(False)
    q1ff.k1.requires_grad_(False)
    with torch.no_grad():
        tdc.voltage = torch.tensor([0.0, V], dtype=dtype)
        out_b = seg_full.track(entrance)
        w = entrance.particle_charges / entrance.particle_charges.sum()
        xb = out_b.x[0]
        sig0 = float(((w * (xb - (w * xb).sum()) ** 2).sum()).sqrt())
        sig_off = float(sigma_x_weighted(out_off))
        tdc.voltage = torch.tensor(V, dtype=dtype)
    res["tests"]["5_batched"] = {"particles_shape": list(out_b.particles.shape),
                                 "batch0_sigma_x_um": sig0 * 1e6,
                                 "unbatched_off_sigma_x_um": sig_off * 1e6,
                                 "rel_err": abs(sig0 - sig_off) / sig_off}
    return res


def disp_slopes(beam):
    p = beam.particles.detach().numpy()
    return {"dx_ddelta": float(np.polyfit(p[..., 5], p[..., 0], 1)[0]),
            "dy_ddelta": float(np.polyfit(p[..., 5], p[..., 2], 1)[0])}


# --------------------------------------------------------------------------- #
# Part 2: valtracks (24 xtcav23 samples, 100k, +/-22 MV)
# --------------------------------------------------------------------------- #
def dtotr_observables(beam, z1=None) -> dict:
    """DTOTR screen-level observables: centroids, sizes, energy axis (dy/d delta), streak."""
    o = {"cx": wmean(beam, 0), "cy": wmean(beam, 2), "sx": wstd(beam, 0), "sy": wstd(beam, 2),
         **disp_slopes(beam)}
    if z1 is not None:
        o["streak_S"] = float(np.polyfit(z1, beam.x.detach().numpy(), 1)[0])
    return o


def valtracks(B, dump: dict, limit: int | None = None) -> dict:
    import torch

    spec_path = B.seg.spec_path("baseline") if B.name == "084" else GPSR_REPO / "configs" / "segment_baseline.json"
    spec = B.seg.load_spec(spec_path)
    E_ref, L_tdc = spec["meta"]["ref_energy_eV"], spec["meta"]["tdc_length_m"]
    man = json.loads((VALTRACKS / "valtracks_manifest.json").read_text())
    seg_exit, _ = B.seg.build_segment_from_spec(spec, start_after="XTCAVF")
    seg_full, _ = B.seg.build_segment_from_spec(spec)
    p2b = lambda P: B.conv.particlegroup_to_beam(P, E_ref, None, dtype=torch.float64)[0]  # noqa: E731
    out = {}
    for name in sorted(man)[:limit]:
        V = man[name]["voltage_V"]
        Px = load_pg(VALTRACKS / f"{name}_XTCAVF_off.h5")
        beam = p2b(Px)
        z1 = -C_LIGHT * (np.asarray(Px.t) - np.average(Px.t, weights=Px.weight))
        entrance = B.seg.back_drift(beam, L_tdc)
        rec = {}
        for case, sign in (("off", 0.0), ("plus", 1.0), ("minus", -1.0)):
            Pd = load_pg(VALTRACKS / f"{name}_DTOTR_{case}.h5")
            if case == "off":
                tracked = seg_exit.track(beam)
            else:
                B.seg.set_tdc(seg_full, sign * V, 0.0)
                tracked = seg_full.track(entrance)
            ref = p2b(Pd)
            aligned = len(Pd.x) == len(Px.x)
            zc = z1 if case != "off" else None
            rec[case] = {"cheetah": dtotr_observables(tracked, zc),
                         "bmad": dtotr_observables(ref, zc if aligned else None),
                         "index_aligned": aligned}
            if aligned:
                cmp = B.conv.compare_beams(tracked, ref, center=("tau",))
                rec[case]["per_particle_rms_over_std"] = {
                    c: cmp[c]["rms"] / max(cmp[c]["std_b"], 1e-30) for c in ("x", "y")}
            dump[f"{name}/{case}"] = tracked.particles[..., :6].detach().numpy()
        out[name] = rec
    return out


def valtracks_summary(vt: dict) -> dict:
    """max / mean over samples of Cheetah-vs-Bmad DTOTR differences, per case."""
    s = {}
    for case in ("off", "plus", "minus"):
        rows = [vt[n][case] for n in vt]
        d = {
            "centroid_x_um": [abs(r["cheetah"]["cx"] - r["bmad"]["cx"]) * 1e6 for r in rows],
            "centroid_y_um": [abs(r["cheetah"]["cy"] - r["bmad"]["cy"]) * 1e6 for r in rows],
            "sigma_x_rel": [abs(r["cheetah"]["sx"] / r["bmad"]["sx"] - 1) for r in rows],
            "sigma_y_rel": [abs(r["cheetah"]["sy"] / r["bmad"]["sy"] - 1) for r in rows],
            "energy_axis_dy_ddelta_rel": [abs(r["cheetah"]["dy_ddelta"] / r["bmad"]["dy_ddelta"] - 1)
                                          for r in rows],
        }
        if case != "off":
            d["streak_S_rel"] = [abs(r["cheetah"]["streak_S"] / r["bmad"]["streak_S"] - 1)
                                 for r in rows if "streak_S" in r["bmad"]]
        if all("per_particle_rms_over_std" in r for r in rows):
            d["per_particle_x_rms_over_std"] = [r["per_particle_rms_over_std"]["x"] for r in rows]
            d["per_particle_y_rms_over_std"] = [r["per_particle_rms_over_std"]["y"] for r in rows]
        s[case] = {k: {"max": float(np.max(v)), "mean": float(np.mean(v)), "n": len(v)}
                   for k, v in d.items() if v}
    return s


# --------------------------------------------------------------------------- #
# Part 3: PENT (0.8.4 only)
# --------------------------------------------------------------------------- #
def split_bunches(cloud: np.ndarray, weight: np.ndarray):
    """(drive, witness) by the +-0.1% weight tag (drive = larger weight), as getDriverAndWitness."""
    w = np.sort(np.unique(weight))
    assert len(w) == 2, f"expected 2 unique weights, got {len(w)}"
    return cloud[weight == w[1]], cloud[weight == w[0]]


def pg_cloud(P) -> np.ndarray:
    return np.stack([P.x, P.y, P.z, P.px, P.py, P.pz], axis=1)


def properties_of(drive: np.ndarray, witness: np.ndarray) -> dict:
    import torch
    from twobunch_s2e_rl.surrogate import properties as props

    t = lambda a: torch.as_tensor(a, dtype=torch.float64).unsqueeze(0)  # noqa: E731
    out = {}
    for tag, c in (("drive", drive), ("witness", witness)):
        for k, v in props.per_bunch(t(c)).items():
            if v.dim() == 1:
                out[f"{tag}_{k}"] = float(v[0])
    out.update({k: float(v[0]) for k, v in props.inter_bunch(t(drive), t(witness)).items()})
    return out


PENT_KEYS = ["bunch_spacing", "drive_sigma_z", "witness_sigma_z",
             "drive_norm_emit_x", "drive_norm_emit_y", "witness_norm_emit_x", "witness_norm_emit_y",
             "drive_bmag_x", "drive_bmag_y", "witness_bmag_x", "witness_bmag_y",
             "drive_centroid_x", "drive_centroid_y", "witness_centroid_x", "witness_centroid_y",
             "transverse_offset", "angular_misalignment", "energy_difference"]


def cheetah_pent(B, spec, Px, knobs: dict):
    """Cheetah XTCAVF exit (V=0 beam) -> PENT with the sample's segment quads. Returns cloud."""
    import torch

    E_ref = spec["meta"]["ref_energy_eV"]
    seg, _ = B.seg.build_segment_from_spec(spec, start_after="XTCAVF", end="PENT")
    quads = {q: float(knobs.get(f"{q}kG", GOLDEN_SEG_KG[q])) for q in B.seg.SEGMENT_QUADS}
    B.seg.set_quads_kG(seg, quads, E_ref)
    beam, _ = B.conv.particlegroup_to_beam(Px, E_ref, None, dtype=torch.float64)
    with torch.no_grad():
        return B.conv.beam_to_cloud(seg.track(beam)).numpy(), quads


def pent_valgate(B) -> dict:
    spec = B.seg.load_spec(B.seg.spec_path("baseline"))
    out = {}
    if not VALGATE.exists():
        return out
    for js in sorted(VALGATE.glob("sample_*.json")):
        stem = js.stem
        fx, fp = VALGATE / f"{stem}_XTCAVF.h5", VALGATE / f"{stem}_PENT.h5"
        if not (fx.exists() and fp.exists()):
            continue
        try:
            d = json.loads(js.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("success", True):
            continue
        Px, Pp = load_pg(fx), load_pg(fp)
        cloud, quads = cheetah_pent(B, spec, Px, d["knobs"])
        pc = properties_of(*split_bunches(cloud, np.asarray(Px.weight)))
        pb = properties_of(*split_bunches(pg_cloud(Pp), np.asarray(Pp.weight)))
        out[stem] = {"block": d.get("block"), "quads_kG": quads,
                     "n_xtcavf": len(Px.x), "n_pent_bmad": len(Pp.x),
                     "cheetah": {k: pc[k] for k in PENT_KEYS}, "bmad": {k: pb[k] for k in PENT_KEYS}}
    return out


def beamspecs_subset(P_drive, P_witness, beta0=0.5, alpha0=0.0) -> dict:
    """getBeamSpecs (FACET2_S2E.beam.analysis) subset, same estimators: medians, pmd twiss
    fraction=0.9 norm_emit + BMAG, zCentroid = median(t*3e8), bunchSpacing, transverse offset."""
    def bmag(b, a):
        g0, g = (1 + alpha0**2) / beta0, (1 + a**2) / b
        return (beta0 * g - 2 * alpha0 * a + g0 * b) / 2

    s = {}
    for tag, P in (("PDrive", P_drive), ("PWitness", P_witness)):
        tw = P.twiss(plane="x", fraction=0.9) | P.twiss(plane="y", fraction=0.9)
        s[f"{tag}_median_x"], s[f"{tag}_median_y"] = float(np.median(P.x)), float(np.median(P.y))
        s[f"{tag}_norm_emit_x"], s[f"{tag}_norm_emit_y"] = tw["norm_emit_x"], tw["norm_emit_y"]
        s[f"{tag}_BMAG_x"] = bmag(tw["beta_x"], tw["alpha_x"])
        s[f"{tag}_BMAG_y"] = bmag(tw["beta_y"], tw["alpha_y"])
        s[f"{tag}_zCentroid"] = float(np.median(P.t * 3e8))
    s["bunchSpacing"] = s["PWitness_zCentroid"] - s["PDrive_zCentroid"]
    s["transverseCentroidOffset"] = float(np.hypot(s["PDrive_median_x"] - s["PWitness_median_x"],
                                                   s["PDrive_median_y"] - s["PWitness_median_y"]))
    return s


BEAMSPEC_KEYS = ["bunchSpacing", "transverseCentroidOffset",
                 "PDrive_norm_emit_x", "PDrive_norm_emit_y", "PWitness_norm_emit_x",
                 "PWitness_norm_emit_y", "PDrive_BMAG_x", "PDrive_BMAG_y", "PWitness_BMAG_x",
                 "PWitness_BMAG_y", "PDrive_median_x", "PDrive_median_y", "PWitness_median_x",
                 "PWitness_median_y"]


def cloud_to_pg(cloud: np.ndarray, weight: np.ndarray):
    """Cheetah PENT cloud -> ParticleGroup; t from z (t = -z/c, arbitrary common offset)."""
    from pmd_beamphysics import ParticleGroup

    n = len(weight)
    return ParticleGroup(data={"x": cloud[:, 0], "y": cloud[:, 1], "z": cloud[:, 2],
                               "px": cloud[:, 3], "py": cloud[:, 4], "pz": cloud[:, 5],
                               "t": -cloud[:, 2] / C_LIGHT, "status": np.ones(n, dtype=int),
                               "weight": weight, "species": "electron"})


def pent_secondary_xtcav23_full(B, n: int) -> dict:
    """Cheetah XTCAVF->PENT on xtcav23_full saved XTCAVF beams vs the campaign's PENT
    getBeamSpecs (no PENT beams saved). Same-estimator subset of getBeamSpecs."""
    spec = B.seg.load_spec(B.seg.spec_path("baseline"))
    jsons = sorted(XTCAV23_FULL.glob("sample_*.json"))
    picks = [jsons[int(i)] for i in np.linspace(0, len(jsons) - 1, n)]
    out = {}
    for js in picks:
        d = json.loads(js.read_text())
        fx = XTCAV23_FULL / f"{js.stem}_XTCAVF.h5"
        pent = d.get("specs", {}).get("PENT", {})
        if not (d.get("success") and fx.exists() and "bunchSpacing" in pent):
            continue
        Px = load_pg(fx)
        if pent.get("n_live_total") != len(Px.x):
            continue  # losses between XTCAVF and PENT in Bmad: not comparable bunch-for-bunch
        cloud, _ = cheetah_pent(B, spec, Px, d["knobs"])
        w = np.asarray(Px.weight)
        P = cloud_to_pg(cloud, w)
        wu = np.sort(np.unique(w))
        s = beamspecs_subset(P[P.weight == wu[1]], P[P.weight == wu[0]])
        out[js.stem] = {"cheetah": {k: s[k] for k in BEAMSPEC_KEYS},
                        "bmad_getBeamSpecs": {k: pent[k] for k in BEAMSPEC_KEYS}}
    return out


def diff_summary(rows: dict, keys: list, a: str, b: str) -> dict:
    s = {}
    for k in keys:
        ca = np.array([r[a][k] for r in rows.values()])
        cb = np.array([r[b][k] for r in rows.values()])
        dd = ca - cb
        rel = np.abs(dd) / np.maximum(np.abs(cb), 1e-30)
        s[k] = {"abs_max": float(np.abs(dd).max()), "abs_mean": float(np.abs(dd).mean()),
                "signed_mean": float(dd.mean()), "rel_max": float(rel.max()),
                "rel_median": float(np.median(rel)), "bmad_abs_median": float(np.median(np.abs(cb))),
                "n": int(len(dd))}
    return s


# --------------------------------------------------------------------------- #
# 0.8.4 vs clone
# --------------------------------------------------------------------------- #
def run_clone_subprocess(parts: list, dump_dir: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": ""}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--backend", "clone",
           "--part", ",".join(parts), "--dump-dir", str(dump_dir)]
    print("[clone] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    return json.loads((dump_dir / "clone_results.json").read_text())


def array_diffs(a: dict, b: dict) -> dict:
    """max |a-b| per coordinate over every dumped array (x, px, y, py, tau, delta)."""
    names = ("x", "px", "y", "py", "tau", "delta")
    worst = {n: 0.0 for n in names}
    rmat = 0.0
    for k in a:
        if k.endswith("R_exit") or k.endswith("R_entrance"):
            rmat = max(rmat, float(np.abs(a[k] - b[k]).max()))
            continue
        d = np.abs(a[k] - b[k]).max(axis=0)
        for i, n in enumerate(names):
            worst[n] = max(worst[n], float(d[i]))
    return {"max_abs_per_coord": worst, "max_abs_rmatrix": rmat, "n_arrays": len(a)}


def nested_max_reldiff(a, b, path="") -> list:
    """Walk two results dicts; list (path, a, b, rel diff) for every float leaf."""
    rows = []
    if isinstance(a, dict):
        for k in a:
            if k in b:
                rows += nested_max_reldiff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        rows.append((path, float(a), float(b), abs(a - b) / max(abs(b), 1e-300)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", help="comma list of stageA,valtracks,pent or 'all'")
    ap.add_argument("--backend", choices=["084", "clone"], default="084")
    ap.add_argument("--dump-dir", default=None, help="(internal) where the clone run saves its outputs")
    ap.add_argument("--no-clone", action="store_true", help="skip the 0.8.4-vs-clone comparison")
    ap.add_argument("--n-secondary", type=int, default=200, help="xtcav23_full samples for the "
                    "secondary PENT getBeamSpecs check")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    parts = ["stageA", "valtracks", "pent"] if args.part == "all" else args.part.split(",")

    import torch

    torch.set_num_threads(args.threads)
    B = load_backend(args.backend)
    print(f"[{B.name}] cheetah {B.version} from {B.cheetah_file}", flush=True)

    results, dump = {"backend": B.name, "cheetah_version": B.version, "cheetah_file": B.cheetah_file}, {}
    t0 = time.time()
    if "stageA" in parts:
        results["stageA"] = {o: stage_a(B, o, dump) for o in ("baseline", "configurator")}
        print(f"[{B.name}] stageA done {time.time() - t0:.0f}s", flush=True)
    if "valtracks" in parts:
        results["valtracks"] = valtracks(B, dump)
        print(f"[{B.name}] valtracks done {time.time() - t0:.0f}s", flush=True)

    if args.backend == "clone":  # internal mode: save and exit
        dd = Path(args.dump_dir)
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / "clone_arrays.npz", **dump)
        (dd / "clone_results.json").write_text(json.dumps(results, indent=1))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "cheetah_084_file": B.cheetah_file,
            "gpsr_commit": GPSR_COMMIT, "clone": str(CHEETAH_CLONE)}

    clone_parts = [p for p in parts if p in ("stageA", "valtracks")]
    clone = None
    if clone_parts and not args.no_clone:
        dump_dir = Path(tempfile.mkdtemp(prefix="seg084_clone_dump_"))
        clone = run_clone_subprocess(clone_parts, dump_dir)
        carr = dict(np.load(dump_dir / "clone_arrays.npz"))
        cmp = {"particle_arrays": array_diffs(dump, carr)}
        for p in clone_parts:
            rows = nested_max_reldiff(results[p], clone[p])
            worst = sorted(rows, key=lambda r: -r[3])[:10]
            cmp[f"{p}_results_reldiff_top10"] = [
                {"path": r[0], "v084": r[1], "clone": r[2], "rel": r[3]} for r in worst]
            cmp[f"{p}_n_leaves"] = len(rows)
        (OUT_DIR / "clone_vs_084.json").write_text(json.dumps({**meta, **cmp}, indent=1))
        print("[084 vs clone]", json.dumps(cmp["particle_arrays"], indent=1), flush=True)
        for f in dump_dir.iterdir():
            f.unlink()
        dump_dir.rmdir()

    if "stageA" in parts:
        for o, r in results["stageA"].items():
            archived = GPSR_REPO / "results" / "stageA" / f"validation_{o}.json"
            r["vs_archived_2026-07-30"] = None
            if archived.exists():
                rows = nested_max_reldiff(r["tests"], json.loads(archived.read_text())["tests"])
                worst = max(rows, key=lambda x: x[3])
                r["vs_archived_2026-07-30"] = {"n_leaves": len(rows), "max_rel": worst[3],
                                               "max_rel_path": worst[0]}
            (OUT_DIR / f"stageA_{o}.json").write_text(json.dumps({**meta, **r}, indent=1))
    if "valtracks" in parts:
        vt = results["valtracks"]
        out = {**meta, "summary_084_vs_bmad": valtracks_summary(vt)}
        if clone is not None:
            out["summary_clone_vs_bmad"] = valtracks_summary(clone["valtracks"])
        out["per_sample"] = vt
        (OUT_DIR / "valtracks.json").write_text(json.dumps(out, indent=1))
    if "pent" in parts:
        vg = pent_valgate(B)
        out = {**meta, "source": str(VALGATE), "n_samples": len(vg)}
        if vg:
            out["summary_cheetah_vs_bmad"] = diff_summary(vg, PENT_KEYS, "cheetah", "bmad")
            for blk in sorted({r["block"] for r in vg.values()}):
                sub = {k: r for k, r in vg.items() if r["block"] == blk}
                out[f"summary_{blk}"] = diff_summary(sub, PENT_KEYS, "cheetah", "bmad")
        out["per_sample"] = vg
        (OUT_DIR / "pent_valgate.json").write_text(json.dumps(out, indent=1))
        print(f"[pent] valgate samples with XTCAVF+PENT beams: {len(vg)}", flush=True)
        sec = pent_secondary_xtcav23_full(B, args.n_secondary)
        out2 = {**meta, "source": str(XTCAV23_FULL), "n_samples": len(sec),
                "note": "Bmad side = campaign getBeamSpecs at PENT (sample json); Cheetah side = the "
                        "same estimators (medians, pmd twiss fraction=0.9) on the Cheetah PENT beam "
                        "tracked from the saved XTCAVF beam. Samples with Bmad losses skipped."}
        if sec:
            out2["summary"] = diff_summary(sec, BEAMSPEC_KEYS, "cheetah", "bmad_getBeamSpecs")
        out2["per_sample"] = sec
        (OUT_DIR / "pent_secondary_xtcav23_full.json").write_text(json.dumps(out2, indent=1))
        print(f"[pent] secondary xtcav23_full samples: {len(sec)}", flush=True)
    print(f"done {time.time() - t0:.0f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
