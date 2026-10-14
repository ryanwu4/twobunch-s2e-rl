"""The one Bmad tracking path (FACET2_S2E / Tao): used by datagen and validation.

Runs in the bmad-qpad-dev env. `setLattice` layers its overrides on FACET2-S2E's defaults.yml (not
on the baseline yml), so every call must pass the FULL merged settings dict
`{**baseline_yml, **fixed_overrides, **knobs}` — `merged_settings` builds it.
"""
from __future__ import annotations

import numpy as np

# getBeamSpecs only knows golden Twiss at these elements; any other name must pass None
_TWISS_POINTS = ("PR10571", "BEGBC20", "MFFF", "PENT")


def merged_settings(baseline: dict, fixed_overrides: dict | None, knobs: dict) -> dict:
    return {**baseline, **(fixed_overrides or {}), **knobs}


def make_tao(qs, s2e_root, baseline: dict, *, csr: bool, wakes: bool, n_particles: int, scratch,
             saved_at=()):
    """`saved_at`: extra elements whose beam Tao must keep in RAM so `track` can read them (Tao
    saves every MARKER plus FACET2-S2E's own list; e.g. the BLEN monitors BL11356 / BL14888 are not
    markers)."""
    tao = qs.initializeTao(
        filePath=str(s2e_root), inputBeamFilePathSuffix=baseline["inputBeamFilePathSuffix"],
        csrTF=csr, transverseWakes=wakes, numMacroParticles=n_particles,
        scratchPath=str(scratch), randomizeFileNames=True)
    if saved_at:
        tao.cmd(f"set beam add_saved_at = {', '.join(saved_at)}")
    return tao


def track(qs, tao, s2e_root, settings: dict, points, *, n_in: int, save_beams_at=(),
          out_prefix=None) -> dict:
    """setLattice + trackBeam with the full merged `settings`, then per treaty point: optionally save
    the beam (`<out_prefix>_<ELE>.h5`), and record live counts + getBeamSpecs. Returns
    {"specs": {ele: {...}}} — the same structure as the campaign sample jsons."""
    qs.setLattice(tao, **settings)
    qs.trackBeam(tao, str(s2e_root), **settings)
    record = {"specs": {}}
    for ele in points:
        P = qs.getBeamAtElement(tao, ele)
        # save before specs: degenerate (fully scraped) beams can crash getBeamSpecs
        if ele in save_beams_at and len(P.x) and out_prefix is not None:
            P.write(f"{out_prefix}_{ele}.h5")
        specs = {"n_live_total": len(P.x), "transmission_total": len(P.x) / n_in}
        if len(P.x):
            # whole-beam moments (a BPM reads the charge-weighted centroid; mean energy = what the
            # energy BPMs / feedbacks see); sigma_z_rms_total is NOT the BLEN signal (coherent
            # radiation ~ form factors of both bunches) -- save beams at the BLEN points for that
            w = np.asarray(P.weight, float)
            specs["sigma_z_rms_total"] = float(np.std(P.z))
            specs["mean_energy_eV"] = float(np.average(P.energy, weights=w))
            specs["mean_x_m"] = float(np.average(P.x, weights=w))
            specs["mean_y_m"] = float(np.average(P.y, weights=w))
            specs["charge_C"] = float(w.sum())
        if len(np.unique(P.weight)) == 2:
            PDrive, PWitness = qs.getDriverAndWitness(P)
            specs["n_live_drive"] = len(PDrive.x)
            specs["n_live_witness"] = len(PWitness.x)
            specs["sigma_z_rms_drive"] = float(np.std(PDrive.z))
            specs["sigma_z_rms_witness"] = float(np.std(PWitness.z))
        try:
            specs.update(qs.getBeamSpecs(P, targetTwiss=ele if ele in _TWISS_POINTS else None) or {})
        except Exception as e:  # noqa: BLE001 - record, don't lose the sample
            specs["specs_error"] = f"{type(e).__name__}: {e}"
        record["specs"][ele] = specs
    return record


def split_bunches(P):
    """(drive, witness) ParticleGroups split by per-particle weight: drive = higher weight
    (FACET2_S2E.getDriverAndWitness convention). Witness is None if only one weight survives."""
    w = np.unique(P.weight)
    if len(w) < 2:
        return P, None
    return P[P.weight == w[-1]], P[P.weight == w[0]]


def cloud_metrics(beam_h5) -> dict[str, float]:
    """Truth metrics of a saved two-bunch beam with the SAME kernels the surrogate and reward use
    (`surrogate.properties`), on the full clouds. Needs torch (available in bmad-qpad-dev)."""
    import torch
    from pmd_beamphysics import ParticleGroup
    from ..surrogate import COORD_KEYS
    from ..surrogate.properties import inter_bunch, per_bunch

    drive, witness = split_bunches(ParticleGroup(str(beam_h5)))
    if witness is None:
        return {"witness_lost": 1.0}
    t = lambda pg: torch.tensor(np.stack([getattr(pg, k) for k in COORD_KEYS], 1)[None],
                                dtype=torch.float64)
    d, w = t(drive), t(witness)
    out = {f"drive_{k}": float(v) for k, v in per_bunch(d).items()}
    out.update({f"witness_{k}": float(v) for k, v in per_bunch(w).items()})
    out.update({k: float(v) for k, v in inter_bunch(d, w).items()})
    out["drive_charge_fraction"] = float(drive.charge / (drive.charge + witness.charge))
    out["witness_lost"] = 0.0
    return out

