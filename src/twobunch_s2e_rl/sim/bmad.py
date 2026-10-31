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
             saved_at=(), n_bin: int = 32):
    """`saved_at`: extra elements whose beam Tao must keep in RAM so `track` can read them (Tao
    saves every MARKER plus FACET2-S2E's own list; e.g. the BLEN monitors BL11356 / BL14888 are not
    markers). `n_bin`: longitudinal bins of the 1D CSR calculation (FACET2-S2E's default 32)."""
    tao = qs.initializeTao(
        filePath=str(s2e_root), inputBeamFilePathSuffix=baseline["inputBeamFilePathSuffix"],
        csrTF=csr, transverseWakes=wakes, numMacroParticles=n_particles,
        scratchPath=str(scratch), randomizeFileNames=True, n_bin=n_bin)
    if saved_at:
        tao.cmd(f"set beam add_saved_at = {', '.join(saved_at)}")
    return tao


def l2_machine_solve(v_fixed, v_scan, v_pair, phi_m_deg, spec) -> dict:
    """L2 as the machine's L2_PHASE multiknob scans it (E300/E331 DAQs): the LI12-14 sub-boosters
    move together by `delta_deg`, LI11 (11-4..11-8) stays put, and the BC14 energy feedback opens or
    closes the 14-4 / 14-6 pair (psi -/+ theta about the LI14 sub-booster phase psi) to hold the
    energy. v_*: on-crest voltages of the fixed / scanned / pair groups (pair = (14-4, 14-6)) at the
    model's uniform setting phi_m_deg, whose vector sum Z is the operating point at the scan centre.

    spec: theta0_deg (pair half-angle at the centre), phi11_deg (LI11 phase; None = in phase with
    LI12-14 at the centre), delta_deg (this step's multiknob offset). A common gradient scale and
    the centre sub-booster phase are solved so the centre reproduces Z; at delta the pair angle is
    re-solved so Re (the energy gain) stays at Re Z."""
    p4, p6 = v_pair
    Z = (v_fixed + v_scan + p4 + p6) * np.exp(1j * np.deg2rad(phi_m_deg))
    th0 = np.deg2rad(spec["theta0_deg"])
    Q = lambda th: v_scan + p4 * np.exp(-1j * th) + p6 * np.exp(1j * th)
    if spec.get("phi11_deg") is None:
        W = v_fixed + Q(th0)
        scale, phis = abs(Z) / abs(W), np.angle(Z) - np.angle(W)
        phi11 = phis
    else:
        phi11 = np.deg2rad(spec["phi11_deg"])
        F = v_fixed * np.exp(1j * phi11)
        # |Z/scale - F| = |Q(th0)|, positive root in u = 1/scale
        b, q = (Z * np.conj(F)).real, abs(Q(th0))
        u = (b + np.sqrt(b * b - abs(Z) ** 2 * (abs(F) ** 2 - q * q))) / abs(Z) ** 2
        scale = 1.0 / u
        phis = np.angle((Z * u - F) / Q(th0))
    psi = phis + np.deg2rad(spec["delta_deg"])
    fixed = scale * v_fixed * np.exp(1j * phi11)
    # Re[scale e^{i psi} Q(th)] = scale [v_scan cos psi + A cos th + B sin th]
    A, B = (p4 + p6) * np.cos(psi), (p4 - p6) * np.sin(psi)
    c = (Z.real - fixed.real) / scale - v_scan * np.cos(psi)
    R, beta = np.hypot(A, B), np.arctan2(B, A)
    saturated = abs(c) > R
    roots = [beta + s * np.arccos(np.clip(c / R, -1, 1)) for s in (1, -1)]
    th = min(roots, key=lambda r: abs(np.angle(np.exp(1j * (r - th0)))))
    M = fixed + scale * np.exp(1j * psi) * Q(th)
    deg = lambda r: float(np.rad2deg(r))
    return {"scale": float(scale), "phi11_deg": deg(phi11), "psi_deg": deg(psi), "theta_deg": deg(th),
            "phi_eff_deg": deg(np.angle(M)), "energy_ratio": float(M.real / Z.real),
            "amp_ratio": float(abs(M) / abs(Z)), "saturated": bool(saturated)}


def hold_quad_fields(tao) -> dict:
    """Freeze every quad's field at the present energy profile (FIELD_MASTER = T), as on the machine
    between LEMs; by default Bmad holds K1, i.e. re-LEMs whenever the energy changes. Returns
    {element index: K1} for release_quad_fields."""
    f = "-no_slaves -array_out"
    ix = tao.lat_list("Quadrupole::*", "ele.ix_ele", flags=f)
    k1 = tao.lat_list("Quadrupole::*", "ele.K1", flags=f)
    tao.cmd("set ele Quadrupole::* FIELD_MASTER = T")
    return {int(i): float(k) for i, k in zip(ix, k1)}


def release_quad_fields(tao, k1: dict):
    """Undo hold_quad_fields. Toggling FIELD_MASTER back keeps the held field, so K1 is restored
    explicitly (setLattice never rewrites the linac quads)."""
    tao.cmd("set ele Quadrupole::* FIELD_MASTER = F")
    for i, k in k1.items():
        tao.cmd(f"set ele {i} K1 = {k!r}")


def apply_l2_machine(tao, spec) -> tuple[dict, dict | None]:
    """After setLattice (every L2 cavity at the row's L2PhaseSet = the scan centre, uniform gradient
    on the energy target): re-phase and rescale L2 per l2_machine_solve. Returns its solution plus
    the L2 energy gain Tao reports before and after (equal when the solve held the energy), and the
    hold_quad_fields record when spec has "hold_delta_deg" (else None).

    hold_delta_deg: the multiknob offset the machine was LEM'd at. L2 is first set to that offset and
    every quad's field frozen there, then set to this step's offset, so the quads see the energy-
    profile change as a focusing error. The caller must release_quad_fields after tracking."""
    from FACET2_S2E.simulators.bmad.lattice.linac import getLinacMatchStrings
    names = [str(n) for n in getLinacMatchStrings(tao)[1]]
    get = lambda n, a: float(tao.lat_list(n, f"ele.{a}", flags="-no_slaves -array_out")[0])
    V = {n: get(n, "VOLTAGE") for n in names}
    G = {n: get(n, "GRADIENT") for n in names}
    phi0 = {n: get(n, "PHI0") for n in names}
    phi_m = 360.0 * phi0[names[0]]
    assert all(abs(360.0 * p - phi_m) < 1e-9 for p in phi0.values()), "L2 not at one phase"
    group = lambda n: ("fixed" if n.startswith("K11_") else "p4" if n.startswith("K14_4")
                       else "p6" if n.startswith("K14_6") else "scan")
    tot = {g: sum(v for n, v in V.items() if group(n) == g) for g in ("fixed", "scan", "p4", "p6")}
    energy = lambda: sum(get(n, "VOLTAGE") * np.cos(2 * np.pi * get(n, "PHI0")) for n in names)

    def set_l2(delta):
        sol = l2_machine_solve(tot["fixed"], tot["scan"], (tot["p4"], tot["p6"]), phi_m,
                               {**spec, "delta_deg": delta})
        phase = {"fixed": sol["phi11_deg"], "scan": sol["psi_deg"],
                 "p4": sol["psi_deg"] - sol["theta_deg"], "p6": sol["psi_deg"] + sol["theta_deg"]}
        for n in names:
            tao.cmd(f"set ele {n} PHI0 = {phase[group(n)] / 360.0}")
            tao.cmd(f"set ele {n} GRADIENT = {G[n] * sol['scale']}")
        return sol

    e0, held = energy(), None
    if spec.get("hold_delta_deg") is not None:
        set_l2(spec["hold_delta_deg"])
        held = hold_quad_fields(tao)
    sol = set_l2(spec["delta_deg"])
    sol["energy_before_eV"], sol["energy_after_eV"] = e0, energy()
    return sol, held


def track(qs, tao, s2e_root, settings: dict, points, *, n_in: int, save_beams_at=(),
          out_prefix=None, laser_heater: dict | None = None, l2_machine: dict | None = None) -> dict:
    """setLattice + trackBeam with the full merged `settings`, then per treaty point: optionally save
    the beam (`<out_prefix>_<ELE>.h5`), and record live counts + getBeamSpecs. Returns
    {"specs": {ele: {...}}} — the same structure as the campaign sample jsons.

    laser_heater: optional FACET2-S2E addLHmodulation kwargs (laserHeater_laserEnergy [J],
    laserHeater_sigma_t [s], laserHeater_offset [laser sigmas from the beam's mean t]) to turn the
    heater on at HTRUNDF. trackBeam forwards ALL its extra kwargs (here: every lattice setting) to
    addLHmodulation, so the heater call is wrapped to receive only these.

    l2_machine: optional apply_l2_machine spec (machine-like L2 scan step; settings' L2PhaseSet must
    be the scan centre; "hold_delta_deg" freezes the quad fields at that offset). Its solution is
    returned under "l2_machine"."""
    qs.setLattice(tao, **settings)
    l2, held = apply_l2_machine(tao, l2_machine) if l2_machine else (None, None)
    if not laser_heater:
        qs.trackBeam(tao, str(s2e_root), **settings)
    else:
        from FACET2_S2E.simulators.bmad import core
        orig = core.addLHmodulation
        core.addLHmodulation = lambda P, **_: orig(P, **laser_heater)
        try:
            qs.trackBeam(tao, str(s2e_root), laserHeater=True, **settings)
        finally:
            core.addLHmodulation = orig
    record = {"specs": {}, **({"l2_machine": l2} if l2 else {})}
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
    if held:
        release_quad_fields(tao, held)
    return record


# (drive, witness) split by per-particle weight: twobunch_cnf.beam_io (identical code)
from twobunch_cnf.beam_io import split_bunches  # noqa: E402,F401


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

