"""Plants for DeltaControlEnv: plain functions knobs_norm (B, n) -> {observable: (B,) tensor}.

- `flow_plant`: the flow's own observables at its treaty point (cloud mode, ground truth).
- `xtcav_screen_plant`: xtcav23 flow at XTCAVF -> back-drift to the TCAV entrance -> Cheetah
  segment with the TCAV on -> DTOTR -> particle-space screen features (screen mode). The
  ground-truth cloud twin of every screen feature is returned alongside as `<key>_cloud`, and the
  flow's own observables (survival heads etc.) pass through unchanged.

The segment quads (Q2/Q1/Q0FF) are held at the golden values stored in the segment spec in this
first version (they are downstream of XTCAVF and outside the controller's action set).
"""
from __future__ import annotations

import json

import torch

from ..surrogate.properties import inter_bunch, per_bunch

NC_TO_C = 1e-9


def _flow_observables(flow, knobs_norm, n):
    """TwoBunchFlow.observables, but also returning the sampled clouds."""
    drive = flow.sample_bunch(knobs_norm, 0, n)
    witness = flow.sample_bunch(knobs_norm, 1, n)
    p_surv, t_d, t_w = flow.feasibility(knobs_norm)
    out = {}
    for name, parts in (("drive", drive), ("witness", witness)):
        for k, v in per_bunch(parts).items():
            out[f"{name}_{k}"] = v
    out.update(inter_bunch(drive, witness))
    out.update(p_surv=p_surv, T_drive=t_d, T_witness=t_w)
    return out, drive, witness


def flow_plant(flow, n_particles: int):
    return lambda knobs_norm: flow.observables(knobs_norm, n=n_particles)


def xtcav_screen_plant(flow, n_particles: int, *, spec_path, norm_json, device,
                       tdc_voltage_V: float = 22e6, tdc_phase_deg: float = 0.0,
                       resolution_m: float = 0.0, split: str = "energy"):
    from xtcav_segment.convert import cloud_to_beam
    from xtcav_segment.features import FEATURE_KEYS, calibrate_streak, cloud_features, screen_features
    from xtcav_segment.segment import (back_drift, build_segment_from_spec, element_by_name,
                                      load_spec, set_tdc)

    spec = load_spec(spec_path)
    meta = spec["meta"]
    e_ref, l_tdc = float(meta["ref_energy_eV"]), float(meta["tdc_length_m"])
    eta_y = float(meta["R_entrance_to_exit"][2][5])          # energy-axis dispersion at DTOTR
    # one segment, TCAV voltage batched as (+V, -V): a single Cheetah call tracks both images
    segment, _ = build_segment_from_spec(spec, device=torch.device(device), dtype=torch.float64)
    set_tdc(segment, tdc_voltage_V, tdc_phase_deg)
    tdc = element_by_name(segment, "XTCAVF")
    tdc.voltage = torch.tensor([[tdc_voltage_V], [-tdc_voltage_V]], device=device, dtype=torch.float64)
    norm = json.loads(open(norm_json).read())
    q_full_d, q_full_w = norm["drive_full_charge_nC"], norm["witness_full_charge_nC"]

    def track(cloud, charge_per_particle):
        """(+V streak x, -V streak x, energy-axis y) at DTOTR, each (B, N). cloud (B, N, 6)."""
        beam = cloud_to_beam(cloud.double(), e_ref, charge_per_particle.double(),
                             device=torch.device(device))
        out = segment.track(back_drift(beam, l_tdc))     # exit-plane V=0 clouds -> TDC entrance
        return out.x[0].float(), out.x[1].float(), out.y[0].float()

    # streak calibration at the box centre (the sim analogue of the phase-scan calibration)
    with torch.no_grad():
        k0 = torch.full((1, flow.hparams.condition_dim), 0.5, device=device)
        d0 = flow.sample_bunch(k0, 0, 4096)
        xp, xm, _ = track(d0, torch.full((1, 4096), 1e-13, device=device))
        streak = calibrate_streak(0.5 * (xp - xm).double(), d0[..., 2].double())

    def observe(knobs_norm):
        obs, drive, witness = _flow_observables(flow, knobs_norm, n_particles)
        q_d, q_w = obs["T_drive"] * q_full_d, obs["T_witness"] * q_full_w
        nd, nw = drive.shape[-2], witness.shape[-2]
        qpp = torch.cat([(q_d * NC_TO_C / nd).unsqueeze(-1).expand(-1, nd),
                         (q_w * NC_TO_C / nw).unsqueeze(-1).expand(-1, nw)], -1)
        xp, xm, y = track(torch.cat([drive, witness], -2), qpp)     # both bunches in one beam
        scr = screen_features(xp[:, :nd], xm[:, :nd], y[:, :nd], xp[:, nd:], xm[:, nd:], y[:, nd:],
                              q_d, q_w, streak=streak, eta_y=eta_y, ref_energy_eV=e_ref,
                              resolution_m=resolution_m, split=split)
        cld = cloud_features(obs, q_d, q_w)
        for k in FEATURE_KEYS:
            obs[f"{k}_cloud"] = cld[k]
            obs[k] = scr[k]
        obs["screen_valid"] = scr["screen_valid"]
        obs["screen_misassign"] = scr["screen_misassign"]
        return obs

    observe.streak = streak
    observe.eta_y = eta_y
    return observe
