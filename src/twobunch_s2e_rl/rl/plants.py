"""Plants for DeltaControlEnv: plain functions knobs_norm (B, n) -> {observable: (B,) tensor}.

- `flow_plant`: the flow's own observables at its treaty point (cloud mode, ground truth).
- `xtcav_screen_plant`: xtcav23 flow at XTCAVF -> back-drift to the TCAV entrance -> Cheetah
  segment with the TCAV on -> DTOTR -> particle-space screen features (screen mode). The
  ground-truth cloud twin of every screen feature is returned alongside as `<key>_cloud`, and the
  flow's own observables (survival heads etc.) pass through unchanged.

The segment quads (Q2/Q1/Q0FF, optionally the spectrometer Q0D/Q1D/Q2D) sit downstream of XTCAVF,
so the flow never sees them. By default they are held at the spec's golden values; `quad_jitter`
({name: fractional 1-sigma}) instead draws a hidden per-episode strength for each env
(`observe.resample(mask, generator)`, called by DeltaControlEnv at reset). The streak calibration
stays the golden one, so a quad error also acts as a calibration / resolution / energy-axis error,
as on the machine between phase-scan calibrations.
"""
from __future__ import annotations

import json

import torch

from ..surrogate.properties import inter_bunch, per_bunch

NC_TO_C = 1e-9


def injected_charges_nC(norm: dict, knob_keys, knob_low, knob_high):
    """knobs_norm (B, n) -> (q_drive, q_witness) injected charges [nC], (B,) each.

    Tables preprocessed with transmission='per_sample' label T = charge / that row's injected
    charge, which varies with the injector knobs: there q_in = Q f 1.001 (drive), Q (1 - f) 0.999
    (witness) -- the generation-time weight convention of sim/injector.py, exact to ~1e-4 against
    the recorded injected charges. Flows without injector knobs (or 'max' tables): the constant
    full charges of the norm."""
    keys = list(knob_keys)
    if norm.get("transmission") == "per_sample" and {"drive_fraction", "distgen:total_charge:value"} <= set(keys):
        jf, jq = keys.index("drive_fraction"), keys.index("distgen:total_charge:value")
        lo, hi = torch.as_tensor(knob_low), torch.as_tensor(knob_high)

        def q(knobs_norm):
            phys = lambda j: lo[j].to(knobs_norm) + knobs_norm[:, j] * (hi[j] - lo[j]).to(knobs_norm)
            Q, f = phys(jq) / 1000.0, phys(jf)
            return Q * f * 1.001, Q * (1 - f) * 0.999
        return q
    qd, qw = float(norm["drive_full_charge_nC"]), float(norm["witness_full_charge_nC"])
    return lambda knobs_norm: (torch.full(knobs_norm.shape[:1], qd, device=knobs_norm.device),
                               torch.full(knobs_norm.shape[:1], qw, device=knobs_norm.device))


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
                       resolution_m: float = 0.0, split: str = "energy",
                       threshold: str = "midpoint", row_correct: bool = False,
                       quad_jitter: dict[str, float] | None = None):
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
    quads = {name: element_by_name(segment, name) for name in (quad_jitter or {})}
    k1_golden = {name: q.k1.detach().clone() for name, q in quads.items()}
    norm = json.loads(open(norm_json).read())
    q_inj = injected_charges_nC(norm, flow.hparams.knob_keys or [], flow.knob_low.cpu(), flow.knob_high.cpu())

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
        q_in_d, q_in_w = q_inj(knobs_norm)
        q_d, q_w = obs["T_drive"] * q_in_d, obs["T_witness"] * q_in_w
        nd, nw = drive.shape[-2], witness.shape[-2]
        qpp = torch.cat([(q_d * NC_TO_C / nd).unsqueeze(-1).expand(-1, nd),
                         (q_w * NC_TO_C / nw).unsqueeze(-1).expand(-1, nw)], -1)
        xp, xm, y = track(torch.cat([drive, witness], -2), qpp)     # both bunches in one beam
        scr = screen_features(xp[:, :nd], xm[:, :nd], y[:, :nd], xp[:, nd:], xm[:, nd:], y[:, nd:],
                              q_d, q_w, streak=streak, eta_y=eta_y, ref_energy_eV=e_ref,
                              resolution_m=resolution_m, split=split, threshold=threshold,
                              row_correct=row_correct)
        cld = cloud_features(obs, q_d, q_w)
        for k in FEATURE_KEYS:
            obs[f"{k}_cloud"] = cld[k]
            obs[k] = scr[k]
        obs["screen_valid"] = scr["screen_valid"]
        obs["screen_misassign"] = scr["screen_misassign"]
        return obs

    def resample(mask: torch.Tensor, generator: torch.Generator):
        """Redraw the hidden quad strengths of the envs in `mask` (B,) bool; others keep theirs."""
        for name, q in quads.items():
            k1 = k1_golden[name] * (1 + quad_jitter[name] * torch.randn(
                mask.shape[0], generator=generator, device=mask.device, dtype=torch.float64))
            old = q.k1 if q.k1.dim() == 1 and q.k1.shape[0] == mask.shape[0] else k1_golden[name].expand_as(k1)
            q.k1 = torch.where(mask, k1, old)

    observe.resample = resample if quads else None
    observe.streak = streak
    observe.eta_y = eta_y
    return observe
