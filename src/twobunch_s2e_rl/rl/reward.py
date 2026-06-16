"""Composite, differentiable reward for two-bunch MBRL over the flow surrogate.

The reward consumes the dict returned by `TwoBunchFlow.observables(knobs, n)` (per-bunch +
inter-bunch + feasibility, all (B,) and differentiable w.r.t. the knobs) -- plus the env injects
the actual knobs under "knobs" for the trust-region term -- and maps it to a scalar per-env
reward. Every term is smooth so SHAC/BPTT can backprop through it.

Term shapes (see lab-notebook 2026-06-15 MBRL plan + the 2026-06-15 reward-design review):
- spacing       -> TARGET mode toward 200 um (two-sided; identity transform, signed quantity).
- emittance     -> per {drive,witness}x{x,y} norm_emit, a RANGE-NORMALIZED floor-barrier minimize:
                   monotone (lower better) in log10 down to a campaign-p10 floor (where the term
                   = -1), with a rising penalty BELOW the floor (so the policy can't drift into
                   the flow's sub-floor extrapolation regime). The beneficial (negative) part is
                   SURVIVAL-GATED by the bunch's T head so the agent can't earn emittance reward
                   for a bunch it is scraping away -- gated asymmetrically (only the reward part)
                   so it has no incentive to scrape-to-hide a bad emittance.
- survival      -> one-sided hinge relu((T_min+margin) - T_k) on the SMOOTH transmission heads
                   (NOT p_surv). The margin (~ the witness head MAE) keeps the agent off the cliff
                   where the head false-positives a lost bunch.
- trust-region  -> a knob-boundary penalty (ramps up within `boundary_margin` of a box edge) that
                   discourages railing knobs into the LHS-box corners where the flow extrapolates.
- collinearity  -> optional minimize on transverse_offset / angular_misalignment (default off).

Per-term z-scores / floors come from `compute_metric_norms` + `derive_floors_from_campaign` over
the campaign TRUE clouds, with the SAME flow kernels (RMS norm_emit, not getBeamSpecs SI90). The
cache records its provenance (campaign file + floor_pct) and is invalidated on mismatch.

Reward convention: reward = -(weighted sum of penalties); the trainers maximize reward.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from ..surrogate.properties import inter_bunch, per_bunch

# observables() dict keys grouped for convenience
EMIT_KEYS = ("drive_norm_emit_x", "drive_norm_emit_y",
             "witness_norm_emit_x", "witness_norm_emit_y")
SURV_KEYS = ("T_drive", "T_witness")
KNOBS_KEY = "knobs"   # injected by the env for the trust-region term
# keys exposed to the policy as (z-scored) achieved observables, appended to the knobs in obs
DEFAULT_OBS_KEYS = ("bunch_spacing", "T_drive", "T_witness") + EMIT_KEYS

# transform per metric key: log10 for strictly-positive right-skewed quantities, identity else
_TRANSFORM = {
    "bunch_spacing": "identity",          # signed (can be ~0) -> log ill-defined
    "energy_difference": "identity",
    "transverse_offset": "log10",
    "angular_misalignment": "log10",
    "T_drive": "identity",
    "T_witness": "identity",
    **{k: "log10" for k in EMIT_KEYS},
}

# which transmission head gates each emittance term (survival-gating, fix #1)
_GATE_FOR = {k: ("T_drive" if k.startswith("drive") else "T_witness") for k in EMIT_KEYS}


def _t(x: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log10":
        return torch.log10(x.abs().clamp_min(1e-30))
    return x


def _tval(v: float, transform: str) -> float:
    if transform == "log10":
        return float(np.log10(max(abs(v), 1e-30)))
    return float(v)


# ----- per-metric normalization + floors from the campaign data ----------------

def _campaign_metrics(h5_path: str, chunk: int = 256, device: str = "cpu") -> dict[str, np.ndarray]:
    """Run the flow per_bunch/inter_bunch kernels over the campaign TRUE clouds. Returns a dict
    of metric-key -> 1-D array over the density-valid rows for that metric (so emittance arrays
    cover only the rows where that bunch has a real cloud)."""
    with h5py.File(h5_path, "r") as h:
        n = h["knobs"].shape[0]
        dd_mask = np.asarray(h["drive_density"][...]).astype(bool)
        wd_mask = np.asarray(h["witness_density"][...]).astype(bool)
        dfrac = np.asarray(h["drive_frac"][...]).astype(np.float32)
        wfrac = np.asarray(h["witness_frac"][...]).astype(np.float32)
        acc: dict[str, list[np.ndarray]] = {}

        def push(key, vals, mask):
            acc.setdefault(key, []).append(vals[mask])

        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            d = torch.tensor(np.asarray(h["drive_parts"][sl]), dtype=torch.float64, device=device)
            w = torch.tensor(np.asarray(h["witness_parts"][sl]), dtype=torch.float64, device=device)
            dm, wm = dd_mask[sl], wd_mask[sl]
            both = dm & wm
            with torch.no_grad():
                pb_d = per_bunch(d)
                pb_w = per_bunch(w)
                ib = inter_bunch(d, w)
            for kk, vv in pb_d.items():
                push(f"drive_{kk}", vv.cpu().numpy(), dm)
            for kk, vv in pb_w.items():
                push(f"witness_{kk}", vv.cpu().numpy(), wm)
            for kk, vv in ib.items():
                push(kk, vv.cpu().numpy(), both)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["T_drive"] = dfrac[dd_mask]
    out["T_witness"] = wfrac[wd_mask]
    return out


def compute_metric_norms(h5_path: str, keys=None, chunk: int = 256) -> dict[str, dict]:
    """(transform, mean, std) per metric key over the campaign TRUE clouds, in the metric's
    transform space."""
    keys = keys or (("bunch_spacing", "energy_difference", "transverse_offset",
                     "angular_misalignment") + SURV_KEYS + EMIT_KEYS)
    metrics = _campaign_metrics(h5_path, chunk=chunk)
    norms = {}
    for k in keys:
        if k not in metrics:
            continue
        transform = _TRANSFORM.get(k, "identity")
        a = metrics[k]
        a = a[np.isfinite(a)]
        if a.size == 0:
            raise ValueError(f"metric {k!r} has no finite campaign rows -- cannot normalize")
        if transform == "log10":
            a = np.log10(np.clip(np.abs(a), 1e-30, None))
        norms[k] = {"transform": transform, "mean": float(a.mean()),
                    "std": float(max(a.std(), 1e-12))}
    return norms


def derive_floors_from_campaign(h5_path: str, pct: float = 10.0,
                                emit_keys=EMIT_KEYS, chunk: int = 256) -> dict[str, float]:
    """Per-emittance floor = the `pct`-percentile of the campaign distribution, in PHYSICAL
    units (the flow RMS norm_emit kernel)."""
    metrics = _campaign_metrics(h5_path, chunk=chunk)
    floors = {}
    for k in emit_keys:
        a = metrics[k]
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            raise ValueError(f"emittance {k!r} has no positive campaign rows -- cannot floor")
        floors[k] = float(np.percentile(a, pct))
    return floors


# ----- norms cache with provenance (fix #5) ------------------------------------

def _provenance(h5_path: str, floor_pct: float) -> dict:
    st = os.stat(h5_path)
    return {"campaign_h5": os.path.abspath(h5_path), "mtime": st.st_mtime,
            "size": st.st_size, "floor_pct": float(floor_pct)}


def save_norms(norms: dict, floors: dict, path: str, provenance: dict | None = None) -> None:
    with open(path, "w") as f:
        json.dump({"provenance": provenance or {}, "norms": norms, "floors": floors}, f, indent=2)


def load_norms(path: str) -> tuple[dict, dict, dict]:
    with open(path) as f:
        d = json.load(f)
    return d.get("provenance", {}), d["norms"], d["floors"]


# ----- reward terms ------------------------------------------------------------

@dataclass
class RewardTerm:
    """One differentiable penalty term. reward contribution = -weight * penalty(obs)."""
    key: str
    mode: str                      # target | minimize | minimize_floor | ceiling | hinge | boundary
    transform: str = "identity"
    mean: float = 0.0
    std: float = 1.0
    weight: float = 1.0
    target: float | None = None    # target mode (physical units)
    floor: float | None = None     # minimize_floor / ceiling (physical units)
    hinge_at: float | None = None  # hinge: relu(hinge_at - x) (physical units, e.g. T_min+margin)
    below_weight: float = 1.0      # minimize_floor: weight of the sub-floor barrier
    gate_key: str | None = None    # minimize_floor: survival head gating the reward part
    gate_low: float = 0.7          # gate ramp: 0 at/below gate_low, 1 at/above gate_high
    gate_high: float = 0.9
    margin: float = 0.05           # boundary mode: edge band width

    def _gate(self, obs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.gate_key is None:
            return None
        T = obs[self.gate_key]
        return ((T - self.gate_low) / max(self.gate_high - self.gate_low, 1e-6)).clamp(0.0, 1.0)

    def penalty(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "hinge":                                   # survival floor on T_k
            return torch.relu(self.hinge_at - obs[self.key])
        if self.mode == "boundary":                                # knob trust-region (fix #3)
            k = obs[self.key]                                      # (B, n_knob) in [0,1]
            prox = (torch.relu(k - (1.0 - self.margin)) + torch.relu(self.margin - k)) / self.margin
            return prox.mean(dim=-1)
        tx = _t(obs[self.key], self.transform)
        if self.mode == "target":
            return torch.abs(tx - _tval(self.target, self.transform)) / self.std
        if self.mode == "minimize":
            return (tx - self.mean) / self.std
        if self.mode == "maximize":
            return -(tx - self.mean) / self.std
        if self.mode == "ceiling":                                 # 0 below ceiling, ramp above
            return torch.relu(tx - _tval(self.floor, self.transform)) / self.std
        if self.mode == "minimize_floor":
            tfloor = _tval(self.floor, self.transform)
            denom = max(self.mean - tfloor, 1e-6)                  # range-norm: floor -> -1 (fix #2)
            above = (torch.clamp(tx, min=tfloor) - self.mean) / denom
            below = self.below_weight * torch.relu(tfloor - tx) / denom    # sub-floor barrier (fix #3)
            g = self._gate(obs)                                    # survival gating (fix #1)
            if g is not None:
                # gate only the beneficial (reward, negative) part, so a bad (positive) emittance
                # can't be hidden by scraping the bunch away.
                above = torch.where(above < 0, g * above, above)
            return above + below
        raise ValueError(f"unknown reward mode {self.mode!r}")


@dataclass
class TwoBunchRewardSpec:
    """Composite reward: reward = -(sum of weighted term penalties). Also exposes z-scored
    achieved observables for the env obs vector."""
    terms: list[RewardTerm]
    obs_keys: tuple[str, ...] = DEFAULT_OBS_KEYS
    obs_norms: dict[str, dict] = field(default_factory=dict)  # {key: {transform,mean,std}}

    def __call__(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = None
        breakdown = {}
        for term in self.terms:
            pen = term.weight * term.penalty(obs)
            breakdown[f"pen_{term.key}_{term.mode}"] = pen.detach()
            total = pen if total is None else total + pen
        reward = -total
        achieved = {k: obs[k].detach() for k in self._report_keys() if k in obs}
        achieved.update(breakdown)
        achieved["reward"] = reward.detach()
        return reward, achieved

    def _report_keys(self) -> tuple[str, ...]:
        return ("bunch_spacing", "energy_difference", "transverse_offset",
                "angular_misalignment", "p_surv", "T_drive", "T_witness") + EMIT_KEYS

    def obs_scaled(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, len(obs_keys)) z-scored achieved observables for the policy obs. Detached by the
        caller (env) -- the reward path is the only grad channel."""
        cols = []
        for k in self.obs_keys:
            spec = self.obs_norms.get(k, {"transform": _TRANSFORM.get(k, "identity"),
                                          "mean": 0.0, "std": 1.0})
            tx = _t(obs[k], spec["transform"])
            cols.append((tx - spec["mean"]) / spec["std"])
        return torch.stack(cols, dim=-1)

    @property
    def n_obs_extra(self) -> int:
        return len(self.obs_keys)


def build_twobunch_reward_spec(
    norms: dict[str, dict],
    floors: dict[str, float],
    *,
    spacing_target_m: float = 2e-4,
    surv_T_min: float = 0.9,
    surv_margin: float = 0.05,
    emit_mode: str = "minimize_floor",
    emit_keys=EMIT_KEYS,
    emit_below_weight: float = 1.0,
    emit_gate_band: float = 0.2,
    w_spacing: float = 1.0,
    w_emit: float = 1.0,
    w_emit_witness: float | None = None,
    w_surv: float = 1.0,
    w_ood: float = 0.5,
    boundary_margin: float = 0.05,
    w_collinearity: float = 0.0,
    obs_keys=DEFAULT_OBS_KEYS,
) -> TwoBunchRewardSpec:
    """Assemble the default composite spec.

    - emittance terms are range-normalized floor-barriers (fix #2/#3), survival-gated by the
      bunch's T head (fix #1); `w_emit_witness` overrides the witness weight (weakest observable).
    - survival hinge uses `surv_T_min + surv_margin` (fix #4).
    - a knob trust-region term (weight `w_ood`) discourages box-corner extrapolation (fix #3).
    """
    def nrm(k):
        if k not in norms:
            raise KeyError(f"metric {k!r} missing from norms; available: {sorted(norms)}")
        s = norms[k]
        return s["transform"], s["mean"], s["std"]

    terms: list[RewardTerm] = []

    tr, m, s = nrm("bunch_spacing")
    terms.append(RewardTerm("bunch_spacing", "target", tr, m, s, w_spacing, target=spacing_target_m))

    for k in emit_keys:
        tr, m, s = nrm(k)
        w = w_emit_witness if (w_emit_witness is not None and k.startswith("witness")) else w_emit
        terms.append(RewardTerm(
            k, emit_mode, tr, m, s, w, floor=floors.get(k),
            below_weight=emit_below_weight,
            gate_key=_GATE_FOR.get(k) if emit_mode == "minimize_floor" else None,
            gate_low=surv_T_min - emit_gate_band, gate_high=surv_T_min))

    for k in SURV_KEYS:
        terms.append(RewardTerm(k, "hinge", "identity", weight=w_surv,
                                hinge_at=surv_T_min + surv_margin))

    if w_ood > 0:
        terms.append(RewardTerm(KNOBS_KEY, "boundary", weight=w_ood, margin=boundary_margin))

    if w_collinearity > 0:
        for k in ("transverse_offset", "angular_misalignment"):
            tr, m, s = nrm(k)
            terms.append(RewardTerm(k, "minimize", tr, m, s, w_collinearity))

    obs_norms = {k: norms[k] for k in obs_keys if k in norms}
    return TwoBunchRewardSpec(terms=terms, obs_keys=tuple(obs_keys), obs_norms=obs_norms)


def reward_spec_from_campaign(h5_path: str, *, floor_pct: float = 10.0,
                              cache_json: str | None = None, **kwargs) -> TwoBunchRewardSpec:
    """Compute norms+floors from the campaign h5 (or load a provenance-matched cache) and build
    the spec. The cache is invalidated if the campaign file or floor_pct changed (fix #5)."""
    if cache_json is not None and os.path.exists(cache_json):
        want = _provenance(h5_path, floor_pct)
        prov, norms, floors = load_norms(cache_json)
        if prov == want:
            return build_twobunch_reward_spec(norms, floors, **kwargs)
    norms = compute_metric_norms(h5_path)
    floors = derive_floors_from_campaign(h5_path, pct=floor_pct)
    if cache_json is not None:
        save_norms(norms, floors, cache_json, _provenance(h5_path, floor_pct))
    return build_twobunch_reward_spec(norms, floors, **kwargs)
