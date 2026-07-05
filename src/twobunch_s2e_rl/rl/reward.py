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
DRIVE_EMIT_KEYS = ("drive_norm_emit_x", "drive_norm_emit_y")
WITNESS_EMIT_KEYS = ("witness_norm_emit_x", "witness_norm_emit_y")
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

# default metric keys normalized from the campaign (the observables the reward can act on today)
_DEFAULT_NORM_KEYS = (("bunch_spacing", "energy_difference", "transverse_offset",
                       "angular_misalignment") + SURV_KEYS + EMIT_KEYS)

# goal-conditioned target: which obs key carries the per-env goal for a given metric. Only
# bunch_spacing is wired end-to-end in the env today (self._spacing_goal -> obs["spacing_goal"]).
_GOAL_KEY_FOR = {"bunch_spacing": "spacing_goal"}


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
    keys = keys or _DEFAULT_NORM_KEYS
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
                                emit_keys=EMIT_KEYS, chunk: int = 256,
                                pct_by_key: dict[str, float] | None = None) -> dict[str, float]:
    """Per-emittance floor = the percentile of the campaign distribution, in PHYSICAL units (the
    flow RMS norm_emit kernel). `pct` is the default; `pct_by_key` overrides it per metric (e.g. a
    lower drive percentile lets the policy chase the campaign's lower emittance tail)."""
    metrics = _campaign_metrics(h5_path, chunk=chunk)
    pct_by_key = pct_by_key or {}
    floors = {}
    for k in emit_keys:
        a = metrics[k]
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            raise ValueError(f"emittance {k!r} has no positive campaign rows -- cannot floor")
        floors[k] = float(np.percentile(a, pct_by_key.get(k, pct)))
    return floors


# ----- norms cache with provenance (fix #5) ------------------------------------

def _provenance(h5_path: str, floor_desc) -> dict:
    """`floor_desc` is whatever uniquely identifies the floors (a scalar pct, or a dict of
    {floor_pct, pct_by_key} for per-bunch floors) so the cache invalidates when floors change."""
    st = os.stat(h5_path)
    return {"campaign_h5": os.path.abspath(h5_path), "mtime": st.st_mtime,
            "size": st.st_size, "floor": floor_desc}


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
    goal_key: str | None = None    # target mode: per-env target read from obs[goal_key] (B,), physical
                                   # units; overrides the scalar `target` for goal-conditioned RL

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
            # per-env goal (goal-conditioned) overrides the scalar target; the goal is a detached
            # constant w.r.t. the knobs, so only `tx` carries grad -- d(penalty)/d(knobs) is unchanged.
            if self.goal_key is not None:
                tgt = _t(obs[self.goal_key], self.transform)   # (B,) physical -> transform space
            else:
                tgt = _tval(self.target, self.transform)       # scalar fallback (back-compat)
            return torch.abs(tx - tgt) / self.std
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


def _term_from_spec(e: dict, norms: dict, floors: dict, *, surv_T_min: float, surv_margin: float,
                    emit_below_weight: float, emit_gate_band: float,
                    boundary_margin: float) -> RewardTerm:
    """Build one RewardTerm from a config objective entry {key, kind, weight, ...}.

    kinds: target (scalar `target` or `goal:[lo,hi]` -> per-env goal_key), minimize/maximize/
    ceiling (optional `floor`), minimize_floor (campaign floor + survival gate), hinge
    (`hinge_at`, default surv_T_min+margin), boundary (knob trust-region, `margin`).
    """
    key, kind, w = e["key"], e["kind"], float(e.get("weight", 1.0))
    if kind == "boundary":
        return RewardTerm(key, "boundary", weight=w, margin=float(e.get("margin", boundary_margin)))
    if kind == "hinge":
        return RewardTerm(key, "hinge", "identity", weight=w,
                          hinge_at=float(e.get("hinge_at", surv_T_min + surv_margin)))
    if key not in norms:
        raise KeyError(f"objective metric {key!r} has no campaign norm; available: {sorted(norms)}")
    s = norms[key]
    tr, m, sd = s["transform"], s["mean"], s["std"]
    if kind == "target":
        goal_key = None
        if e.get("goal") is not None:
            if key not in _GOAL_KEY_FOR:
                raise ValueError(f"goal-conditioning not wired for {key!r} "
                                 f"(env supports {list(_GOAL_KEY_FOR)})")
            goal_key = _GOAL_KEY_FOR[key]
        return RewardTerm(key, "target", tr, m, sd, w,
                          target=float(e.get("target", 0.0)), goal_key=goal_key)
    if kind in ("minimize", "maximize", "ceiling"):
        return RewardTerm(key, kind, tr, m, sd, w,
                          floor=(float(e["floor"]) if "floor" in e else None))
    if kind == "minimize_floor":
        return RewardTerm(key, "minimize_floor", tr, m, sd, w, floor=floors.get(key),
                          below_weight=float(e.get("below_weight", emit_below_weight)),
                          gate_key=_GATE_FOR.get(key),
                          gate_low=surv_T_min - emit_gate_band, gate_high=surv_T_min)
    raise ValueError(f"unknown objective kind {kind!r} for {key!r}")


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
    spacing_goal_key: str | None = None,
    objectives: list[dict] | None = None,
) -> TwoBunchRewardSpec:
    """Assemble the composite spec.

    If `objectives` (a config list of {key, kind, weight, ...}) is given, terms are built from it
    and obs_keys are derived from the objective keys (+ per-env goal keys). Otherwise the default
    hardcoded composite is built (back-compat):
    - emittance terms are range-normalized floor-barriers (fix #2/#3), survival-gated by the
      bunch's T head (fix #1); `w_emit_witness` overrides the witness weight (weakest observable).
    - survival hinge uses `surv_T_min + surv_margin` (fix #4).
    - a knob trust-region term (weight `w_ood`) discourages box-corner extrapolation (fix #3).
    """
    if objectives is not None:
        terms = [_term_from_spec(e, norms, floors, surv_T_min=surv_T_min, surv_margin=surv_margin,
                                 emit_below_weight=emit_below_weight, emit_gate_band=emit_gate_band,
                                 boundary_margin=boundary_margin) for e in objectives]
        # obs = objective metrics (knobs are already the obs prefix), then per-env goals appended
        # at the end so the obs prefix stays stable (mirrors the goal-append invariant below).
        obs_keys = tuple(dict.fromkeys(e["key"] for e in objectives if e["key"] != KNOBS_KEY))
        obs_norms = {k: norms[k] for k in obs_keys if k in norms}
        for t in terms:
            if t.goal_key is not None and t.goal_key not in obs_keys:
                obs_keys = obs_keys + (t.goal_key,)
                obs_norms[t.goal_key] = norms[t.key]  # goal z-scored like its target metric
        return TwoBunchRewardSpec(terms=terms, obs_keys=obs_keys, obs_norms=obs_norms)

    def nrm(k):
        if k not in norms:
            raise KeyError(f"metric {k!r} missing from norms; available: {sorted(norms)}")
        s = norms[k]
        return s["transform"], s["mean"], s["std"]

    terms: list[RewardTerm] = []

    tr, m, s = nrm("bunch_spacing")
    terms.append(RewardTerm("bunch_spacing", "target", tr, m, s, w_spacing,
                            target=spacing_target_m, goal_key=spacing_goal_key))

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

    obs_keys = tuple(obs_keys)
    # goal-conditioned: expose the per-env target spacing to the policy, APPENDED to the end so the
    # obs prefix stays byte-identical to a non-goal run (preserves checkpoint/obs ordering). z-scored
    # with the bunch_spacing norms (same physical scale as the achieved spacing already in obs).
    if spacing_goal_key is not None and spacing_goal_key not in obs_keys:
        obs_keys = obs_keys + (spacing_goal_key,)
    obs_norms = {k: norms[k] for k in obs_keys if k in norms}
    if spacing_goal_key is not None:
        obs_norms[spacing_goal_key] = norms["bunch_spacing"]
    return TwoBunchRewardSpec(terms=terms, obs_keys=obs_keys, obs_norms=obs_norms)


def reward_spec_from_campaign(h5_path: str, *, floor_pct: float = 10.0,
                              drive_floor_pct: float | None = None,
                              witness_floor_pct: float | None = None,
                              cache_json: str | None = None,
                              objectives: list[dict] | None = None, **kwargs) -> TwoBunchRewardSpec:
    """Compute norms+floors from the campaign h5 (or load a provenance-matched cache) and build the
    spec. `floor_pct` is the default emittance-floor percentile; `drive_floor_pct`/`witness_floor_pct`
    override it per bunch (a lower drive percentile lets the policy chase a lower drive emittance,
    at the campaign tail's expense). With `objectives`, norms cover every objective metric and floors
    every minimize_floor key (per-entry `floor_pct` honored). The cache invalidates if the campaign
    file, any floor, or the objective set changed."""
    pct_by_key: dict[str, float] = {}
    if drive_floor_pct is not None:
        pct_by_key.update({k: float(drive_floor_pct) for k in DRIVE_EMIT_KEYS})
    if witness_floor_pct is not None:
        pct_by_key.update({k: float(witness_floor_pct) for k in WITNESS_EMIT_KEYS})

    # config-driven objectives: norms must cover every objective metric, floors every minimize_floor
    if objectives is not None:
        obj_keys = [e["key"] for e in objectives if e["key"] != KNOBS_KEY]
        norm_keys = tuple(dict.fromkeys(_DEFAULT_NORM_KEYS + tuple(obj_keys)))
        floor_keys = tuple(e["key"] for e in objectives if e.get("kind") == "minimize_floor")
        for e in objectives:
            if e.get("kind") == "minimize_floor" and "floor_pct" in e:
                pct_by_key[e["key"]] = float(e["floor_pct"])
        floor_desc = {"floor_pct": float(floor_pct), "pct_by_key": pct_by_key,
                      "objectives": [[e["key"], e.get("kind")] for e in objectives]}
    else:
        norm_keys, floor_keys = None, EMIT_KEYS
        floor_desc = ({"floor_pct": float(floor_pct), "pct_by_key": pct_by_key}
                      if pct_by_key else float(floor_pct))

    if cache_json is not None and os.path.exists(cache_json):
        want = _provenance(h5_path, floor_desc)
        prov, norms, floors = load_norms(cache_json)
        if prov == want:
            return build_twobunch_reward_spec(norms, floors, objectives=objectives, **kwargs)
    norms = compute_metric_norms(h5_path, keys=norm_keys)
    floors = derive_floors_from_campaign(h5_path, pct=floor_pct, emit_keys=floor_keys,
                                         pct_by_key=pct_by_key)
    if cache_json is not None:
        save_norms(norms, floors, cache_json, _provenance(h5_path, floor_desc))
    return build_twobunch_reward_spec(norms, floors, objectives=objectives, **kwargs)
