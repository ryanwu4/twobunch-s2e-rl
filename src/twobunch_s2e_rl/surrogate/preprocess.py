"""Build the flow-surrogate training table from a campaign data dir.

Walks data/<subdir>/sample_*.json (+ sample_*_PENT.h5) and writes
processed/twobunch_flow.h5:
  knobs          (N, 8)     float32  -- the 8 sweep knobs (raw)
  drive_parts    (N, P, 6)  float32  -- drive 6D cloud (x,y,z,px,py,pz), 0 if absent
  witness_parts  (N, P, 6)  float32  -- witness 6D cloud, 0 if not density-trainable
  drive_present  (N,)       bool     -- PDrive specs present (drive survived + characterized)
  witness_viable (N,)       bool     -- PWitness specs present (witness survived)
  drive_density  (N,)       bool     -- drive has >= P live particles (usable for NLL)
  witness_density(N,)       bool     -- witness viable AND >= P live particles
  drive_frac     (N,)       float32  -- drive surviving fraction (charge / full), clipped
  witness_frac   (N,)       float32  -- witness surviving fraction (0 if destroyed)
  n_drive,n_witness (N,)    int32    -- live macroparticle counts
  idx            (N,)       int32    -- sample index (traceability)

+ processed/twobunch_flow_norm.json: knob bounds (from sweep_params), per-bunch
StandardScaler (drive/witness mean[6], std[6]), full per-bunch charges, P, coord order.

Split into drive/witness by the two unique per-particle weights (driver = higher weight),
matching FACET2_S2E.getDriverAndWitness. Witness density/coords are kept only where the
json marks the witness viable; the ~31% destroyed witnesses still contribute the
feasibility (viability/transmission) labels. Viable witnesses with < P live particles
(~4% of viable -- near-destroyed, low-surviving-fraction remnants, NOT high-emittance
survivors) are dropped from density training but still feed the feasibility heads.

Run in the slac-rl env:
  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.preprocess [--subdir full] [--P 1024]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
from pmd_beamphysics import ParticleGroup

from ..datagen.paths import repo_root
from ..datagen.sweep_params import resolve_sweep_set
from . import COORD_KEYS, DEFAULT_P, ELECTRON_MC2_EV

# Default treaty point; override with --treaty-point (e.g. "XTCAVF" for the Stage-B
# GPSR campaign, where the surrogate hands off at the XTCAV exit plane, V=0).
PENT = "PENT"


def _stack(pg: ParticleGroup) -> np.ndarray:
    """(n, 6) in COORD_KEYS order: (x, y, z, px, py, pz). pos [m], mom [eV/c]."""
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float32)


def _subsample(coords: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(coords.shape[0], p, replace=False)
    return coords[idx]


def parse_one(json_path: str, p: int, rng: np.random.Generator, keys: list,
              treaty_point: str = PENT, fill: dict | None = None,
              transmission_ref: str = "BEGBC20") -> dict | None:
    with open(json_path) as f:
        d = json.load(f)
    if not d.get("success"):
        return None
    spec = d.get("specs", {}).get(treaty_point, {})
    # knobs this campaign did not sweep come from what it held them at (`fill`, from its
    # campaign.json); missing from both -> KeyError, never a silent default
    knobs = np.array([d["knobs"][k] if k in d["knobs"] else fill[k] for k in keys]
                     if fill else [d["knobs"][k] for k in keys], dtype=np.float32)

    drive_present = spec.get("PDrive_norm_emit_x") is not None
    witness_viable = spec.get("PWitness_norm_emit_x") is not None
    drive_charge = float(spec.get("PDrive_charge_nC") or 0.0)
    witness_charge = float(spec.get("PWitness_charge_nC") or 0.0)

    rec = {
        "idx": int(d["idx"]),
        "knobs": knobs,
        "drive_present": drive_present,
        "witness_viable": witness_viable,
        "drive_charge": drive_charge,
        "witness_charge": witness_charge,
        "drive_parts": np.zeros((p, 6), np.float32),
        "witness_parts": np.zeros((p, 6), np.float32),
        "drive_density": False,
        "witness_density": False,
        "n_drive": 0,
        "n_witness": 0,
        # injected per-bunch charge (injector campaigns record it; NaN = not recorded, i.e. the
        # golden beam's charge up to downsample noise). T = charge / full stays the training label.
        "drive_q_in": float(d.get("injected_charge_nC", {}).get("drive", np.nan)),
        "witness_q_in": float(d.get("injected_charge_nC", {}).get("witness", np.nan)),
        # per-bunch charge at an upstream reference point (stand-in for the injected charge of
        # campaigns that did not record it; xtcav23 has no loss upstream of BC20)
        "drive_q_ref": float((d.get("specs", {}).get(transmission_ref, {}).get("PDrive_charge_nC")) or np.nan),
        "witness_q_ref": float((d.get("specs", {}).get(transmission_ref, {}).get("PWitness_charge_nC")) or np.nan),
        # split-group fields (injector campaigns)
        "block": d.get("block"), "injector_point": d.get("injector_point"), "linac_id": d.get("linac_id"),
        # top-up campaigns: the campaign whose rows this row pairs with / whose injector point it shares
        "pair_campaign": d.get("pair_campaign"), "point_campaign": d.get("point_campaign"),
        "holdout": bool(d.get("holdout", False)),     # pre-registered test rows (injector campaigns)
    }

    if not (drive_present or witness_viable):
        return rec  # specs_error: feasibility-only (no particles extracted)

    h5_path = json_path.replace(".json", f"_{treaty_point}.h5")
    if not os.path.exists(h5_path):
        return rec
    try:
        P = ParticleGroup(h5_path)
    except Exception:
        return rec
    w = np.unique(P.weight)

    # drive = higher-weight subset (matches getDriverAndWitness); witness = lower.
    if len(w) >= 2:
        drive_pg = P[P.weight == w[-1]]
        witness_pg = P[P.weight == w[0]] if witness_viable else None
    else:
        drive_pg, witness_pg = P, None  # single surviving bunch is the drive

    if drive_present:
        nd = len(drive_pg.x)
        rec["n_drive"] = nd
        if nd >= p:
            rec["drive_parts"] = _subsample(_stack(drive_pg), p, rng)
            rec["drive_density"] = True
    if witness_viable and witness_pg is not None:
        nw = len(witness_pg.x)
        rec["n_witness"] = nw
        if nw >= p:
            rec["witness_parts"] = _subsample(_stack(witness_pg), p, rng)
            rec["witness_density"] = True
    return rec


def _scaler(records, key_parts, key_density, mode="intrabunch"):
    """Per-dim mean/std over density-trainable bunches of one species.

    mode='pooled'     : std of pooled raw particles (v1) -- for the witness this is dominated
                        by inter-sample centroid spread (~25x the intra-bunch size), so the
                        standardized intra-bunch shape becomes tiny and the whitening Sigma_k
                        target is badly conditioned.
    mode='intrabunch' : std of pooled PER-SAMPLE-CENTERED particles (v2 default) -> std reflects
                        the typical intra-bunch size, so standardized intra-bunch shape is ~O(1)
                        and the whitening regresses a well-scaled (mu_k, Sigma_k). Mean is the
                        pooled raw mean either way (the placement frame).
    """
    # float64 accumulation: pz ~1e10 eV/c summed over millions of float32 particles loses
    # ~340 MeV of precision in the mean (= ~21 sigma of the *witness* pz spread, but only
    # ~2 sigma for the fatter drive) -- which corrupts the witness LPS frame. Cast first.
    parts = [r[key_parts].astype(np.float64) for r in records if r[key_density]]
    if not parts:
        return np.zeros(6, np.float32), np.ones(6, np.float32)
    pooled = np.concatenate(parts, axis=0)
    mean = pooled.mean(axis=0)
    if mode == "pooled":
        std = pooled.std(axis=0)
    else:
        centered = np.concatenate([p - p.mean(axis=0, keepdims=True) for p in parts], axis=0)
        std = centered.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-12).astype(np.float32)


# fixed-block entries that must agree between merged campaigns (unless downstream)
_MERGE_KEYS = ("baseline_config_sha256", "input_beam_sha256", "csrTF", "transverseWakes",
               "num_macro_particles")


def campaign_fill(camp: dict, keys, knob_low, knob_high, treaty_point: str,
                  allow_drop=()) -> dict:
    """Values for the target knobs this campaign did NOT sweep (its recorded fixed settings), after
    the knob-drop rule and the in-frame check. Raises ValueError on a violation.

    `allow_drop`: swept knobs the caller explicitly marginalizes over (e.g. the "passive" injector
    knobs, whose effect then becomes label noise) -- an opt-in exception to the knob-drop rule.
    Injector knobs a campaign held fixed come from fixed["injector_settings"] (golden for every
    golden-beam campaign; see datagen/injector_campaign.py)."""
    from ..datagen.sweep_params import DOWNSTREAM_OF
    swept = set(camp["swept"]["knob_keys"])
    dropped = sorted(swept - set(keys) - set(DOWNSTREAM_OF.get(treaty_point, [])) - set(allow_drop))
    if dropped:
        raise ValueError(f"campaign {camp['name']!r} varied {dropped}, which the target model "
                         f"lacks and which are not downstream of {treaty_point}")
    fixed = {**camp["fixed"]["baseline_settings"], **camp["fixed"]["fixed_overrides"],
             **camp["fixed"].get("injector_settings", {})}
    fill = {}
    for k, lo, hi in zip(keys, knob_low, knob_high):
        if k in swept:
            continue
        if k not in fixed:
            raise ValueError(f"campaign {camp['name']!r} records no fixed value for {k!r}")
        v = float(fixed[k])
        if not (lo - 1e-9 * abs(hi - lo) <= v <= hi + 1e-9 * abs(hi - lo)):
            raise ValueError(f"campaign {camp['name']!r} held {k}={v}, outside the target frame "
                             f"[{lo}, {hi}] -- use a frame that contains it (e.g. a union set)")
        fill[k] = v
    return fill


def _beam_identity(camp: dict):
    """What the campaign's input beam is, for merge checks. Golden-beam campaigns: the h5 sha.
    Impact-T injector campaigns (fixed.injector.source == 'impact'): the sha of the golden beam
    their injector-at-golden point stands in for -- so they merge with golden-beam campaigns
    (embedded at injector = golden), and with each other only at identical injector fidelity."""
    inj = camp["fixed"].get("injector")
    if inj and inj.get("source") == "impact":
        return inj.get("golden_equivalent_input_beam_sha256")
    return camp["fixed"].get("input_beam_sha256")


# injector-fidelity entries that must agree between merged Impact-T campaigns
_INJ_MERGE_KEYS = ("n_particles", "grid", "code_version", "arrival", "hammersley_burnin",
                   "l0af_reference")


def check_mergeable(camps: list[dict], treaty_point: str, allow_mismatch=()) -> None:
    """Campaigns merged into one table must share every fixed setting that affects the beam at the
    treaty point (fixed_overrides other than downstream-only settings, plus _MERGE_KEYS).
    allow_mismatch: _MERGE_KEYS explicitly allowed to differ (opt-in, recorded in the table attrs;
    e.g. num_macro_particles after a convergence study showed the difference is negligible)."""
    from ..datagen.sweep_params import DOWNSTREAM_OF, DOWNSTREAM_SETTINGS
    skip = set(DOWNSTREAM_SETTINGS) | set(DOWNSTREAM_OF.get(treaty_point, []))
    ref = camps[0]
    inj_camps = [c for c in camps if (c["fixed"].get("injector") or {}).get("source") == "impact"]
    for c in inj_camps[1:]:
        for k in _INJ_MERGE_KEYS:
            if c["fixed"]["injector"].get(k) != inj_camps[0]["fixed"]["injector"].get(k):
                raise ValueError(f"cannot merge {inj_camps[0]['name']!r} and {c['name']!r}: "
                                 f"injector {k} differs")
    for c in camps[1:]:
        if _beam_identity(c) != _beam_identity(ref):
            raise ValueError(f"cannot merge {ref['name']!r} and {c['name']!r}: fixed input beam "
                             "(input_beam_sha256 / injector golden equivalent) differs")
        for k in _MERGE_KEYS:
            if k == "input_beam_sha256" or k in allow_mismatch:
                continue
            if c["fixed"].get(k) != ref["fixed"].get(k):
                raise ValueError(f"cannot merge {ref['name']!r} and {c['name']!r}: fixed {k} differs")
        fo_a, fo_b = ref["fixed"]["fixed_overrides"], c["fixed"]["fixed_overrides"]
        for k in (set(fo_a) | set(fo_b)) - skip:
            if fo_a.get(k) != fo_b.get(k):
                raise ValueError(f"cannot merge {ref['name']!r} and {c['name']!r}: fixed_overrides "
                                 f"{k} = {fo_a.get(k)} vs {fo_b.get(k)}")


def split_groups(records) -> np.ndarray:
    """Group id per row for a leak-free train/val split: rows of the same (campaign, injector
    point) form one group, and a paired_golden row joins the group of the nested row it pairs with
    (same campaign + linac_id). The shared golden injector point is NOT a link (it would chain the
    whole campaign together); rows without injector-campaign fields are their own group."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_point, by_linac = {}, {}
    for i, r in enumerate(records):
        if r.get("injector_point") not in (None, "golden", "golden_h5"):
            by_point.setdefault((r.get("point_campaign") or r.get("campaign_name", r["campaign"]),
                                 r["injector_point"]), []).append(i)
        if r.get("block") in ("nested", "paired_golden") and r.get("linac_id") is not None:
            by_linac.setdefault((r.get("pair_campaign") or r.get("campaign_name", r["campaign"]),
                                 r["linac_id"]), []).append(i)
    for members in list(by_point.values()) + list(by_linac.values()):
        for j in members[1:]:
            union(members[0], j)
    roots = [find(i) for i in range(len(records))]
    _, ids = np.unique(roots, return_inverse=True)
    return ids.astype(np.int32)


def preprocess(subdir="full", p=DEFAULT_P, max_samples=None, seed=0, out=None,
               scaler="intrabunch", sweep_set="original8", treaty_point=PENT,
               verbose=True, data_root=None, campaign_jsons=None, allow_drop=(),
               transmission="max", transmission_ref="BEGBC20", allow_fixed_mismatch=()):
    # transmission: "max" (v1-v3 tables): T = charge / max charge over records -- carries the
    #   +-0.9% downsampling noise of the golden-beam campaigns, so T ~0.986 with no loss.
    #   "per_sample": T = charge / that row's injected charge (recorded by injector campaigns; else
    #   the charge at `transmission_ref`). The norm then records transmission="per_sample" and the
    #   median injected charges as the full charges.
    # subdir may be comma-separated to MERGE campaigns (e.g. "tightbox_v2_full,expanded_full").
    # When merging boxes with different bounds, pass a union sweep_set (e.g. "tightbox+expanded")
    # so every draw is normalized in one common frame; parse_one stores raw knobs, so the frame
    # lives entirely in norm.json (knob_low/high) -- the RL env must resolve the same union name.
    subdirs = [s.strip() for s in str(subdir).split(",") if s.strip()]
    data_root = Path(data_root) if data_root else repo_root() / "data"
    keys, knob_low, knob_high, _ = resolve_sweep_set(sweep_set)
    files, file_campaign, fills, camps = [], [], {}, []
    for ci, sd in enumerate(subdirs):
        fs = sorted(glob.glob(str(data_root / sd / "sample_*.json")))
        if max_samples:
            fs = fs[:max_samples]
        if not fs:
            raise SystemExit(f"No sample_*.json under data/{sd}")
        files.extend(fs)
        file_campaign.extend([ci] * len(fs))
        # campaigns that recorded campaign.json can be embedded into a bigger knob space; older
        # campaigns (no campaign.json) are parsed exactly as before
        # campaign_jsons: {subdir: path} -- a campaign record kept OUTSIDE its data dir (e.g. a
        # backfill for a campaign that predates campaign.json; data/ stays untouched)
        cj = Path((campaign_jsons or {}).get(sd, data_root / sd / "campaign.json"))
        if cj.exists():
            camp = json.loads(cj.read_text())
            camps.append(camp)
            fills[ci] = campaign_fill(camp, keys, knob_low, knob_high, treaty_point, allow_drop)
    if len(camps) > 1:
        check_mergeable(camps, treaty_point, allow_mismatch=tuple(allow_fixed_mismatch))
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"Parsing {len(files)} samples from {subdirs} (P={p}, sweep_set={sweep_set}, "
              f"{len(keys)} knobs, treaty_point={treaty_point})")

    records = []
    for i, (fp, ci) in enumerate(zip(files, file_campaign)):
        rec = parse_one(fp, p, rng, keys, treaty_point=treaty_point, fill=fills.get(ci),
                        transmission_ref=transmission_ref)
        if rec is not None:
            rec["campaign"] = ci
            rec["campaign_name"] = subdirs[ci]
            records.append(rec)
        if verbose and (i + 1) % 500 == 0:
            nd = sum(r["drive_density"] for r in records)
            nw = sum(r["witness_density"] for r in records)
            print(f"  {i+1}/{len(files)}: kept {len(records)} (drive-dens {nd}, witness-dens {nw})")

    n = len(records)
    # Guard against the silent wrong---sweep-set footgun: a 23/26-dim campaign parsed
    # with a smaller set's keys succeeds (key subset) but yields an out-of-frame table.
    K = np.stack([r["knobs"] for r in records]).astype(np.float64)
    lo, hi = np.array(knob_low, np.float64), np.array(knob_high, np.float64)
    tol = 1e-6 * (hi - lo)
    oob = ((K < lo - tol) | (K > hi + tol)).any(axis=1)
    if oob.any():
        print(f"WARNING: {int(oob.sum())}/{n} samples have knobs OUTSIDE the "
              f"'{sweep_set}' bounds — wrong --sweep-set for this campaign?")

    drive_mean, drive_std = _scaler(records, "drive_parts", "drive_density", scaler)
    witness_mean, witness_std = _scaler(records, "witness_parts", "witness_density", scaler)
    if transmission == "max" and any(np.isfinite(r["drive_q_in"]) for r in records):
        raise ValueError("rows record injected charges (an injector campaign): transmission='max' "
                         "would fold the drive-fraction/charge variation into T; use per_sample")
    if transmission == "max":
        drive_full = max((r["drive_charge"] for r in records), default=1.0) or 1.0
        witness_full = max((r["witness_charge"] for r in records), default=1.0) or 1.0
        for r in records:
            r["drive_ref"], r["witness_ref"] = drive_full, witness_full
    elif transmission == "per_sample":
        for r in records:
            for b in ("drive", "witness"):
                q = r[f"{b}_q_in"] if np.isfinite(r[f"{b}_q_in"]) else r[f"{b}_q_ref"]
                if not np.isfinite(q) or q <= 0:
                    raise ValueError(f"row idx={r['idx']} (campaign {r['campaign']}): no injected "
                                     f"charge and no {transmission_ref} charge for the {b}")
                r[f"{b}_ref"] = q
        drive_full = float(np.median([r["drive_ref"] for r in records]))
        witness_full = float(np.median([r["witness_ref"] for r in records]))
    else:
        raise ValueError(f"transmission must be 'max' or 'per_sample', got {transmission!r}")

    def frac(c, full):
        return float(np.clip(c / full, 0.0, 1.05))

    out = out or str(repo_root() / "processed" / "twobunch_flow.h5")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, "w") as h:
        h.create_dataset("knobs", data=np.stack([r["knobs"] for r in records]), compression="gzip")
        h.create_dataset("drive_parts", data=np.stack([r["drive_parts"] for r in records]),
                         compression="gzip", chunks=(1, p, 6))
        h.create_dataset("witness_parts", data=np.stack([r["witness_parts"] for r in records]),
                         compression="gzip", chunks=(1, p, 6))
        for col, dt in [("drive_present", bool), ("witness_viable", bool),
                        ("drive_density", bool), ("witness_density", bool)]:
            h.create_dataset(col, data=np.array([r[col] for r in records], dtype=dt))
        h.create_dataset("drive_frac", data=np.array([frac(r["drive_charge"], r["drive_ref"]) for r in records], np.float32))
        h.create_dataset("witness_frac", data=np.array([frac(r["witness_charge"], r["witness_ref"]) for r in records], np.float32))
        if transmission == "per_sample":
            h.create_dataset("drive_q_ref_nC", data=np.array([r["drive_ref"] for r in records], np.float32))
            h.create_dataset("witness_q_ref_nC", data=np.array([r["witness_ref"] for r in records], np.float32))
        h.create_dataset("group", data=split_groups(records))
        h.create_dataset("holdout", data=np.array([r["holdout"] for r in records], bool))
        # pairs (injector campaigns): a nested row and its paired_golden row share (campaign,
        # pair_key); pair_ref marks the injector=golden member. -1 = not part of a pair.
        # globally unique across campaigns: 10^7 * index of the pairing campaign + linac_id (a top-up
        # row's pair_campaign links it to its partner in another campaign; consumers pair by key alone)
        cidx = {name: j for j, name in enumerate(subdirs)}
        pkey = lambda r: 10_000_000 * cidx[r.get("pair_campaign") or r["campaign_name"]] + int(r["linac_id"])
        h.create_dataset("pair_key", data=np.array(
            [pkey(r) if r.get("block") in ("nested", "paired_golden") and r.get("linac_id") is not None
             else -1 for r in records], np.int64))
        blocks = ["", "crossed", "nested", "paired_golden", "bridge", "jacobian"]
        h.create_dataset("block", data=np.array([blocks.index(r.get("block")) if r.get("block") in blocks else 0
                                                 for r in records], np.int8))
        h.attrs["block_codes"] = ",".join(blocks)
        h.create_dataset("linac_id", data=np.array([int(r["linac_id"]) if r.get("linac_id") is not None else -1
                                                    for r in records], np.int64))
        h.create_dataset("injector_golden", data=np.array([r.get("injector_point") == "golden" for r in records], bool))
        h.create_dataset("pair_ref", data=np.array([r.get("block") == "paired_golden" for r in records], bool))
        h.create_dataset("n_drive", data=np.array([r["n_drive"] for r in records], np.int32))
        h.create_dataset("n_witness", data=np.array([r["n_witness"] for r in records], np.int32))
        h.create_dataset("idx", data=np.array([r["idx"] for r in records], np.int32))
        for col in ("drive_q_in", "witness_q_in"):
            v = np.array([r[col] for r in records], np.float32)
            if np.isfinite(v).any():          # only injector campaigns record it
                h.create_dataset(f"{col}_nC", data=v)
        # source campaign of every row (idx collides across merged campaigns)
        h.create_dataset("campaign", data=np.array([r["campaign"] for r in records], np.int16))
        h.attrs["campaigns"] = ",".join(subdirs)
        if allow_fixed_mismatch:
            h.attrs["allowed_fixed_mismatch"] = ",".join(allow_fixed_mismatch)
        h.attrs["P"] = p
        h.attrs["coord_order"] = ",".join(COORD_KEYS)

    norm = {
        "knob_keys": list(keys),
        "knob_low": [float(v) for v in knob_low],
        "knob_high": [float(v) for v in knob_high],
        "drive_mean": drive_mean.tolist(), "drive_std": drive_std.tolist(),
        "witness_mean": witness_mean.tolist(), "witness_std": witness_std.tolist(),
        "drive_full_charge_nC": drive_full, "witness_full_charge_nC": witness_full,
        "coord_keys": list(COORD_KEYS), "electron_mc2_ev": ELECTRON_MC2_EV,
        "P": int(p), "n_records": n, "scaler": scaler, "sweep_set": sweep_set,
        "treaty_point": treaty_point, "transmission": transmission,
    }
    with open(out.replace(".h5", "_norm.json"), "w") as f:
        json.dump(norm, f, indent=2)

    if verbose:
        nd = sum(r["drive_density"] for r in records)
        nw = sum(r["witness_density"] for r in records)
        wv = sum(r["witness_viable"] for r in records)
        print(f"Wrote {out}: N={n}, drive-density={nd}, witness-viable={wv}, witness-density={nw}")
        print(f"  full charge nC: drive {drive_full:.4f}, witness {witness_full:.4f}")
    return norm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdir", default="full",
                    help="data/<subdir>; comma-separated to MERGE campaigns "
                         "(e.g. 'tightbox_v2_full,expanded_full')")
    ap.add_argument("--P", type=int, default=DEFAULT_P)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scaler", choices=["pooled", "intrabunch"], default="intrabunch",
                    help="per-bunch std basis (v2 default 'intrabunch'; v1 used 'pooled')")
    ap.add_argument("--sweep-set", default="original8",
                    help="knob set (sweep_params.SWEEP_SETS): original8 | expanded | expanded_anchored "
                         "| tightbox | xtcav23; a '+'-joined name (e.g. 'tightbox+expanded') uses the "
                         "union bounds -- the common frame when merging boxes")
    ap.add_argument("--treaty-point", default=PENT,
                    help="element whose specs/beams feed the table (default PENT; the Stage-B "
                         "GPSR campaign uses XTCAVF = handoff at the XTCAV exit plane, V=0)")
    ap.add_argument("--campaign-json", action="append", default=[], metavar="SUBDIR=PATH",
                    help="campaign record for SUBDIR kept outside data/<SUBDIR> (e.g. a backfill); "
                         "repeatable")
    ap.add_argument("--transmission", choices=["max", "per_sample"], default="max",
                    help="transmission label: 'max' (v1-v3; charge / max over records) or "
                         "'per_sample' (charge / that row's injected charge)")
    ap.add_argument("--transmission-ref", default="BEGBC20",
                    help="treaty point whose charge stands in for the injected charge when a row "
                         "did not record it (per_sample mode)")
    ap.add_argument("--allow-drop", default="",
                    help="comma-separated swept knobs to marginalize over (opt-in exception to the "
                         "knob-drop rule)")
    ap.add_argument("--allow-fixed-mismatch", default="",
                    help="comma-separated fixed campaign keys allowed to differ between merged "
                         "campaigns (opt-in; recorded in the table attrs), e.g. num_macro_particles")
    args = ap.parse_args()
    cjs = dict(s.split("=", 1) for s in args.campaign_json)
    preprocess(subdir=args.subdir, p=args.P, max_samples=args.max_samples,
               seed=args.seed, out=args.out, scaler=args.scaler, sweep_set=args.sweep_set,
               treaty_point=args.treaty_point, campaign_jsons=cjs,
               allow_drop=[k for k in args.allow_drop.split(",") if k],
               transmission=args.transmission, transmission_ref=args.transmission_ref,
               allow_fixed_mismatch=[k for k in args.allow_fixed_mismatch.split(",") if k])


if __name__ == "__main__":
    main()
