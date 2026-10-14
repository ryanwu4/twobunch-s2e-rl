"""Minimal Impact-T injector wrapper: cathode (distgen) -> L0AFEND two-bunch beam.

Adapted from FACET2_S2E.simulators.impact.runImpact, in the form that produced the golden
beam beams/2024-10-14_Impact_TwoBunch/2024-10-14_TwoBunch.h5 (Source.zip notebook
"2024-10-14 New golden IMPACT two-bunch": runImpact(gridCount=100, numMacroParticles=5e5,
GFILESuffix='distgen_twobunch.yaml', solenoidTValue=-0.39), then
nudgeMacroparticleWeights(trailingBunchFraction=0.75, trailingBunchType="driver"), then
drift_to_z()). Differences from runImpact, all deliberate:

- Per-run working directory. The FACET2-S2E impact/ templates are COPIED into
  <workdir>/inputs/; nothing is ever written into FACET2-S2E and no shared ImpactT.yaml is
  rewritten (runImpact rewrites <root>/impact/ImpactT.yaml on every call).
- Every injector variable is a settings key (lume-impact / distgen keys where one exists;
  a few derived keys documented in DERIVED_KEYS).
- L0AF phase reference pinned: autophase ONCE at the golden gun settings
  (pin_l0af_reference), cache the phase/scale, and reuse it for every perturbed run, so an
  injector perturbation propagates through L0AF instead of being re-phased away.
- Drive/witness tagged AT GENERATION (see tag_components) and carried through Impact-T by
  the per-particle id column, instead of the post-hoc z-quantile split of
  nudgeMacroparticleWeights. Weight convention is the golden one exactly (see below).
- <t> at L0AFEND is recorded BEFORE time-centring (runImpact / modifyAndSaveInputBeam both
  subtract it, which erases injector arrival-time shifts before Bmad).

Charge-ratio / weight convention (read from the golden beam, preserved exactly):
  distgen t_dist is a `superposition` of two Gaussians, d1 (avg_t 0 ps, weight 0.25) and
  d2 (avg_t 9 ps, weight 0.75). distgen normalizes each component by its PEAK (not area),
  so for equal sigma_t the weights are the fraction of MACROPARTICLES per pulse; all
  macroparticles leave Impact-T with the same charge. The charge ratio is therefore encoded
  in particle COUNTS (golden: 125000 witness / 375000 drive of 5e5), not weights. Identity is
  then a +-0.1% weight nudge: drive = 1.001 q, witness = 0.999 q (q = Q/N), which also raises
  the total charge by 0.05% (golden: 1.6008 nC). Drive = d2 = the later-emitted, 75% pulse
  (it TRAILS the witness at L0AFEND; golden: trailingBunchType="driver"). Downstream code
  splits by np.unique(P.weight): drive = higher weight.

Runs in the bmad-qpad-dev env (distgen + lume-impact + ImpactTexe). The pure helpers
(settings resolution, tagging, cache key, workdir layout) import nothing heavy, so the unit
tests run in either env.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------------------
# Constants (from runImpact / the golden notebook)
# ---------------------------------------------------------------------------------------

DISTGEN_FILE = "distgen_twobunch.yaml"
IMPACT_YAML = "ImpactT-template.yaml"
IMPACT_IN = "ImpactT_template.in"
RFDATA = ("rfdata4", "rfdata5", "rfdata6", "rfdata7", "rfdata102", "rfdata201")
TEMPLATE_FILES = (DISTGEN_FILE, IMPACT_YAML, IMPACT_IN) + RFDATA

L0AF_E_GAIN_EV = 62924849.46502216   # L0AF energy gain from Jan-2024 power readbacks (runImpact)
SOL_PV_TO_SIM = 1.6                  # solenoid PV value -> Impact-T field scale (runImpact)
L0AFEND_Z_M = 4.127448               # write_beam location of L0AFEND in ImpactT_template.in
STOP_S_M = 4.2                       # runImpact 'stop_1:s'
S_BAND_DEG_PER_PS = 360 * 2.856e9 * 1e-12   # 1.028 deg/ps
WEIGHT_NUDGE = 1e-3                  # nudgeMacroparticleWeights: drive 1.001 q, witness 0.999 q

# Component names in distgen_twobunch.yaml t_dist superposition.
WITNESS_COMPONENT = "d1"   # avg_t 0 ps, weight 0.25 -> leads at L0AFEND
DRIVE_COMPONENT = "d2"     # avg_t 9 ps, weight 0.75 -> trails at L0AFEND

# Golden settings of the 2024-10-14 beam. Units: deg, V/m-scale, field scale, ps, pC, mm, meV.
GOLDEN_SETTINGS = {
    # lume-impact keys
    "GUNF:theta0_deg": 29.3 - 90.0,                       # "New from Eric 2024-10-07"
    "GUNF:rf_field_scale": 119e6 / 2.44885,
    "SOL10111:solenoid_field_scale": -0.39 / SOL_PV_TO_SIM,  # golden solenoidTValue = -0.39
    # distgen keys (values in the units of distgen_twobunch.yaml)
    "distgen:total_charge:value": 1600.0,                 # pC
    "distgen:r_dist:sigma_xy:value": 1.49,                # mm
    "distgen:r_dist:truncation_radius:value": 2.52,       # mm
    "distgen:start:MTE:value": 400.0,                     # meV
    "distgen:t_dist:dists:d1:sigma_t:value": 1.17,        # ps
    "distgen:t_dist:dists:d2:sigma_t:value": 1.17,        # ps
    "distgen:t_dist:dists:d2:avg_t:value": 9.0,           # ps; inter-bunch delay (d1 at 0)
    "distgen:t_dist:weights:d1": 0.25,                    # witness fraction (macroparticles)
    "distgen:t_dist:weights:d2": 0.75,                    # drive fraction
    # derived keys (DERIVED_KEYS)
    "laser_timing_ps": 0.0,
    "L0AF_phase_offset_deg": 0.0,
    "hammersley_burnin": -1,
}

# Keys with no single lume-impact/distgen counterpart, and what they do.
DERIVED_KEYS = {
    "laser_timing_ps": "common-mode laser-to-RF timing: added to BOTH t_dist avg_t (d1 and d2). "
                       "write_impact keeps absolute emission time (Tini = min t), so this shifts "
                       "the emission relative to the RF clock (+ = laser later).",
    "L0AF_phase_offset_deg": "added to the pinned L0AF_phase reference (theta0_deg).",
    "hammersley_burnin": "distgen random:burnin (Hammersley start index; -1 = distgen default "
                         "= max prime used, as in the golden run).",
    "drive_fraction": "convenience: sets t_dist weights d2 = f, d1 = 1 - f.",
    "sigma_t_ps": "convenience: sets sigma_t of both laser pulses.",
}

CODE_VERSION = "injector-v1"   # bump if the physics of injector_beam changes (cache key)


# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------

def resolve_settings(settings: dict | None = None) -> dict:
    """Golden settings overridden by `settings`, with convenience keys expanded into the
    native keys. Unknown keys raise (a typo would otherwise silently run the golden point)."""
    out = dict(GOLDEN_SETTINGS)
    for key, val in (settings or {}).items():
        if key == "drive_fraction":
            out["distgen:t_dist:weights:d2"] = float(val)
            out["distgen:t_dist:weights:d1"] = 1.0 - float(val)
        elif key == "sigma_t_ps":
            out["distgen:t_dist:dists:d1:sigma_t:value"] = float(val)
            out["distgen:t_dist:dists:d2:sigma_t:value"] = float(val)
        elif key in GOLDEN_SETTINGS:
            out[key] = val
        else:
            raise KeyError(f"unknown injector setting {key!r}; known: "
                           f"{sorted(GOLDEN_SETTINGS) + ['drive_fraction', 'sigma_t_ps']}")
    out["hammersley_burnin"] = int(out["hammersley_burnin"])
    return out


def golden_value(key: str) -> float:
    """Golden value of a native or convenience key (for relative perturbations)."""
    if key == "drive_fraction":
        return GOLDEN_SETTINGS["distgen:t_dist:weights:d2"]
    if key == "sigma_t_ps":
        return GOLDEN_SETTINGS["distgen:t_dist:dists:d1:sigma_t:value"]
    return GOLDEN_SETTINGS[key]


def split_settings(resolved: dict) -> tuple[dict, dict]:
    """(impact_settings, distgen_settings) for a resolved settings dict.

    impact: lume-impact keys set with I[key] = val. distgen: keys WITHOUT the 'distgen:'
    prefix, set with G[key] = val; the common-mode laser timing is folded into both
    avg_t here (d1's golden avg_t is 0 ps). L0AF and the Hammersley burnin are handled
    separately (pinned reference / generator input)."""
    impact = {k: float(v) for k, v in resolved.items()
              if ":" in k and not k.startswith("distgen:")}
    dg = {k[len("distgen:"):]: float(v) for k, v in resolved.items() if k.startswith("distgen:")}
    shift = float(resolved["laser_timing_ps"])
    dg["t_dist:dists:d1:avg_t:value"] = 0.0 + shift
    dg["t_dist:dists:d2:avg_t:value"] = dg["t_dist:dists:d2:avg_t:value"] + shift
    return impact, dg


def _grid(grid):
    """Space-charge grid as (Nx, Ny, Nz): an int means isotropic (the campaign default); a
    3-sequence gives an anisotropic grid (e.g. finer Nz, since one grid spans both pulses)."""
    g = (int(grid),) * 3 if isinstance(grid, (int, np.integer)) else tuple(int(v) for v in grid)
    if len(g) != 3:
        raise ValueError(f"grid must be an int or (Nx, Ny, Nz), got {grid!r}")
    return g


def _grid_record(grid):
    """JSON form: int when isotropic (so existing cache keys are unchanged), else [Nx, Ny, Nz]."""
    g = _grid(grid)
    return g[0] if len(set(g)) == 1 else list(g)


def cache_key(resolved: dict, n_particles: int, grid, l0af_reference: dict) -> str:
    """Deterministic hash of everything that defines an injector beam."""
    blob = {
        "settings": {k: resolved[k] for k in sorted(resolved)},
        "n_particles": int(n_particles),
        "grid": _grid_record(grid),
        "l0af_reference": {k: float(l0af_reference[k]) for k in sorted(l0af_reference)},
        "code_version": CODE_VERSION,
    }
    text = json.dumps(blob, sort_keys=True, default=float)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------
# Workdir
# ---------------------------------------------------------------------------------------

def default_template_dir() -> Path:
    """FACET2-S2E/impact (read only)."""
    from ..datagen.paths import facet2_root   # noqa: PLC0415 (imports FACET2_S2E if no env var)

    return facet2_root() / "impact"


def prepare_workdir(workdir, template_dir=None) -> Path:
    """Create <workdir>/inputs/ with private copies of the Impact-T + distgen templates.
    Returns the inputs dir. Existing copies are kept (resume)."""
    template_dir = Path(template_dir) if template_dir else default_template_dir()
    inputs = Path(workdir) / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_FILES:
        dst = inputs / name
        if not dst.exists():
            shutil.copy2(template_dir / name, dst)
    return inputs


def _impact_command(command=None) -> str:
    """ImpactTexe of the running env (as runImpact does), unless given."""
    if command:
        return command
    exe = Path(sys.executable).parent / "ImpactTexe"
    return str(exe) if exe.exists() else (shutil.which("ImpactTexe") or "ImpactTexe")


# ---------------------------------------------------------------------------------------
# Drive/witness tagging
# ---------------------------------------------------------------------------------------

def tag_components(t, resolved: dict) -> np.ndarray:
    """Laser-pulse membership of generated particles from their emission time t [s].

    distgen's superposition does not report which component a particle came from, so the
    particle is assigned to the component with the larger weighted density at its t
    (w_k N(t; avg_t_k, sigma_t_k), the Bayes rule for the mixture). With the golden
    9 ps / 1.17 ps pulses the components are 7.7 sigma apart, so the expected number of
    particles with an ambiguous origin is ~1e-4 per particle (counted in injector_info
    as n_ambiguous). Returns is_drive (bool array)."""
    _, dg = split_settings(resolved)
    t_ps = np.asarray(t) * 1e12
    dens = {}
    for c in (WITNESS_COMPONENT, DRIVE_COMPONENT):
        mu = dg[f"t_dist:dists:{c}:avg_t:value"]
        sig = dg[f"t_dist:dists:{c}:sigma_t:value"]
        w = dg[f"t_dist:weights:{c}"]
        dens[c] = w * np.exp(-0.5 * ((t_ps - mu) / sig) ** 2) / sig
    return dens[DRIVE_COMPONENT] > dens[WITNESS_COMPONENT]


def ambiguous_count(t, resolved: dict, threshold: float = 1e-3) -> int:
    """Particles whose component posterior is within [threshold, 1 - threshold]."""
    _, dg = split_settings(resolved)
    t_ps = np.asarray(t) * 1e12
    d = []
    for c in (WITNESS_COMPONENT, DRIVE_COMPONENT):
        mu = dg[f"t_dist:dists:{c}:avg_t:value"]
        sig = dg[f"t_dist:dists:{c}:sigma_t:value"]
        d.append(dg[f"t_dist:weights:{c}"] * np.exp(-0.5 * ((t_ps - mu) / sig) ** 2) / sig)
    post = d[1] / np.maximum(d[0] + d[1], 1e-300)
    return int(np.sum((post > threshold) & (post < 1 - threshold)))


def apply_weight_tags(P, is_drive):
    """Golden weight convention in place: drive 1.001 q, witness 0.999 q, q = the uniform
    Impact-T macrocharge (nudgeMacroparticleWeights uses P.weight[0])."""
    q = float(P.weight[0])
    P.weight = np.where(is_drive, (1 + WEIGHT_NUDGE) * q, (1 - WEIGHT_NUDGE) * q)
    return P


def check_tagging(P) -> None:
    """Invariant every consumer relies on: exactly two unique weights; drive (higher weight)
    strictly greater than witness."""
    w = np.unique(P.weight)
    if len(w) != 2:
        raise ValueError(f"expected exactly 2 unique weights (drive/witness), got {len(w)}")
    if not w[1] > w[0]:
        raise ValueError("drive weight must be strictly greater than witness weight")


def split_drive_witness(P):
    """(drive, witness) ParticleGroups by weight: drive = higher weight."""
    check_tagging(P)
    w = np.unique(P.weight)
    return P[P.weight == w[1]], P[P.weight == w[0]]


# ---------------------------------------------------------------------------------------
# L0AF reference (pinned)
# ---------------------------------------------------------------------------------------

def _make_impact(inputs: Path, impact_settings: dict, grid, command=None):
    from impact import Impact  # noqa: PLC0415

    I = Impact.from_yaml(str(inputs / IMPACT_YAML))
    I["header:Nx"], I["header:Ny"], I["header:Nz"] = _grid(grid)
    I["stop_1:s"] = STOP_S_M
    for key, val in impact_settings.items():
        I[key] = val
    I.command = _impact_command(command)
    I.numprocs = 1
    return I


def pin_l0af_reference(cache_path, workdir, template_dir=None, command=None) -> dict:
    """Autophase L0AF ONCE at the golden gun settings and cache the result (json).

    Same procedure as runImpact: single particle tracked to s=0.9 m for the gun energy E,
    then autophase_and_scale(L0AF_phase, L0AF_scale) to E + L0AF_E_GAIN_EV starting from
    (29 deg, 30 MV/m), and the phase offset of 0 (= max-energy phase). Golden ImpactT.yaml
    in Source.zip recorded 359.86189 deg / 32599504.36."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    import pmd_beamphysics  # noqa: PLC0415
    from impact.autophase import autophase_and_scale  # noqa: PLC0415

    inputs = prepare_workdir(workdir, template_dir)
    impact_settings, _ = split_settings(resolve_settings())
    I = _make_impact(inputs, impact_settings, grid=8, command=command)
    I.workdir = str(workdir)
    P0 = pmd_beamphysics.single_particle(pz=1e-15, z=1e-15)
    gun_energy = copy.deepcopy(I).track(P0, s=0.9)["mean_energy"]
    I["L0AF_scale"]["rf_field_scale"] = 30e6
    I["L0AF_phase"]["theta0_deg"] = 29
    phase, scale = autophase_and_scale(
        I, phase_ele_name="L0AF_phase", scale_ele_name="L0AF_scale",
        target=gun_energy + L0AF_E_GAIN_EV, scale_range=(10e6, 100e6),
        initial_particles=P0, verbose=False)
    ref = {"L0AF_phase:theta0_deg": float(phase), "L0AF_scale:rf_field_scale": float(scale),
           "gun_energy_eV": float(gun_energy)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(ref, f, indent=2)
    return ref


# ---------------------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------------------

def _generate(inputs: Path, resolved: dict, n_particles: int):
    """distgen beam at the cathode with the resolved settings (Hammersley sampling)."""
    from distgen import Generator  # noqa: PLC0415

    with open(inputs / DISTGEN_FILE) as f:
        dg_input = yaml.safe_load(f)
    # Expose the Hammersley start index: 'random_type: hammersley' -> 'random: {type, burnin}'
    dg_input.pop("random_type", None)
    dg_input["random"] = {"type": "hammersley", "burnin": int(resolved["hammersley_burnin"])}
    G = Generator(dg_input)
    G.verbose = False
    _, dg = split_settings(resolved)
    G["n_particle"] = int(n_particles)
    for key, val in dg.items():
        G[key] = val
    G.run()
    return G.particles


def injector_beam(settings: dict | None, workdir, n_particles: int, grid=32,
                  l0af_reference: dict | None = None, template_dir=None, command=None):
    """Run distgen + Impact-T to L0AFEND for `settings` (see GOLDEN_SETTINGS / DERIVED_KEYS).

    Returns (P, info): P is the L0AFEND ParticleGroup in the legacy Bmad-input form
    (drift_to_z onto the L0AFEND plane, z = 0, t centred by the UNWEIGHTED mean exactly as
    runImpact/modifyAndSaveInputBeam), drive/witness weight-tagged. info holds the arrival
    time BEFORE centring (mean_t_l0afend_s, unweighted, at the fixed L0AFEND plane), per-bunch
    counts, the resolved settings, and the cache key.

    Results are cached in <workdir>/beam_L0AFEND.h5 + injector_info.json; a later call with
    the same cache key just loads them. l0af_reference comes from pin_l0af_reference."""
    from pmd_beamphysics import ParticleGroup  # noqa: PLC0415
    from impact.parsers import parse_impact_particles  # noqa: PLC0415

    if l0af_reference is None:
        raise ValueError("pass l0af_reference=pin_l0af_reference(...) (pinned once at golden)")
    workdir = Path(workdir)
    resolved = resolve_settings(settings)
    key = cache_key(resolved, n_particles, grid, l0af_reference)
    beam_path, info_path = workdir / "beam_L0AFEND.h5", workdir / "injector_info.json"
    if beam_path.exists() and info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if info.get("cache_key") == key:
            return ParticleGroup(str(beam_path)), info

    t_start = time.perf_counter()
    inputs = prepare_workdir(workdir, template_dir)

    # Cathode distribution + generation-time tags
    P_cath = _generate(inputs, resolved, n_particles)
    is_drive_cath = tag_components(P_cath.t, resolved)

    # Impact-T with the pinned L0AF reference (+ offset)
    impact_settings, _ = split_settings(resolved)
    I = _make_impact(inputs, impact_settings, grid, command)
    I["L0AF_phase"]["theta0_deg"] = (l0af_reference["L0AF_phase:theta0_deg"]
                                     + float(resolved["L0AF_phase_offset_deg"]))
    I["L0AF_scale"]["rf_field_scale"] = l0af_reference["L0AF_scale:rf_field_scale"]
    I.initial_particles = P_cath
    I.workdir = str(workdir)
    I.verbose = False
    I.run()
    if I.output.get("run_info", {}).get("error"):
        raise RuntimeError(f"Impact-T failed in {I.path}")

    # Carry the tags through Impact-T by particle id (= 1-based row of partcl.data, i.e. of
    # P_cath). lume-impact drops the id column when it builds the ParticleGroup, so re-read
    # the raw write_beam file; rows are in the same order as I.particles['L0AFEND'].
    fname = next(e["filename"] for e in I.input["lattice"] if e.get("name") == "L0AFEND")
    raw = parse_impact_particles(os.path.join(I.path, fname))
    ids = raw["id"].astype(np.int64)
    P = I.particles["L0AFEND"].copy()
    if len(ids) != len(P) or not np.allclose(raw["x"], P.x):
        raise RuntimeError("raw L0AFEND rows do not match lume-impact particles")
    if len(np.unique(ids)) != len(ids) or ids.min() < 1 or ids.max() > n_particles:
        raise RuntimeError("Impact-T particle ids are not a subset of 1..N")
    is_drive = is_drive_cath[ids - 1]
    apply_weight_tags(P, is_drive)
    check_tagging(P)

    # Arrival time at the fixed L0AFEND plane, BEFORE centring (unweighted, as the centring)
    P.drift_to_z(L0AFEND_Z_M)
    mean_t = float(np.mean(P.t))
    drive_pg, witness_pg = split_drive_witness(P)
    info = {
        "cache_key": key,
        "code_version": CODE_VERSION,
        "settings": resolved,
        "l0af_reference": l0af_reference,
        "l0af_phase_used_deg": float(I["L0AF_phase"]["theta0_deg"]),
        "n_particles": int(n_particles),
        "grid": _grid_record(grid),
        "n_cathode_drive": int(is_drive_cath.sum()),
        "n_cathode_witness": int((~is_drive_cath).sum()),
        "n_ambiguous": ambiguous_count(P_cath.t, resolved),
        "n_l0afend": int(len(P)),
        "n_drive": int(len(drive_pg)),
        "n_witness": int(len(witness_pg)),
        "mean_t_l0afend_s": mean_t,
        "mean_t_weighted_l0afend_s": float(P["mean_t"]),
        "mean_t_drive_s": float(np.mean(drive_pg.t)),
        "mean_t_witness_s": float(np.mean(witness_pg.t)),
        "mean_energy_eV": float(P["mean_energy"]),
        "charge_C": float(P.charge),
        "impact_run_s": float(I.output["run_info"].get("run_time", np.nan)),
        "wall_s": None,
    }

    # Legacy Bmad-input form (runImpact): z = 0, t centred by the unweighted mean.
    P.z[:] = 0
    P.t[:] -= mean_t
    info["wall_s"] = time.perf_counter() - t_start
    P.write(str(beam_path))
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=float)
    shutil.rmtree(I.path, ignore_errors=True)   # lume temp dir (fort files); beam is saved
    return P, info
