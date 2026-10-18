"""Monte Carlo of per-shot machine jitter -> XTCAV screen measurement errors (2026-10-18).

Per-shot independent Gaussian knob jitter (Yocky / Zack jitter sheet rows mapped to surrogate knobs) through the
env's screen pipeline (flow at XTCAVF -> Cheetah TCAV +/-V -> DTOTR -> deployed split / validity features), with the
env's screen settings (configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml `screen:` block) but GOLDEN optics,
nominal TCAV voltage (22 MV) and no quad jitter (hidden per-episode states, not per-shot jitter).

Per shot the plant's screen_features is run on ONE cloud for both signs (paired +V/-V). The deployed analysis uses
different shots for +V and -V and median-aggregates per sign; so we also record per-sign single-image quantities
(band offset x_d - x_w, drive variance, drive charge fraction, crop-edge fraction) via a patched copy of
xtcav_segment.features.screen_features (adds the per-sign values; src/ untouched), and build steps of 30 +V and 30 -V
images from DIFFERENT shots, exactly as online.features.step_features does.

  PYTHONPATH=src python scripts/jitter_mc_2026-10-18.py mc      # surrogate MC (map grid + calibration lines)
  PYTHONPATH=src python scripts/jitter_mc_2026-10-18.py calib   # fit to the real per-image scatter
  PYTHONPATH=src python scripts/jitter_mc_2026-10-18.py map     # env noise maps (needs calib)
  PYTHONPATH=src python scripts/jitter_mc_2026-10-18.py bmad    # Bmad cross-check (spawns bmad-qpad-dev server)
  PYTHONPATH=src python scripts/jitter_mc_2026-10-18.py figs
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/jitter_mc_2026-10-18"
CKPT = ROOT / "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
CFG = ROOT / "configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml"
REPLAY = ROOT / "results/online_replay_v6_median_cut_2026-10-15"
SCANS = {"TEST_15668": -23.7, "E300_15671": -26.0, "E300_15673": -20.0}
M_IMG = 30                 # images per sign per controller step (env shots_per_step)
FLOOR = 60e-6              # deployed |spacing| floor
N_PART = 1024              # env n_particles

# ----------------------------------------------------------------------------------------------------------------
# jitter scenarios: per-shot rms, physical knob units. Amplitudes: % x section gain (SECTION_GAIN_EV); gun amp and
# charge relative to the golden value. L1 = common-mode of L1A / L1B: uncorrelated sqrt(a^2+b^2)/2, correlated (a+b)/2.
# Not mapped (no knob): source position, L0A amplitude, L3 phase, magnet dB/B (LH, DL10, BC11, BC14, BC20), vibration.
GAIN = {"L0BEnergyOffset": 59.5e6, "L1EnergyOffset": 210e6, "L2EnergyOffset": 4165e6, "L3EnergyOffset": 5500e6}


def _l1(a, b, corr):
    return 0.5 * (a + b) if corr else 0.5 * float(np.hypot(a, b))


def sheet_row(tag, gunf, q):
    """Sheet columns -> knob sigmas. tag in TDR / 2024u / 2024c / 2025u / 2025c."""
    base = tag[:4] if tag != "TDR" else "TDR"
    corr = tag.endswith("c")
    if base == "TDR":
        r = dict(Q=1.0, laser_fs=200, gun_p=0.1, gun_a=0.25, l0a_p=0.1, l0b_p=0.1, l0b_a=0.25, l1a_p=0.1, l1a_a=0.1,
                 l1b_p=0.1, l1b_a=0.1, l2_p=0.25, l2_a=0.25, l3_a=0.25)
    else:
        r = dict(Q=2.3, laser_fs=50, gun_p=0.15, gun_a=0.25, l0a_p=0.1, l0b_p=0.1, l0b_a=0.5, l1a_p=0.7, l1a_a=0.6,
                 l1b_p=0.5, l1b_a=0.7, l2_p=0.4, l2_a=0.3, l3_a=0.3)
        if base == "2025":     # Zack 10/2024 (1000 shots): Q, gun phase, gun amp; L1A/B amplitudes
            r.update(Q=2.74, gun_p=0.5, gun_a=0.1, l1a_a=0.25, l1b_a=0.25)
    return {"L1PhaseSet": _l1(r["l1a_p"], r["l1b_p"], corr),
            "L1EnergyOffset": 0.01 * _l1(r["l1a_a"], r["l1b_a"], corr) * GAIN["L1EnergyOffset"],
            "L2PhaseSet": r["l2_p"], "L2EnergyOffset": 0.01 * r["l2_a"] * GAIN["L2EnergyOffset"],
            "L3EnergyOffset": 0.01 * r["l3_a"] * GAIN["L3EnergyOffset"],
            "L0BPhaseSet": r["l0b_p"], "L0BEnergyOffset": 0.01 * r["l0b_a"] * GAIN["L0BEnergyOffset"],
            "laser_timing_ps": 1e-3 * r["laser_fs"], "GUNF:rf_field_scale": 0.01 * r["gun_a"] * gunf,
            "GUNF:theta0_deg": r["gun_p"], "distgen:total_charge:value": 0.01 * r["Q"] * q,
            "L0AF_phase_offset_deg": r["l0a_p"]}


# ----------------------------------------------------------------------------------------------------------------
def _patch_screen_features():
    """A copy of xtcav_segment.features.screen_features that also stashes per-sign single-image quantities in
    LAST (the plant only forwards the standard keys)."""
    import xtcav_segment.features as F
    src = inspect.getsource(F.screen_features)
    a = "            edge_frac = torch.maximum(edge(xp, mp), edge(xm, mm))"
    assert a in src
    src = src.replace(a, "            edge_p, edge_m = edge(xp, mp), edge(xm, mm)\n"
                         "            edge_frac = torch.maximum(edge_p, edge_m)")
    b = '    out = {"bunch_spacing": spacing,'
    assert b in src
    src = src.replace(b, '    LAST.clear()\n'
                         '    LAST.update(off_p=_wmean(xp, wdp) - _wmean(xp, wwp), off_m=_wmean(xm, wdm) - _wmean(xm, wwm),\n'
                         '                var_p=_wvar(xp, wdp), var_m=_wvar(xm, wdm),\n'
                         '                frac_p=wdp.sum(-1) / qp.sum(-1).clamp_min(1e-30),\n'
                         '                frac_m=wdm.sum(-1) / qm.sum(-1).clamp_min(1e-30), peaks_ok=peaks_ok.float())\n'
                         '    if roi is not None:\n'
                         '        LAST.update(edge_p=edge_p, edge_m=edge_m)\n' + b)
    ns = dict(vars(F))
    ns["LAST"] = {}
    exec(compile(src, "<patched screen_features>", "exec"), ns)
    F.screen_features = ns["screen_features"]
    return ns["LAST"]


class Pipeline:
    def __init__(self, device="cuda:0", n_particles=N_PART, seed=0):
        import yaml
        from twobunch_s2e_rl.artifacts import flow_knob_space
        from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
        from twobunch_s2e_rl.rl._train_utils import flow_norm_json
        from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
        self.LAST = _patch_screen_features()
        from twobunch_s2e_rl.rl.plants import xtcav_screen_plant
        de = yaml.safe_load(CFG.read_text())["params"]["diff_env"]
        sc = de["screen"]
        self.device = device
        flow = TwoBunchFlow.load_from_checkpoint(str(CKPT), map_location=device).to(device).eval()
        for p in flow.parameters():
            p.requires_grad_(False)
        self.flow = flow
        self.ks = flow_knob_space(flow)
        self.keys = list(self.ks.keys)
        self.base = dict(resolve_sweep_set(de["baseline_set"])[3])
        self.base.update(de.get("baseline_override", {}))
        self.lo = flow.knob_low.cpu().numpy().astype(float)
        self.hi = flow.knob_high.cpu().numpy().astype(float)
        self.screen_cfg = {k: sc.get(k) for k in ("split", "threshold", "row_correct", "roi", "arrival_jitter_s",
                                                   "orientation", "deployed_floor_m", "edge_frac_max", "edge_band_m",
                                                   "extra_resolution_m", "pm_energy_axes", "tdc_voltage_V")}
        self.plant = xtcav_screen_plant(
            flow, n_particles, spec_path=str(ROOT / sc.get("spec")), norm_json=flow_norm_json(CKPT, ROOT / de["campaign_h5"]),
            device=device, tdc_voltage_V=float(sc["tdc_voltage_V"]), split=sc["split"],
            resolution_m=float(sc.get("extra_resolution_m", 0.0)), threshold=sc["threshold"],
            row_correct=bool(sc["row_correct"]), quad_jitter=None, optics_choices=None, tdc_voltage_range=None,
            roi=sc.get("roi"), arrival_jitter_s=float(sc.get("arrival_jitter_s", 0.0)), orientation=sc["orientation"],
            deployed_floor_m=float(sc["deployed_floor_m"]), seed=seed, edge_frac_max=float(sc["edge_frac_max"]),
            edge_band_m=float(sc["edge_band_m"]), pm_energy_axes=bool(sc["pm_energy_axes"]))
        self.S = float(self.plant.streak)
        self.golden = np.array([float(self.base[k]) for k in self.keys])

    def phys_batch(self, L1, L2, sig: dict, n, rng, clip_sigma=None):
        P = np.tile(self.golden, (n, 1))
        P[:, self.keys.index("L1PhaseSet")] = L1
        P[:, self.keys.index("L2PhaseSet")] = L2
        for k, s in sig.items():
            if s > 0:
                d = rng.standard_normal(n)
                if clip_sigma:
                    d = np.clip(d, -clip_sigma, clip_sigma)
                P[:, self.keys.index(k)] += s * d
        return P

    @torch.no_grad()
    def shots(self, phys: np.ndarray, chunk=512):
        """phys (n, d) -> dict of per-shot numpy arrays (SI units)."""
        res = {}
        for i0 in range(0, len(phys), chunk):
            kn = torch.as_tensor((phys[i0:i0 + chunk] - self.lo) / (self.hi - self.lo), dtype=torch.float32,
                                 device=self.device)
            obs = self.plant(kn)
            ex = dict(self.LAST)
            row = {"spacing_pair": obs["bunch_spacing"], "sigz_pair": obs["drive_sigma_z"],
                   "frac_pair": obs["drive_charge_fraction"], "valid_pair": obs["screen_valid_deployed"],
                   "valid_truth_pair": obs["screen_valid"],
                   "spacing_cloud": obs["bunch_spacing_cloud"], "sigz_cloud": obs["drive_sigma_z_cloud"],
                   "frac_cloud": obs["drive_charge_fraction_cloud"], "T_drive": obs["T_drive"],
                   "T_witness": obs["T_witness"]}
            row["sp_p"] = ex["off_p"] / self.S                      # single-image signed spacing, +V
            row["sp_m"] = -ex["off_m"] / self.S                     # -V (streak -S)
            for k in ("var_p", "var_m", "frac_p", "frac_m", "peaks_ok", "edge_p", "edge_m"):
                row[k] = ex[k]
            for k, v in row.items():
                res.setdefault(k, []).append(v.detach().double().cpu().numpy())
        return {k: np.concatenate(v) for k, v in res.items()}


# ----------------------------------------------------------------------------------------------------------------
def rsig(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return float("nan")
    q75, q25 = np.percentile(v, [75, 25])
    return float((q75 - q25) / 1.349)


def steps_from_shots(sh, S, n_steps, rng, m=M_IMG, edge_max=0.01):
    """Deployed step features from per-shot single-image values: m +V shots and m DIFFERENT -V shots, per-sign
    medians, half difference (online.features.step_features)."""
    n = len(sh["sp_p"])
    out = {k: np.empty(n_steps) for k in ("spacing", "sigz", "frac", "valid", "spacing_pairmed", "spacing_cloud")}
    for s in range(n_steps):
        idx = rng.choice(n, 2 * m, replace=False)
        ip, im = idx[:m], idx[m:]
        sp = 0.5 * (np.median(sh["sp_p"][ip]) + np.median(sh["sp_m"][im]))
        out["spacing"][s] = sp
        out["sigz"][s] = np.sqrt(0.5 * (np.median(sh["var_p"][ip]) + np.median(sh["var_m"][im]))) / abs(S)
        out["frac"][s] = 0.5 * (np.median(sh["frac_p"][ip]) + np.median(sh["frac_m"][im]))
        edge = max(np.median(sh["edge_p"][ip]), np.median(sh["edge_m"][im]))
        pk = np.median(sh["peaks_ok"][ip]) > 0.5               # the per-step cut (from the +V images) is found
        out["valid"][s] = float(abs(sp) > FLOOR and edge < edge_max and pk)
        out["spacing_pairmed"][s] = np.median(sh["spacing_pair"][ip])     # plant-style: paired shots, median
        out["spacing_cloud"][s] = 0.5 * (np.mean(sh["spacing_cloud"][ip]) + np.mean(sh["spacing_cloud"][im]))
    return out


def point_stats(sh, sh0, S, rng, n_steps=2000):
    """Per operating point: per-image + per-step statistics of the jittered pool `sh` vs the zero-jitter pool `sh0`."""
    st = steps_from_shots(sh, S, n_steps, rng)
    st0 = steps_from_shots(sh0, S, n_steps, rng)
    um = 1e6
    single = lambda d: 0.5 * (rsig(d["sp_p"]) + rsig(d["sp_m"]))
    r = {
        "noiseless_spacing_um": um * float(np.median(st0["spacing"])),
        "noiseless_sigz_um": um * float(np.median(st0["sigz"])),
        "noiseless_frac": float(np.median(st0["frac"])),
        "noiseless_valid_step": float(np.mean(st0["valid"])),
        "cloud_spacing_um": um * float(np.mean(sh0["spacing_cloud"])),
        "cloud_sigz_um": um * float(np.mean(sh0["sigz_cloud"])),
        # per image
        "img_spacing_rsig_um": um * single(sh),                    # per-sign single-image, as the real replay
        "img_spacing_rsig_um_zero": um * single(sh0),              # particle sampling (zero jitter)
        "img_spacing_std_um": um * 0.5 * (np.std(sh["sp_p"]) + np.std(sh["sp_m"])),
        "img_pair_spacing_rsig_um": um * rsig(sh["spacing_pair"]),
        "img_pair_spacing_rsig_um_zero": um * rsig(sh0["spacing_pair"]),
        "img_cloud_spacing_rsig_um": um * rsig(sh["spacing_cloud"]),
        "img_cloud_spacing_rsig_um_zero": um * rsig(sh0["spacing_cloud"]),
        "img_cloud_sigz_rel_rsig": rsig(sh["sigz_cloud"]) / float(np.median(sh0["sigz_cloud"])),
        "img_sigz_rel_rsig": rsig(np.sqrt(0.5 * (sh["var_p"] + sh["var_m"])) / abs(S)) / float(np.median(st0["sigz"])),
        "img_sigz_rel_rsig_zero": rsig(np.sqrt(0.5 * (sh0["var_p"] + sh0["var_m"])) / abs(S)) / float(np.median(st0["sigz"])),
        "img_frac_rsig": rsig(sh["frac_pair"]),
        "img_valid_rate": float(np.mean(sh["valid_pair"])),
        "img_valid_rate_zero": float(np.mean(sh0["valid_pair"])),
        # per step (30 + 30 images, deployed medians)
        "step_spacing_std_um": um * float(np.std(st["spacing"])),
        "step_spacing_std_um_zero": um * float(np.std(st0["spacing"])),
        "step_spacing_bias_um": um * float(np.mean(st["spacing"]) - np.mean(st0["spacing"])),
        "step_spacing_p95abs_err_um": um * float(np.percentile(np.abs(st["spacing"] - np.mean(st0["spacing"])), 95)),
        "step_pairmed_std_um": um * float(np.std(st["spacing_pairmed"])),
        "step_pairmed_bias_um": um * float(np.mean(st["spacing_pairmed"]) - np.mean(st0["spacing_pairmed"])),
        "step_cloud_bias_um": um * float(np.mean(st["spacing_cloud"]) - np.mean(st0["spacing_cloud"])),
        "step_sigz_rel_std": float(np.std(st["sigz"]) / np.mean(st0["sigz"])),
        "step_sigz_rel_std_zero": float(np.std(st0["sigz"]) / np.mean(st0["sigz"])),
        "step_sigz_bias_um": um * float(np.mean(st["sigz"]) - np.mean(st0["sigz"])),
        "step_frac_std": float(np.std(st["frac"])),
        "step_valid_rate": float(np.mean(st["valid"])),
        "step_valid_rate_zero": float(np.mean(st0["valid"])),
        "T_drive_mean": float(np.mean(sh["T_drive"])), "T_witness_mean": float(np.mean(sh["T_witness"])),
    }
    return r, st, st0


def cmd_mc(n_shot=512, n_zero=512):
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline()
    gunf, q = float(pipe.base["GUNF:rf_field_scale"]), float(pipe.base["distgen:total_charge:value"])
    scen = {t: sheet_row(t, gunf, q) for t in ("TDR", "2024u", "2024c", "2025u")}
    s24 = scen["2024u"]
    # calibration components (on the scan lines): unit 0.5 deg L1 / L2 phase only; everything else at 2024
    comp = {"L1only": {"L1PhaseSet": 0.5}, "L2only": {"L2PhaseSet": 0.5},
            "rest2024": {k: v for k, v in s24.items() if k not in ("L1PhaseSet", "L2PhaseSet")}}
    rng = np.random.default_rng(20261018)
    t0 = time.time()
    meta = {"scenarios": scen, "components": comp, "S_streak": pipe.S, "screen": pipe.screen_cfg,
            "n_shot": n_shot, "n_zero": n_zero, "n_particles": N_PART, "golden": dict(zip(pipe.keys, pipe.golden.tolist())),
            "knob_low": dict(zip(pipe.keys, pipe.lo.tolist())), "knob_high": dict(zip(pipe.keys, pipe.hi.tolist()))}
    # out-of-box check: fraction of jittered shots outside the flow's training box, per knob
    oob = {}
    for t, sg in scen.items():
        P = pipe.phys_batch(-22.5, -37.0, sg, 20000, np.random.default_rng(1))
        oob[t] = {k: float(np.mean((P[:, pipe.keys.index(k)] < pipe.lo[pipe.keys.index(k)])
                                   | (P[:, pipe.keys.index(k)] > pipe.hi[pipe.keys.index(k)]))) for k in sg}
    meta["out_of_box_frac"] = oob
    (OUT / "mc_meta.json").write_text(json.dumps(meta, indent=1))

    def run_point(L1, L2, sets):
        sh0 = pipe.shots(pipe.phys_batch(L1, L2, {}, n_zero, rng))
        rows, pools = {}, {"zero": sh0}
        for name, sg in sets.items():
            sh = pipe.shots(pipe.phys_batch(L1, L2, sg, n_shot, rng))
            rows[name], st, _ = point_stats(sh, sh0, pipe.S, rng)
            pools[name] = sh
        return rows, pools

    # 1) env grid (5 x 9) for the map scenarios
    L1g = np.array([-26.5, -24.5, -22.5, -20.5, -18.5])
    L2g = np.arange(-41.0, -32.9, 1.0)
    grid = []
    keep = {}
    for L1 in L1g:
        for L2 in L2g:
            rows, pools = run_point(float(L1), float(L2), scen)
            grid.append({"L1": float(L1), "L2": float(L2), **{k: v for k, v in rows.items()}})
            if (L1, L2) in [(-22.5, -38.0), (-24.5, -36.0), (-20.5, -36.0), (-22.5, -35.0)]:
                keep[f"{L1}_{L2}"] = pools
            print(f"grid L1 {L1:6.1f} L2 {L2:6.1f} | noiseless {rows['2024u']['noiseless_spacing_um']:6.1f} um "
                  f"| img rsig 2024u {rows['2024u']['img_spacing_rsig_um']:5.1f} (zero {rows['2024u']['img_spacing_rsig_um_zero']:4.1f})"
                  f" step std {rows['2024u']['step_spacing_std_um']:4.1f} bias {rows['2024u']['step_spacing_bias_um']:+5.1f}"
                  f" valid {rows['2024u']['step_valid_rate']:.2f}/{rows['2024u']['noiseless_valid_step']:.2f}"
                  f" | {time.time() - t0:5.0f}s", flush=True)
    (OUT / "mc_grid.json").write_text(json.dumps(grid, indent=1))
    torch.save(keep, OUT / "mc_pools_examples.pt")
    # 2) calibration lines at the scans' L1
    lines = []
    for scan, L1 in SCANS.items():
        for L2 in np.arange(-41.5, -31.9, 0.5):
            rows, _ = run_point(float(L1), float(L2), {**comp, "2024u": s24})
            lines.append({"scan": scan, "L1": float(L1), "L2": float(L2), **rows})
            print(f"line {scan} L1 {L1} L2 {L2:6.1f} | noiseless {rows['L1only']['noiseless_spacing_um']:6.1f} | img rsig "
                  + " ".join(f"{k} {v['img_spacing_rsig_um']:5.1f}" for k, v in rows.items())
                  + f" zero {rows['L1only']['img_spacing_rsig_um_zero']:4.1f} | {time.time() - t0:5.0f}s", flush=True)
    (OUT / "mc_lines.json").write_text(json.dumps(lines, indent=1))


def cmd_probe():
    """Timing + out-of-box behaviour of the flow along charge / gun phase / laser timing."""
    pipe = Pipeline()
    rng = np.random.default_rng(0)
    t = time.time()
    sh = pipe.shots(pipe.phys_batch(-22.5, -37.0, {}, 512, rng))
    print("512 shots", time.time() - t, "s; S", pipe.S, "spacing", 1e6 * np.median(sh["spacing_pair"]),
          "single rsig", 1e6 * rsig(sh["sp_p"]), 1e6 * rsig(sh["sp_m"]), "cloud", 1e6 * np.mean(sh["spacing_cloud"]))
    for k, vals in (("distgen:total_charge:value", np.linspace(1520, 1680, 9)), ("GUNF:theta0_deg", np.linspace(-62, -59.4, 9)),
                    ("laser_timing_ps", np.linspace(-0.6, 0.6, 9)), ("L0BPhaseSet", np.linspace(-15.9, -14.1, 7)),
                    ("L0AF_phase_offset_deg", np.linspace(-0.45, 0.45, 7)), ("GUNF:rf_field_scale", 48594238.1 * np.linspace(0.99, 1.01, 9))):
        out = []
        for v in vals:
            P = pipe.phys_batch(-22.5, -37.0, {}, 256, rng)
            P[:, pipe.keys.index(k)] = v
            s = pipe.shots(P)
            out.append((float(v), 1e6 * float(np.mean(s["spacing_cloud"])), 1e6 * float(np.median(s["spacing_pair"])),
                        1e6 * float(np.mean(s["sigz_cloud"]))))
        j = pipe.keys.index(k)
        print(k, "box", pipe.lo[j], pipe.hi[j])
        for o in out:
            print("   %.5g cloud %.1f screen %.1f sigz %.1f" % o)


# ----------------------------------------------------------------------------------------------------------------
BMAD_POINTS = [(-22.5, -38.0), (-24.5, -36.0), (-20.5, -36.0)]


def cmd_bmad(n_shot=48, n_seed=16, workers=96):
    """Linac-only 2024u jitter (injector knobs need Impact-T; golden injector beam) through Bmad to XTCAVF.
    Jittered shots share ONE Tao seed (common random numbers: differences = jitter); a zero-jitter set with
    n_seed different seeds measures the Tao (CSR/ISR) seed noise. The surrogate runs the IDENTICAL knob draws."""
    from twobunch_s2e_rl.datagen.sweep_params import INJECTOR_PARAMS, injector_golden, resolve_sweep_set
    from twobunch_s2e_rl.rl.closed_loop_bmad import COV_CFG, BmadServer, _stack
    from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
    from pmd_beamphysics import ParticleGroup
    n_shot, n_seed, workers = int(n_shot), int(n_seed), int(workers)
    out = OUT / "bmad"
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(device="cuda:1")
    gunf, q = float(pipe.base["GUNF:rf_field_scale"]), float(pipe.base["distgen:total_charge:value"])
    s24 = sheet_row("2024u", gunf, q)
    lin = [k for k in pipe.keys if k not in INJECTOR_PARAMS]
    sig = {k: v for k, v in s24.items() if k in lin}
    rng = np.random.default_rng(777)
    tasks, rows = [], []
    for ip, (L1, L2) in enumerate(BMAD_POINTS):
        P = pipe.phys_batch(L1, L2, sig, n_shot, rng)
        P0 = pipe.phys_batch(L1, L2, {}, n_seed, rng)
        for kind, PP in (("jit", P), ("zero", P0)):
            for i, ph in enumerate(PP):
                tid = len(tasks)
                seed = 4242 if kind == "jit" else 5000 + i
                tasks.append({"id": tid, "knobs": {k: float(ph[pipe.keys.index(k)]) for k in lin}, "beam": None,
                              "seed": seed, "out": str(out / f"beams/p{ip}_{kind}_{i:03d}")})
                rows.append({"point": ip, "L1": L1, "L2": L2, "kind": kind, "i": i, "seed": seed, "phys": ph.tolist()})
    (out / "beams").mkdir(exist_ok=True)
    golden = injector_golden(resolve_sweep_set("inj12")[0])
    t0 = time.time()
    server = BmadServer(ROOT / COV_CFG, 20000, workers, out / "scratch")
    try:
        paths = server.request({"cmd": "prepare", "points": [{"key": "golden", "injector": dict(golden)}]})
        for t in tasks:
            t["beam"] = paths["golden"]
        res = server.request({"cmd": "track", "tasks": tasks}, timeout=4 * 3600)
    finally:
        server.close()
    print(f"bmad done {time.time() - t0:.0f}s", flush=True)
    res = {r["id"]: r for r in res}
    for tid, r in enumerate(rows):
        rr = res[tid]
        r.update(ok=bool(rr.get("ok")), wall_s=rr.get("wall_s"), error=rr.get("error", "")[-300:])
        if not r["ok"]:
            continue
        Pg = ParticleGroup(rr["h5"])
        w = np.unique(Pg.weight)
        d = torch.as_tensor(_stack(Pg[Pg.weight == w[-1]]))[None].double()
        wt = torch.as_tensor(_stack(Pg[Pg.weight == w[0]]))[None].double()
        ib, pb = inter_bunch(d, wt), per_bunch(d)
        r.update(spacing_cloud=float(ib["bunch_spacing"][0]), sigz_cloud=float(pb["sigma_z"][0]),
                 T_drive=rr["n_live_drive"] / max(rr["n_in_drive"], 1),
                 T_witness=rr["n_live_witness"] / max(rr["n_in_witness"], 1))
    # surrogate at the identical knob vectors: cloud truth, mean over 8 particle draws of 4096
    with torch.no_grad():
        for r in rows:
            r["flow_spacing_cloud"], r["flow_sigz_cloud"] = [], []
        PH = np.array([r["phys"] for r in rows])
        kn = torch.as_tensor((PH - pipe.lo) / (pipe.hi - pipe.lo), dtype=torch.float32, device=pipe.device)
        acc_s, acc_z = [], []
        for _ in range(8):
            o = pipe.flow.observables(kn, n=4096)
            acc_s.append(o["bunch_spacing"].double().cpu().numpy())
            acc_z.append(o["drive_sigma_z"].double().cpu().numpy())
        for j, r in enumerate(rows):
            r["flow_spacing_cloud"] = float(np.mean([a[j] for a in acc_s]))
            r["flow_sigz_cloud"] = float(np.mean([a[j] for a in acc_z]))
            r["flow_spacing_cloud_draw_sd"] = float(np.std([a[j] for a in acc_s]))
    (out / "bmad_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    summ = []
    for ip, (L1, L2) in enumerate(BMAD_POINTS):
        J = [r for r in rows if r["point"] == ip and r["kind"] == "jit" and r["ok"]]
        Z = [r for r in rows if r["point"] == ip and r["kind"] == "zero" and r["ok"]]
        a = lambda L, k: np.array([r[k] for r in L])
        bs, fs = a(J, "spacing_cloud"), a(J, "flow_spacing_cloud")
        bz, fz = a(J, "sigz_cloud"), a(J, "flow_sigz_cloud")
        slope = float(np.polyfit(fs - fs.mean(), bs - bs.mean(), 1)[0]) if len(J) > 3 else float("nan")
        summ.append({"L1": L1, "L2": L2, "n_jit_ok": len(J), "n_zero_ok": len(Z),
                     "bmad_spacing_mean_um": 1e6 * bs.mean(), "flow_spacing_mean_um": 1e6 * fs.mean(),
                     "bmad_spacing_std_um": 1e6 * bs.std(ddof=1), "flow_spacing_std_um": 1e6 * fs.std(ddof=1),
                     "bmad_seed_spacing_std_um": 1e6 * a(Z, "spacing_cloud").std(ddof=1) if len(Z) > 1 else None,
                     "bmad_sigz_mean_um": 1e6 * bz.mean(), "flow_sigz_mean_um": 1e6 * fz.mean(),
                     "bmad_sigz_rel_std": bz.std(ddof=1) / bz.mean(), "flow_sigz_rel_std": fz.std(ddof=1) / fz.mean(),
                     "bmad_seed_sigz_rel_std": (a(Z, "sigz_cloud").std(ddof=1) / a(Z, "sigz_cloud").mean()) if len(Z) > 1 else None,
                     "per_shot_corr_spacing": float(np.corrcoef(bs, fs)[0, 1]),
                     "per_shot_corr_sigz": float(np.corrcoef(bz, fz)[0, 1]),
                     "slope_bmad_on_flow_spacing": slope,
                     "rms_resid_spacing_um": 1e6 * float(np.std((bs - bs.mean()) - (fs - fs.mean()), ddof=1))})
        print(json.dumps(summ[-1], default=float), flush=True)
    (out / "bmad_summary.json").write_text(json.dumps(summ, indent=1, default=float))


# ----------------------------------------------------------------------------------------------------------------
def real_steps():
    """Valid real steps (deployed validity) with a sane robust sigma (< 50 um: no band-swap mixtures)."""
    out = []
    for scan, L1 in SCANS.items():
        d = json.loads((REPLAY / f"{scan}.json").read_text())
        for k, v in d["steps"].items():
            n = v["n_plus"] + v["n_minus"]
            out.append({"scan": scan, "L1": L1, "step": int(k), "scan_val": v["scan_val"], "spacing_um": v["spacing_um"],
                        "sigz_um": v["drive_sigma_z_um"], "valid": v["valid"], "sig1_um": v["single_image_robust_sigma_um"],
                        "sig1_se_um": 1.166 * v["single_image_robust_sigma_um"] / np.sqrt(n), "n": n,
                        "boot15_um": v["step_bootstrap_sigma_um"], "n_plus": v["n_plus"], "n_minus": v["n_minus"],
                        "use": bool(v["valid"] > 0.5 and v["single_image_robust_sigma_um"] < 50)})
    return out


def _jit_var(row, key="img_spacing_rsig_um"):
    """jitter-only per-image variance [um^2]: total minus the zero-jitter (particle sampling) variance."""
    return max(row[key] ** 2 - row[key + "_zero"] ** 2, 0.0)


def line_tables():
    lines = json.loads((OUT / "mc_lines.json").read_text())
    tab = {}
    for scan in SCANS:
        L = [r for r in lines if r["scan"] == scan]
        tab[scan] = {"L2": np.array([r["L2"] for r in L]),
                     "s0": np.array([r["L1only"]["noiseless_spacing_um"] for r in L]),
                     "zero": np.array([r["L1only"]["img_spacing_rsig_um_zero"] for r in L]),
                     "VL1": np.array([_jit_var(r["L1only"]) for r in L]) / 0.25,     # per deg^2
                     "VL2": np.array([_jit_var(r["L2only"]) for r in L]) / 0.25,
                     "Vrest": np.array([_jit_var(r["rest2024"]) for r in L]),
                     "V24": np.array([_jit_var(r["2024u"]) for r in L])}
    return tab


def _mono_run(s0):
    """(a, b): the longest strictly decreasing, positive run of the line's noiseless spacing (the max can sit in the
    edge-clipped corner; the zero crossing is band-overlap chaos)."""
    runs, a = [], 0
    for b in range(1, len(s0) + 1):
        if b == len(s0) or not (s0[b] < s0[b - 1] and s0[b] > 0):
            runs.append((a, b))
            a = b
    return max(runs, key=lambda r: r[1] - r[0])


def _match(tab, spacing_um):
    """Sim L2 on the scan's L1 line whose noiseless screen spacing equals the measured one (monotone branch:
    spacing decreasing with L2 from the line's maximum on). None if outside the sim range."""
    s0, L2 = tab["s0"], tab["L2"]
    a, b = _mono_run(s0)
    s, l = s0[a:b], L2[a:b]
    if not (s.min() <= spacing_um <= s.max()):
        return None
    return float(np.interp(spacing_um, s[::-1], l[::-1]))


def cmd_calib():
    from scipy.optimize import least_squares
    tab = line_tables()
    rs = real_steps()
    pts = []
    for r in rs:
        t = tab[r["scan"]]
        l2 = _match(t, r["spacing_um"])
        r["sim_L2"] = l2
        if l2 is None:
            r["use"] = False
            continue
        f = lambda k: float(np.interp(l2, t["L2"], t[k]))
        r.update(VL1=f("VL1"), VL2=f("VL2"), Vrest=f("Vrest"), V24=f("V24"), zero_um=f("zero"))
        if r["use"]:
            pts.append(r)
    y = np.array([p["sig1_um"] for p in pts])
    se = np.array([p["sig1_se_um"] for p in pts])
    A = np.array([[p["VL1"], p["VL2"], p["Vrest"]] for p in pts])

    def model(th, A=A):
        a1, a2, c, kr = th
        return np.sqrt(a1 ** 2 * A[:, 0] + a2 ** 2 * A[:, 1] + kr * A[:, 2] + c ** 2)

    fits = {}

    def fit(name, free, x0, A=A, y=y, se=se, pts=pts):
        def res(x):
            th = dict(zip(["a1", "a2", "c", "kr"], [0.43, 0.4, 0.0, 1.0]))
            th.update(zip(free, x))
            return (model([th["a1"], th["a2"], th["c"], th["kr"]], A) - y) / se
        sol = least_squares(res, x0, bounds=([0] * len(free), [5] * len(free)))
        J = sol.jac
        dof = max(len(y) - len(free), 1)
        chi2 = float((sol.fun ** 2).sum())
        cov = np.linalg.pinv(J.T @ J) * max(chi2 / dof, 1.0)       # inflate by reduced chi2
        out = {"params": dict(zip(free, sol.x.tolist())), "err": dict(zip(free, np.sqrt(np.diag(cov)).tolist())),
               "chi2": chi2, "dof": dof, "n": len(y)}
        # bootstrap over steps (per scan correlated systematics -> also leave-one-scan-out below)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            ii = rng.integers(0, len(y), len(y))
            s2 = least_squares(lambda x: res(x)[ii], sol.x, bounds=([0] * len(free), [5] * len(free)))
            bs.append(s2.x)
        bs = np.array(bs)
        out["boot_p16_p84"] = {k: np.percentile(bs[:, j], [16, 84]).tolist() for j, k in enumerate(free)}
        fits[name] = out
        print(name, json.dumps(out, default=float))
        return sol.x
    # (A) common scale on the 2024 phase jitters + camera term; (B) separate L1 / L2 phase + camera; (C) B with the
    # non-phase rest scaled too; (D) B with no camera term
    fit("B_a1_a2_c", ["a1", "a2", "c"], [0.4, 0.4, 3.0])
    fit("C_a1_a2_c_kr", ["a1", "a2", "c", "kr"], [0.4, 0.4, 3.0, 1.0])
    fit("D_a1_a2", ["a1", "a2"], [0.4, 0.4])
    # (E) the L1 phase pinned at the DAQ FAST_PACT per-shot rms (0.085-0.14 deg -> 0.11), camera term fixed at
    # 0 / 3 / 5 um (5 = the flat-region upper bound below); free L2 phase and a scale kr on the non-phase 2024 rows
    for cfix in (0.0, 3.0, 5.0):
        def resE(x, cfix=cfix):
            return (model([0.11, x[0], cfix, x[1]]) - y) / se
        sol = least_squares(resE, [0.3, 1.0], bounds=([0, 0], [5, 5]))
        chi2 = float((sol.fun ** 2).sum())
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * max(chi2 / (len(y) - 2), 1.0)
        rng_ = np.random.default_rng(1)
        bs = np.array([least_squares(lambda x: resE(x)[ii], sol.x, bounds=([0, 0], [5, 5])).x
                       for ii in [rng_.integers(0, len(y), len(y)) for _ in range(300)]])
        fits[f"E_a1pact_c{cfix:.0f}"] = {"params": {"a1": 0.11, "a2": sol.x[0], "kr": sol.x[1], "c": cfix},
                                         "err": {"a2": float(np.sqrt(cov[0, 0])), "kr": float(np.sqrt(cov[1, 1]))},
                                         "boot_p16_p84": {"a2": np.percentile(bs[:, 0], [16, 84]).tolist(),
                                                          "kr": np.percentile(bs[:, 1], [16, 84]).tolist()},
                                         "chi2": chi2, "dof": len(y) - 2, "n": len(y)}
        print(f"E c={cfix}", fits[f"E_a1pact_c{cfix:.0f}"])
    # (G) profile: chi2 over a common phase scale (L1 = L2 = p) with the rest at 2024 and c in {0, 3, 5}
    prof = {}
    for cfix in (0.0, 3.0, 5.0):
        pp = np.linspace(0, 0.5, 51)
        prof[f"c{cfix:.0f}"] = {"p": pp.tolist(), "chi2": [float((((model([p_, p_, cfix, 1.0]) - y) / se) ** 2).sum()) for p_ in pp]}
    fits["G_profile_common_phase_rest2024"] = prof
    # (H) upper bound: all scatter from the phases (non-phase rows off), c = 0; common p and separate a1 / a2
    sol = least_squares(lambda x: (model([x[0], x[0], 0.0, 0.0]) - y) / se, [0.3], bounds=([0], [5]))
    sol2 = least_squares(lambda x: (model([x[0], x[1], 0.0, 0.0]) - y) / se, [0.3, 0.3], bounds=([0, 0], [5, 5]))
    fits["H_phase_only_upper"] = {"common_p": float(sol.x[0]), "chi2_common": float((sol.fun ** 2).sum()),
                                  "a1": float(sol2.x[0]), "a2": float(sol2.x[1]), "chi2_sep": float((sol2.fun ** 2).sum())}
    print("H", fits["H_phase_only_upper"])
    # common scale k on (L1 0.43, L2 0.4, rest 1): sigma^2 = k^2 V24 + c^2
    V24 = np.array([p["V24"] for p in pts])
    solk = least_squares(lambda x: (np.sqrt(x[0] ** 2 * V24 + x[1] ** 2) - y) / se, [1.0, 3.0], bounds=([0, 0], [5, 30]))
    Jk = solk.jac
    chi2k = float((solk.fun ** 2).sum())
    covk = np.linalg.pinv(Jk.T @ Jk) * max(chi2k / (len(y) - 2), 1)
    fits["A_scale_c"] = {"params": {"k": solk.x[0], "c": solk.x[1]}, "err": dict(zip(["k", "c"], np.sqrt(np.diag(covk)).tolist())),
                         "chi2": chi2k, "dof": len(y) - 2, "n": len(y)}
    print("A_scale_c", fits["A_scale_c"])
    # leave-one-scan-out for B
    loso = {}
    for scan in SCANS:
        ii = np.array([p["scan"] != scan for p in pts])
        if ii.sum() < 4:
            continue
        def r_(x):
            return (model([x[0], x[1], x[2], 1.0], A[ii]) - y[ii]) / se[ii]
        loso[f"without_{scan}"] = least_squares(r_, [0.4, 0.4, 3.0], bounds=([0] * 3, [5] * 3)).x.tolist()
    fits["B_leave_one_scan_out"] = loso
    # per-scan single-parameter fit: one phase jitter scale on 2024 phases per scan, c fixed at the B value
    cB = fits["B_a1_a2_c"]["params"]["c"]
    per_scan = {}
    for scan in SCANS:
        ii = np.array([p["scan"] == scan for p in pts])
        if ii.sum() < 2:
            continue
        r_ = lambda x: (np.sqrt(x[0] ** 2 * V24[ii] + cB ** 2) - y[ii]) / se[ii]
        per_scan[scan] = {"k_on_2024": float(least_squares(r_, [1.0], bounds=([0], [5])).x[0]), "n": int(ii.sum())}
    fits["per_scan_k_c_fixed"] = per_scan
    # saturated low-jitter region (TEST steps 29-33 sit at -50 um, flat): an upper bound on the camera term
    sat = [r["sig1_um"] for r in rs if r["scan"] == "TEST_15668" and r["step"] >= 29]
    fits["camera_upper_bound_um_TEST_flat_steps"] = float(np.median(sat))
    # step-level wander from the real scans: residual of the step spacing about a smooth (cubic) fit vs the
    # measurement noise of the step (bootstrap at M=15 rescaled to the step's own image count)
    wander = {}
    for scan in ("TEST_15668", "E300_15673"):
        R = [r for r in rs if r["scan"] == scan and r["use"]]
        x = np.array([r["scan_val"] for r in R])
        sp = np.array([r["spacing_um"] for r in R])
        meas = np.array([r["boot15_um"] * np.sqrt(15 / min(r["n_plus"], r["n_minus"])) for r in R])
        cf = np.polyfit(x, sp, 3)
        res_ = sp - np.polyval(cf, x)
        slope = np.abs(np.polyval(np.polyder(cf), x))
        dof = len(x) - 4
        rms = float(np.sqrt((res_ ** 2).sum() / dof))
        exc = float(np.sqrt(max(rms ** 2 - np.mean(meas ** 2), 0)))
        wander[scan] = {"n": len(x), "resid_rms_um": rms, "meas_rms_um": float(np.sqrt(np.mean(meas ** 2))),
                        "excess_um": exc, "mean_slope_um_per_deg_L2": float(slope.mean()),
                        "excess_deg_L2_equiv": exc / float(slope.mean())}
    fits["step_wander_from_scans"] = wander
    print(json.dumps(wander, indent=1))
    (OUT / "calib.json").write_text(json.dumps({"fits": fits, "steps": rs}, indent=1, default=float))


# ----------------------------------------------------------------------------------------------------------------
EXAMPLES = [(-22.5, -38.0), (-24.5, -36.0), (-20.5, -36.0), (-22.5, -35.0)]


def daqcal_sigmas(gunf, q):
    """DAQ-calibrated scenario = fit E (c = 3 um): L1 phase at the FAST_PACT per-shot rms 0.11 deg, fitted L2 phase,
    the non-phase 2024 rows scaled by the fitted kr (sigma x sqrt(kr): kr scales the variance)."""
    fit = json.loads((OUT / "calib.json").read_text())["fits"]["E_a1pact_c3"]["params"]
    sg = {k: v * np.sqrt(float(fit["kr"])) for k, v in sheet_row("2024u", gunf, q).items()}
    sg.update(L1PhaseSet=float(fit["a1"]), L2PhaseSet=float(fit["a2"]))
    return sg, float(fit["c"])


def cmd_map(n_shot=512, n_zero=512):
    """DAQ-calibrated scenario on the env grid + env-ready jitter maps (npz) for DAQcal and 2024u."""
    pipe = Pipeline()
    gunf, q = float(pipe.base["GUNF:rf_field_scale"]), float(pipe.base["distgen:total_charge:value"])
    sg, c_cam = daqcal_sigmas(gunf, q)
    rng = np.random.default_rng(1018)
    grid = json.loads((OUT / "mc_grid.json").read_text())
    keep = torch.load(OUT / "mc_pools_examples.pt", weights_only=False)
    for g in grid:
        sh0 = pipe.shots(pipe.phys_batch(g["L1"], g["L2"], {}, n_zero, rng))
        sh = pipe.shots(pipe.phys_batch(g["L1"], g["L2"], sg, n_shot, rng))
        g["DAQcal"], _, _ = point_stats(sh, sh0, pipe.S, rng)
        if (g["L1"], g["L2"]) in EXAMPLES:
            keep[f"{g['L1']}_{g['L2']}"]["DAQcal"] = sh
            keep[f"{g['L1']}_{g['L2']}"]["zero_daqcal"] = sh0
        print(f"DAQcal L1 {g['L1']} L2 {g['L2']} img {g['DAQcal']['img_spacing_rsig_um']:.1f} step {g['DAQcal']['step_spacing_std_um']:.1f}", flush=True)
    (OUT / "mc_grid.json").write_text(json.dumps(grid, indent=1))
    torch.save(keep, OUT / "mc_pools_examples.pt")
    write_maps(grid, c_cam, sg)


def write_maps(grid, c_cam, sg):
    L1g = np.array(sorted({g["L1"] for g in grid}))
    L2g = np.array(sorted({g["L2"] for g in grid}))
    cal = json.loads((OUT / "calib.json").read_text())["fits"]
    wander = cal["step_wander_from_scans"]
    rt = np.sqrt(M_IMG)
    doc = {}
    for scen in ("DAQcal", "2024u", "2025u", "TDR"):
        F = {k: np.full((len(L1g), len(L2g)), np.nan) for k in
             ("bunch_spacing_sigma_img", "drive_sigma_z_sigma_img_rel", "bunch_spacing_bias_step", "drive_sigma_z_bias_step",
              "p_keep_valid_step", "info_img_single_spacing_jit_rsig", "info_img_valid_rate", "info_img_valid_rate_zero",
              "info_noiseless_spacing", "info_step_spacing_std_total", "info_step_valid_rate", "info_noiseless_valid_step",
              "info_cloud_spacing_rsig_jit", "info_img_sigz_rel_jit")}
        for g in grid:
            if scen not in g:
                continue
            r = g[scen]
            i, j = int(np.where(L1g == g["L1"])[0][0]), int(np.where(L2g == g["L2"])[0][0])
            F["bunch_spacing_sigma_img"][i, j] = 1e-6 * rt * np.sqrt(max(r["step_spacing_std_um"] ** 2 - r["step_spacing_std_um_zero"] ** 2, 0))
            F["drive_sigma_z_sigma_img_rel"][i, j] = rt * np.sqrt(max(r["step_sigz_rel_std"] ** 2 - r["step_sigz_rel_std_zero"] ** 2, 0))
            F["bunch_spacing_bias_step"][i, j] = 1e-6 * r["step_spacing_bias_um"]
            F["drive_sigma_z_bias_step"][i, j] = 1e-6 * r["step_sigz_bias_um"]
            z = r["step_valid_rate_zero"]
            F["p_keep_valid_step"][i, j] = min(r["step_valid_rate"] / z, 1.0) if z >= 0.5 else 1.0
            F["info_img_single_spacing_jit_rsig"][i, j] = 1e-6 * np.sqrt(_jit_var(r))
            F["info_img_valid_rate"][i, j] = r["img_valid_rate"]
            F["info_img_valid_rate_zero"][i, j] = r["img_valid_rate_zero"]
            F["info_noiseless_spacing"][i, j] = 1e-6 * r["noiseless_spacing_um"]
            F["info_step_spacing_std_total"][i, j] = 1e-6 * r["step_spacing_std_um"]
            F["info_step_valid_rate"][i, j] = r["step_valid_rate"]
            F["info_noiseless_valid_step"][i, j] = r["noiseless_valid_step"]
            F["info_cloud_spacing_rsig_jit"][i, j] = 1e-6 * np.sqrt(_jit_var(r, "img_cloud_spacing_rsig_um"))
            F["info_img_sigz_rel_jit"][i, j] = np.sqrt(max(r["img_sigz_rel_rsig"] ** 2 - r["img_sigz_rel_rsig_zero"] ** 2, 0))
        if np.isnan(F["bunch_spacing_sigma_img"]).all():
            continue
        # cells whose noiseless step is deployed-invalid (corners: ROI edge / band overlap; the reading is invalid
        # anyway) get the median of the valid cells, so the bilinear interpolation does not smear garbage into
        # valid neighbours; marked in info_filled
        # also cells whose zero-jitter single-image reading is degenerate (sampling scatter > 50 um: band split
        # ambiguous, e.g. L1 -26.5 / L2 -41)
        zero_img = np.full_like(F["info_filled"] if "info_filled" in F else F["p_keep_valid_step"], 0.0)
        for g in grid:
            if scen in g:
                zero_img[int(np.where(L1g == g["L1"])[0][0]), int(np.where(L2g == g["L2"])[0][0])] = g[scen]["img_spacing_rsig_um_zero"]
        bad = (F["info_noiseless_valid_step"] < 0.5) | (zero_img > 50)
        F["info_filled"] = bad.astype(float)
        for k in ("bunch_spacing_sigma_img", "drive_sigma_z_sigma_img_rel", "bunch_spacing_bias_step", "drive_sigma_z_bias_step"):
            F[k][bad] = np.nanmedian(F[k][~bad])
        # L1 = -18.5 is the flow's upper L1 bound: half of the jittered shots extrapolate past it (garbage-prone);
        # that row takes the L1 = -20.5 row (info_filled = 2)
        if L1g[-1] >= float(json.loads((OUT / "mc_meta.json").read_text())["knob_high"]["L1PhaseSet"]) - 1e-6:
            for k in ("bunch_spacing_sigma_img", "drive_sigma_z_sigma_img_rel", "bunch_spacing_bias_step", "drive_sigma_z_bias_step",
                      "p_keep_valid_step"):
                F[k][-1] = F[k][-2]
            F["info_filled"][-1] = 2.0
        np.savez(OUT / f"jitter_map_{scen}.npz", L1=L1g, L2=L2g, **F)
        doc[scen] = {k: {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))} for k, v in F.items()}
    rec = {
        "format": ("npz; L1, L2: 1-D grids [deg] (L1PhaseSet, L2PhaseSet, increasing); every other array (len(L1), len(L2)), "
                   "bilinear in (L1, L2), clamped at the grid edge (delta_env_v4._jmap_at). Used by DeltaControlEnvV4 "
                   "(v4.jitter_map): <feature>_sigma_img [SI] / <feature>_sigma_img_rel [relative to the true feature] -> "
                   "per-step noise N(0, sigma_img / sqrt(shots_per_step)); <feature>_bias_step [SI] added; "
                   "p_keep_valid_step multiplies screen_valid_deployed by Bernoulli(p). info_* arrays are ignored by the env."),
        "definitions": {
            "bunch_spacing_sigma_img": "sqrt(30) x the JITTER part of the deployed step-spacing scatter (30 +V and 30 -V images from different shots, per-sign medians, half difference), particle-sampling scatter of the 1024-particle surrogate removed in quadrature. With shots_per_step=30 the env reproduces the MC step scatter exactly (it is NOT the per-image single-sign scatter; that is info_img_single_spacing_jit_rsig, ~1.1-1.3x larger).",
            "drive_sigma_z_sigma_img_rel": "same for the deployed drive sigma_z, relative",
            "bunch_spacing_bias_step": "mean step spacing with jitter minus without (median-of-medians nonlinearity + response curvature)",
            "drive_sigma_z_bias_step": "same for drive sigma_z [m] (jitter lengthens the median-variance sigma_z estimate)",
            "info_filled": "1 = cell replaced by the median of the valid cells (noiseless step deployed-invalid or zero-jitter single-image scatter > 50 um); 2 = L1 = -18.5 row (flow box edge) copied from L1 = -20.5",
            "p_keep_valid_step": "P(step deployed-valid | jitter) / P(step valid | no jitter) where the noiseless step is valid (>= 0.5), capped at 1; 1 elsewhere. The env cannot make a noiseless-invalid step valid.",
        },
        "env_settings": {
            "jitter_map": "results/jitter_mc_2026-10-18/jitter_map_DAQcal.npz (2024 sheet: jitter_map_2024u.npz)",
            "measurement.shot_noise.bunch_spacing_m": 0.886 * c_cam * 1e-6,
            "measurement.shot_noise.note": "camera / analysis part only (fit c per image per sign; step = 0.5 (med+ + med-) -> sigma_step = 0.886 c / sqrt(30); the env adds shot_noise / sqrt(30))",
            "phase_wander_deg": {"L1PhaseSet": 0.10, "L2PhaseSet": None},
            "phase_wander_note": "L1: DAQ FAST_PACT step-mean scatter 0.07-0.16 deg (realism_2026-10-17); L2: see step_wander_from_scans (excess step residual of the real scans, L2-equivalent deg)",
        },
        "daqcal_sigmas": sg, "camera_um": c_cam, "step_wander_from_scans": wander, "ranges": doc,
    }
    # L2 wander given an L1 wander of 0.10 deg: excess^2 = (s_L1 0.10)^2 + (s_L2 w2)^2, s_L1 = the sim per-image L1
    # sensitivity (sqrt VL1, um/deg) at the matched steps, s_L2 = the real scan slope
    steps = json.loads((OUT / "calib.json").read_text())["steps"]
    w2 = {}
    for scan, v in wander.items():
        sl1 = float(np.mean([np.sqrt(r["VL1"]) for r in steps if r["scan"] == scan and r["use"]]))
        w2[scan] = {"s_L1_um_per_deg": sl1, "w2_deg": float(np.sqrt(max(v["excess_um"] ** 2 - (0.10 * sl1) ** 2, 0))
                                                         / v["mean_slope_um_per_deg_L2"])}
    rec["phase_wander_split"] = w2
    rec["env_settings"]["phase_wander_deg"]["L2PhaseSet"] = float(np.round(np.median([x["w2_deg"] for x in w2.values()]), 3))
    (OUT / "jitter_map_README.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec["env_settings"], indent=1, default=float))


def cmd_maps_only():
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
    base = dict(resolve_sweep_set("xtcav23e_l0b2_inj12")[3])
    sg, c = daqcal_sigmas(float(base["GUNF:rf_field_scale"]), float(base["distgen:total_charge:value"]))
    write_maps(json.loads((OUT / "mc_grid.json").read_text()), c, sg)


# ----------------------------------------------------------------------------------------------------------------
def cmd_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = json.loads((OUT / "mc_grid.json").read_text())
    cal = json.loads((OUT / "calib.json").read_text())
    fB = cal["fits"]["B_a1_a2_c"]["params"]
    fE = cal["fits"]["E_a1pact_c3"]["params"]
    c = fE["c"]
    L1g = np.array(sorted({g["L1"] for g in grid}))
    L2g = np.array(sorted({g["L2"] for g in grid}))
    # fig 1: per-image single-sign spacing jitter sigma maps
    scens = [s_ for s_ in ("TDR", "2024u", "2024c", "2025u", "DAQcal") if s_ in grid[0]]
    fig, ax = plt.subplots(1, len(scens), figsize=(3.6 * len(scens), 3.8), sharey=True)
    for a, sc in zip(ax, scens):
        Z = np.full((len(L2g), len(L1g)), np.nan)
        S0 = np.full_like(Z, np.nan)
        V = np.full_like(Z, np.nan)
        for g in grid:
            i, j = int(np.where(L1g == g["L1"])[0][0]), int(np.where(L2g == g["L2"])[0][0])
            Z[j, i] = np.sqrt(_jit_var(g[sc]))
            S0[j, i] = g[sc]["noiseless_spacing_um"]
            V[j, i] = g[sc]["noiseless_valid_step"]
        im = a.pcolormesh(L1g, L2g, Z, vmin=0, vmax=60, cmap="viridis", shading="nearest")
        cs = a.contour(L1g, L2g, S0, levels=[60, 100, 200, 300, 400], colors="w", linewidths=0.8)
        a.clabel(cs, fmt="%d um", fontsize=7)
        for j in range(len(L2g)):
            for i in range(len(L1g)):
                a.text(L1g[i], L2g[j], f"{Z[j, i]:.0f}", ha="center", va="center", fontsize=6.5,
                       color="k" if Z[j, i] > 35 else "w")
                if V[j, i] < 0.5:
                    a.plot(L1g[i], L2g[j], "rx", ms=9, mew=0.8)
        a.set_title(f"{sc}", fontsize=10)
        a.set_xlabel("L1PhaseSet [deg]")
    ax[0].set_ylabel("L2PhaseSet [deg]")
    fig.colorbar(im, ax=ax, shrink=0.8, label="robust sigma, single image, sampling removed [um]")
    fig.suptitle("Per-image (single-sign) measured-spacing jitter; white = noiseless screen spacing, red x = noiseless step deployed-invalid", fontsize=9)
    fig.savefig(OUT / "fig1_spacing_sigma_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # fig 2: calibration
    tab = line_tables()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, scan in zip(ax, SCANS):
        t = tab[scan]
        R = [r for r in cal["steps"] if r["scan"] == scan]
        for use, mk, lab in ((True, "o", "real valid (fit)"), (False, "x", "real excluded")):
            RR = [r for r in R if r["use"] == use and abs(r["spacing_um"]) < 450 and r["sig1_um"] < 60]
            a.errorbar([r["spacing_um"] for r in RR], [r["sig1_um"] for r in RR], [r["sig1_se_um"] for r in RR],
                       fmt=mk, color="k" if use else "0.6", ms=4, label=lab, lw=0.8)
        sl = slice(*_mono_run(t["s0"]))
        x = t["s0"][sl]
        a.plot(x, t["zero"][sl], "--", color="0.5", label="sim zero jitter (particle sampling, not in the curves below)")
        a.plot(x, np.sqrt(t["V24"][sl] + c ** 2), "-", color="C0", label=f"sim 2024 sheet (L1 0.43, L2 0.40) + cam {c:.0f} um")
        a.plot(x, np.sqrt(fB["a1"] ** 2 * t["VL1"][sl] + fB["a2"] ** 2 * t["VL2"][sl] + t["Vrest"][sl]), "-", color="C1", lw=0.8,
               label=f"fit B: L1 {fB['a1']:.2f}, L2 {fB['a2']:.2f}, rest 2024, cam 0")
        a.plot(x, np.sqrt(fE["a1"] ** 2 * t["VL1"][sl] + fE["a2"] ** 2 * t["VL2"][sl] + fE["kr"] * t["Vrest"][sl] + c ** 2),
               "-", color="C3", lw=2, label=f"DAQcal (fit E): L1 {fE['a1']:.2f} (PACT), L2 {fE['a2']:.2f}, rest x{np.sqrt(fE['kr']):.2f}, cam {c:.0f}")
        a.plot(x, np.sqrt(fE["a1"] ** 2 * t["VL1"][sl]), ":", color="C3", label="  DAQcal L1 part")
        a.plot(x, np.sqrt(fE["a2"] ** 2 * t["VL2"][sl]), "-.", color="C3", lw=0.8, label="  DAQcal L2 part")
        a.plot(x, np.sqrt(fE["kr"] * t["Vrest"][sl]), ":", color="C4", label="  DAQcal non-phase part")
        a.set_title(f"{scan} (L1 {SCANS[scan]})", fontsize=9)
        a.set_xlabel("measured / noiseless screen spacing [um]")
        a.set_ylim(0, 60)
        a.set_xlim(-60, 450)
    ax[0].set_ylabel("per-image single-sign spacing robust sigma [um]")
    ax[0].legend(fontsize=6.5)
    fig.savefig(OUT / "fig2_calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # fig 3: per-step error distributions vs the env's current Gaussian model
    keep = torch.load(OUT / "mc_pools_examples.pt", weights_only=False)
    meta = json.loads((OUT / "mc_meta.json").read_text())
    S = meta["S_streak"]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(1, len(keep), figsize=(4 * len(keep), 3.6))
    env_sig = np.hypot(15 / np.sqrt(30), 6.0)
    for a, (k, pools) in zip(np.atleast_1d(ax), keep.items()):
        st0 = steps_from_shots(pools["zero"], S, 4000, rng)
        ref = np.mean(st0["spacing"])
        bins = np.linspace(-30, 30, 61)
        for sc, col in (("2024u", "C0"), ("DAQcal", "C3")):
            if sc not in pools:
                continue
            st = steps_from_shots(pools[sc], S, 4000, rng)
            e = 1e6 * (st["spacing"] - ref) + rng.normal(0, 0.886 * c / np.sqrt(30), 4000)
            a.hist(e, bins, density=True, histtype="step", color=col, label=f"MC {sc} (+cam): sd {e.std():.1f}, bias {e.mean():+.1f}")
        x = np.linspace(-30, 30, 300)
        g = lambda s_: np.exp(-0.5 * (x / s_) ** 2) / (s_ * np.sqrt(2 * np.pi))
        a.plot(x, 0.99 * g(env_sig) + 0.01 * g(np.hypot(env_sig, 30)), "k--", label=f"env now: sd {env_sig:.1f} (+1% x5 outliers)")
        a.set_title(f"L1 {k.split('_')[0]}, L2 {k.split('_')[1]}: noiseless {1e6 * ref:.0f} um", fontsize=9)
        a.set_xlabel("step spacing error [um]")
        a.legend(fontsize=6.5)
    fig.suptitle("Per-step (30 +V / 30 -V images, deployed medians) spacing error vs the noiseless reading; streak-scale error not included", fontsize=9)
    fig.savefig(OUT / "fig3_step_error_dists.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # fig 4: Bmad vs surrogate
    bp = OUT / "bmad/bmad_rows.json"
    if bp.exists():
        rows = json.loads(bp.read_text())
        summ = json.loads((OUT / "bmad/bmad_summary.json").read_text())
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for ip, s_ in enumerate(summ):
            J = [r for r in rows if r["point"] == ip and r["kind"] == "jit" and r["ok"]]
            bs = 1e6 * np.array([r["spacing_cloud"] for r in J]); fs = 1e6 * np.array([r["flow_spacing_cloud"] for r in J])
            bz = 1e6 * np.array([r["sigz_cloud"] for r in J]); fz = 1e6 * np.array([r["flow_sigz_cloud"] for r in J])
            lab = f"L1 {s_['L1']}, L2 {s_['L2']}"
            ax[0].plot(fs - fs.mean(), bs - bs.mean(), "o", ms=3, label=f"{lab}: r={s_['per_shot_corr_spacing']:.2f}")
            ax[1].plot(fz - fz.mean(), bz - bz.mean(), "o", ms=3, label=f"{lab}: r={s_['per_shot_corr_sigz']:.2f}")
        lim = ax[0].get_xlim()
        for a in ax[:2]:
            l_ = max(abs(np.array(a.get_xlim() + a.get_ylim())))
            a.plot([-l_, l_], [-l_, l_], "k:", lw=0.8)
            a.legend(fontsize=7)
        ax[0].set_xlabel("flow cloud spacing - mean [um]"); ax[0].set_ylabel("Bmad cloud spacing - mean [um]")
        ax[1].set_xlabel("flow drive sigma_z - mean [um]"); ax[1].set_ylabel("Bmad drive sigma_z - mean [um]")
        ax[0].set_title("per shot, identical linac-jitter draws (2024u)", fontsize=9)
        xs = np.arange(len(summ))
        ax[2].bar(xs - 0.25, [s_["bmad_spacing_std_um"] for s_ in summ], 0.25, label="Bmad jitter (common seed)")
        ax[2].bar(xs, [s_["flow_spacing_std_um"] for s_ in summ], 0.25, label="flow jitter (cloud, 8x4096 draws)")
        ax[2].bar(xs + 0.25, [s_["bmad_seed_spacing_std_um"] or 0 for s_ in summ], 0.25, label="Bmad seed noise (zero jitter)")
        ax[2].set_xticks(xs, [f"{s_['L1']}/{s_['L2']}" for s_ in summ])
        ax[2].set_ylabel("cloud spacing std [um]"); ax[2].legend(fontsize=7)
        fig.savefig(OUT / "fig4_bmad_vs_surrogate.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    {"mc": cmd_mc, "probe": cmd_probe, "bmad": cmd_bmad, "calib": cmd_calib, "map": cmd_map, "figs": cmd_figs, "maps_only": cmd_maps_only}[sys.argv[1]](*sys.argv[2:])
