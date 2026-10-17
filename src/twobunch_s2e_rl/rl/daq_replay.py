"""L2-only replay of a screen-mode (S2) delta controller on REAL XTCAV scan data: the first time the policy acts on
real images.

The 04-15 DAQ scans step L2_PHASE.MKB in 0.25 deg steps with the BC11 loop on (L1 held), so the plant is the scan
itself: the policy's L2 (beam-effective, physical) maps to a multiknob position mkb = L2 - L2_ref and the reading is
the online-replay features of that scan step (results/online_replay_v6_median_cut_2026-10-15, the deployed chain):
spacing re-drawn per visit from the step's own per-image +/-V offsets (bootstrap of M = shots_per_step / 2 images
per sign, median, half-difference), linearly interpolated between the neighbouring steps; drive sigma_z and the
drive charge fraction from the step values, nearest step; validity = the step's deployed validity (outside the scan:
invalid). The policy's own env runs unchanged except: L1 frozen (cum_limit 0, no drift: the loop holds it), L2 gain
1 / no drift (the scan's response IS the machine's), no synthetic measurement noise (the data has its own), start and
goal points inside the scan's valid window. Goals are the scan's readings at the goal points (as recorded on the
machine). There is no truth on real data: metrics are |mkb_final - mkb_goal| (did it return to the goal's L2
setting) and the measured spacing error at the end vs the goal reading.

  python -m twobunch_s2e_rl.rl.daq_replay --logdir <S2 run> --scan TEST_15668 --k 64 --out results/daq_replay_2026-10-17/<tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

REPLAY = "results/online_replay_v6_median_cut_2026-10-15"
BLEN14 = "results/daq_replay_2026-10-17/blen14_steps.json"   # per-step BLEN_LI14_888_BRAW medians of the same scans
L2_REF = -41.0          # sim L2 of mkb = 0 (the policy sees no absolute L2; keeps the scan inside the knob box)


class ScanPlant:
    def __init__(self, scan, ks=None, shots_per_step=30, seed=0):
        d = json.loads(Path(f"{REPLAY}/{scan}.json").read_text())
        st = sorted(d["steps"].values(), key=lambda s: s["scan_val"])
        self.mkb = np.array([s["scan_val"] for s in st])
        self.op = [np.asarray(s["offsets_plus_um"], float) for s in st]
        self.om = [np.asarray(s["offsets_minus_um"], float) for s in st]
        self.sz = np.array([s["drive_sigma_z_um"] for s in st]) * 1e-6
        self.fq = np.array([s["drive_charge_fraction"] for s in st])
        self.valid = np.array([float(s["valid"]) for s in st])
        self.m = max(1, int(shots_per_step) // 2)
        self.ks, self.j2 = ks, (ks.index("L2PhaseSet") if ks is not None else None)
        self.rng = np.random.default_rng(seed)
        self.visits = []

    def valid_window(self):
        v = self.mkb[self.valid > 0.5]
        return float(v.min()), float(v.max())

    def _spacing(self, i):
        op = self.rng.choice(self.op[i], self.m) if len(self.op[i]) else np.array([np.nan])
        om = self.rng.choice(self.om[i], self.m) if len(self.om[i]) else np.array([np.nan])
        return 0.5 * (np.median(op) - np.median(om)) * 1e-6

    def read(self, l2):
        x = float(l2) - L2_REF
        if x < self.mkb[0] - 1e-9 or x > self.mkb[-1] + 1e-9:
            i = 0 if x < self.mkb[0] else len(self.mkb) - 1
            return self._spacing(i), self.sz[i], self.fq[i], 0.0, x
        k = int(np.clip(np.searchsorted(self.mkb, x) - 1, 0, len(self.mkb) - 2))
        t = (x - self.mkb[k]) / (self.mkb[k + 1] - self.mkb[k])
        sp = (1 - t) * self._spacing(k) + t * self._spacing(k + 1)
        n = k if t < 0.5 else k + 1
        return sp, self.sz[n], self.fq[n], self.valid[n], x


def wrap_observe(base, plant: ScanPlant):
    """The policy's screen plant (flow) for every non-screen key, with the screen readings replaced by the scan's."""
    def observe(knobs_norm):
        obs = base(knobs_norm)
        l2 = plant.ks.denormalize(knobs_norm.detach())[:, plant.j2].cpu().numpy()
        r = [plant.read(v) for v in l2]
        dev = knobs_norm.device
        t = lambda col: torch.tensor([x[col] for x in r], device=dev, dtype=torch.float32)  # noqa: E731
        sp, sz, fq, ok = t(0), t(1), t(2), t(3)
        for key, v in (("bunch_spacing", sp), ("drive_sigma_z", sz), ("drive_charge_fraction", fq)):
            obs[key] = v
            obs[f"{key}_cloud"] = v                # no truth on real data
        obs["screen_valid"] = ok
        obs["screen_valid_deployed"] = ok
        obs["screen_misassign"] = torch.zeros_like(ok)
        plant.visits.append([(x[4], x[3]) for x in r])
        return obs
    observe.resample = getattr(base, "resample", None)
    return observe


def run(logdir, scan, k, out, device="cuda:0", policy="final", seed=777):
    from ..artifacts import policy_flow_ckpt
    from ._train_utils import build_env_fn
    from .rollout import load_policy, rollout
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(logdir) / "cfg.yaml").read_text())
    de = cfg["params"]["diff_env"]
    if de.get("plant") != "xtcav_screen":
        raise SystemExit("DAQ replay needs a screen-mode (S2) policy")
    v4 = de["v4"]
    v4["offcontext_goal_frac"] = 0.0
    v4["actuators"]["L1PhaseSet"]["cum_limit"] = 0.0
    for key in ("readback_noise", "readback_drift"):
        v4["actuators"]["L1PhaseSet"][key] = 0.0
    v4["actuators"]["L2PhaseSet"]["gain_range"] = [1.0, 1.0]
    v4["drift_per_step"] = {}
    v4["drift_ramp"] = {}
    ms = de.setdefault("measurement", {})
    ms["shot_noise"] = {kk: 0.0 for kk in ms.get("shot_noise", {})}
    ms["step_noise_floor"] = {kk: 0.0 for kk in ms.get("step_noise_floor", {})}
    ms["outlier_prob"], ms["scale_error_halfwidth"] = 0.0, 0.0
    de["offset_sigma"] = {}
    shots = int(ms.get("shots_per_step", 30))
    probe = ScanPlant(scan)
    lo, hi = probe.valid_window()
    de["start_box"]["L2PhaseSet"] = [L2_REF + lo, L2_REF + hi]
    n_steps = int(de["episode_length"])
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(num_envs=k, device=device, seed=seed,
                                                       episode_length=10 ** 6, no_grad=True)
    plant = ScanPlant(scan, env._ks, shots, seed=seed)
    env._observe_fn = wrap_observe(env._observe_fn, plant)
    blen_real = bool(getattr(env, "_blen", None))
    if blen_real:
        # BLEN-obs policies: the REAL BC14 BLEN of the scan (ln of the step median, interpolated); BC11 = 0 (its
        # loop held it, and L1 is frozen); slope scale 1 (set after reset); the env's reading noise stays
        b = json.loads(Path(BLEN14).read_text())[scan]
        xs, ln14 = np.asarray(b["scan_val"]), np.log(np.asarray(b["blen14_median"]))
        names = [m[0] for m in env._blen]

        def poly(ctrl):
            x = (ctrl[:, 1] - L2_REF).cpu().numpy()
            cols = [torch.as_tensor(np.interp(x, xs, ln14), dtype=ctrl.dtype, device=ctrl.device)
                    if "LI14" in nm else torch.zeros(ctrl.shape[0], dtype=ctrl.dtype, device=ctrl.device)
                    for nm in names]
            return torch.stack(cols, -1)
        env._blen_poly = poly
    actor, rms = load_policy(str(Path(logdir) / f"{policy}_policy.pt"), device)
    env.reset()                                    # starts, goal points (valid scan readings) via the scan plant
    if blen_real:
        env._blen_s = torch.ones_like(env._blen_s)
    goal_mkb = (env._goal_ctrl[:, 1] - L2_REF).cpu().numpy()
    start_mkb = (env.true_controls()[:, 1] - L2_REF).cpu().numpy()
    traj = rollout(env, actor, rms, n_steps=n_steps)
    end_mkb = (env.true_controls()[:, 1] - L2_REF).cpu().numpy()
    gkey = next(g for key, g, _ in env._goal_terms if key == "bunch_spacing")
    sp_err = (traj["bunch_spacing"][-1] - traj[gkey][-1]).abs().cpu().numpy()
    d_end = np.abs(end_mkb - goal_mkb)
    d0 = np.abs(start_mkb - goal_mkb)
    steps_v = np.array(plant.visits[-n_steps:])             # (n_steps, k, 2): the policy's own steps
    valid_frac = float(steps_v[..., 1].mean())
    l2_path = (traj["true_controls"][..., 1] - L2_REF).cpu().numpy() if "true_controls" in traj else None
    summ = {"logdir": str(logdir), "scan": scan, "k": k, "policy": policy, "valid_window_mkb": [lo, hi],
            "blen_obs_from_daq": blen_real,
            "dmkb_start_median": float(np.median(d0)), "dmkb_final_median": float(np.median(d_end)),
            "dmkb_final_p90": float(np.percentile(d_end, 90)),
            "frac_dmkb_final_le_0p25": float((d_end <= 0.25).mean()),
            "spacing_err_final_meas_median_um": float(np.median(sp_err) * 1e6),
            "spacing_err_final_meas_p90_um": float(np.percentile(sp_err, 90) * 1e6),
            "frac_spacing_within_10um_meas": float((sp_err <= 10e-6).mean()),
            "valid_reading_frac": valid_frac,
            "frac_left_scan": float(((steps_v[..., 0] < probe.mkb[0]) | (steps_v[..., 0] > probe.mkb[-1])).mean()),
            "per_episode": {"start_mkb": start_mkb.tolist(), "goal_mkb": goal_mkb.tolist(), "end_mkb": end_mkb.tolist()}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    np.savez(out / "traj.npz", start=start_mkb, goal=goal_mkb, end=end_mkb, sp_err=sp_err,
             **({"l2_path": l2_path} if l2_path is not None else {}),
             spacing=traj["bunch_spacing"].cpu().numpy(), goal_spacing=traj[gkey].cpu().numpy())
    print(f"[{scan}] |dmkb| start {summ['dmkb_start_median']:.2f} -> final {summ['dmkb_final_median']:.2f} deg "
          f"(p90 {summ['dmkb_final_p90']:.2f}, <=0.25: {summ['frac_dmkb_final_le_0p25']:.2f}); spacing err meas "
          f"{summ['spacing_err_final_meas_median_um']:.1f} um, in10 {summ['frac_spacing_within_10um_meas']:.2f}; "
          f"valid {valid_frac:.2f}, left scan {summ['frac_left_scan']:.2f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--scan", default="TEST_15668")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--policy", default="final")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.logdir, a.scan, a.k, a.out, device=a.device, policy=a.policy)


if __name__ == "__main__":
    main()
