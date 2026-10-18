"""Figures for the 2026-10-17 notebook entry (injector coverage campaign + B2-cov surrogate):
injector coverage, dataset growth, surrogate progression, and held-out pair parity.
Writes lab-notebook/images/2026-10-17_*.png. Read-only on data/, processed/, trained/, results/.

  PYTHONPATH=src OMP_NUM_THREADS=4 python scripts/notebook_figs_2026-10-17_data_surrogate.py [--cache PRED.npz]

The parity figure recomputes predicted vs Bmad injector-induced delta spacing with the logic of
scripts/surrogate_scorecard.py (same table, role map, scaler, CRN seed, n=2048) on the cov table's held-out pairs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#eb6834"  # gate bands (orange, light alpha) as in the 10-16 figures
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

TABLE = "processed/xtcav23e_inj_cov_2026-10-17.h5"
ROLE_MAP = "processed/role_map_ftc_s0_2026-10-16.json"
CAMPS = ("injector_full", "injector_topup", "injector_cov_2026-10-16")
CAMP_LAB = {"injector_full": "injector_full\n(10-15)", "injector_topup": "injector_topup\n(10-16)",
            "injector_cov_2026-10-16": "injector_cov\n(10-16, new)"}
CAMP_COL = {"injector_full": BLUE, "injector_topup": ORANGE, "injector_cov_2026-10-16": AQUA}


def manifest(c):
    return {int(r["idx"]): r for r in json.loads(Path(f"data/{c}/manifest.json").read_text())}


def datamodule(scaler=None):
    from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
    roles = json.loads(Path(ROLE_MAP).read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=TABLE, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    role = np.full(len(dm.full), "train", dtype=object)
    role[np.asarray(dm.val.indices)] = "val"
    role[np.asarray(dm.test.indices)] = "test"
    return dm, role


def table_meta():
    with h5py.File(TABLE) as h:
        t = {k: h[k][...] for k in ("campaign", "idx", "pair_key", "pair_ref", "drive_density", "witness_density",
                                    "holdout", "block")}
        t["names"] = str(h.attrs["campaigns"]).split(",")
        t["codes"] = str(h.attrs["block_codes"]).split(",")
    return t


def pairs_of(t, rows_ok):
    by = {}
    for r in np.flatnonzero(rows_ok & (t["pair_key"] >= 0)):
        by.setdefault(int(t["pair_key"][r]), {})["ref" if t["pair_ref"][r] else "pt"] = r
    return [(x["pt"], x["ref"]) for x in by.values() if len(x) == 2]


# ----------------------------------------------------------------------------------------------------- 1. coverage
def coverage(role):
    from twobunch_s2e_rl.datagen.sweep_params import injector_golden, resolve_sweep_set
    names, lo, hi, *_ = resolve_sweep_set("inj12")
    gold = injector_golden() if callable(injector_golden) else injector_golden
    lo, hi = dict(zip(names, lo)), dict(zip(names, hi))
    half = {k: (hi[k] - lo[k]) / 2 for k in names}
    norm = lambda k, v: (v - gold[k]) / half[k]
    t = table_meta()
    # role of each injector point = role of its table rows (group split keeps a point's rows together)
    pts = {}  # (campaign, point) -> dict(inj=..., roles=Counter, region=...)
    for c in ("injector_full", "injector_cov_2026-10-16"):
        man, ci = manifest(c), t["names"].index(c)
        for r in np.flatnonzero(t["campaign"] == ci):
            m = man[int(t["idx"][r])]
            if m["injector_point"] == "golden":
                continue
            d = pts.setdefault((c, m["injector_point"]), {"inj": m["injector"], "roles": Counter(),
                                                         "region": m.get("injector_region", "full")})
            d["roles"][role[r]] += 1
    groups = {"injector_full LHS (full box)": [d for (c, _), d in pts.items() if c == "injector_full"],
              "cov: full-box LHS": [d for (c, _), d in pts.items() if c != "injector_full" and d["region"] == "full"],
              "cov: DAQ-local": [d for (c, _), d in pts.items() if c != "injector_full" and d["region"] == "local"]}
    cols = dict(zip(groups, (BLUE, ORANGE, AQUA)))
    ntest = sum(d["roles"]["test"] > 0 for g in groups.values() for d in g)
    nval = sum(d["roles"]["val"] > 0 for g in groups.values() for d in g)
    panels = [("drive_fraction", "GUNF:theta0_deg"), ("laser_timing_ps", "distgen:total_charge:value")]
    short = {"drive_fraction": "drive fraction", "GUNF:theta0_deg": "gun phase θ0", "laser_timing_ps": "laser timing",
             "distgen:total_charge:value": "total charge", "GUNF:rf_field_scale": "gun amplitude",
             "distgen:t_dist:dists:d2:avg_t:value": "d2 delay (PSDL)", "L0AF_phase_offset_deg": "L0AF phase",
             "SOL10111:solenoid_field_scale": "solenoid", "distgen:r_dist:sigma_xy:value": "laser σxy",
             "distgen:r_dist:truncation_radius:value": "trunc. radius", "distgen:start:MTE:value": "MTE",
             "sigma_t_ps": "laser σt"}
    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    for j, (kx, ky) in enumerate(panels):
        ax = fig.add_subplot(gs[j])
        bx0, bx1, by0, by1 = norm(kx, lo[kx]), norm(kx, hi[kx]), norm(ky, lo[ky]), norm(ky, hi[ky])
        ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=INK2, lw=0.8, ls="--", zorder=1))
        for lab, g in groups.items():
            x = np.array([norm(kx, d["inj"][kx]) for d in g])
            y = np.array([norm(ky, d["inj"][ky]) for d in g])
            ax.scatter(x, y, s=16, color=cols[lab], alpha=0.8, lw=0.4, ec=SURF, zorder=2,
                       label=f"{lab} (n={len(g)})")
            te = np.array([d["roles"]["test"] > 0 for d in g])
            ax.scatter(x[te], y[te], s=58, facecolor="none", ec=INK, lw=0.9, zorder=3)
        ax.scatter([0], [0], marker="*", s=220, color=INK, ec=SURF, lw=1.0, zorder=4, label="golden")
        ax.scatter([], [], s=58, facecolor="none", ec=INK, lw=0.9, label=f"held-out test point (n={ntest})")
        ax.set_xlabel(f"{short[kx]}  (x − golden) / half-box")
        ax.set_ylabel(f"{short[ky]}  (x − golden) / half-box")
        ax.set(xlim=(-1.4, 1.2), ylim=(-1.2, 1.2), aspect="equal")
        ax.set_title(f"{short[kx]} × {short[ky]}  (dashed = box)", fontsize=8.5, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.36, 0.035))
    # box half-width in units of the local (DAQ shift-to-shift) sigma
    ctx = yaml.safe_load(Path("configs/rl/eval/injector_ctx_2026-10-16.yaml").read_text())["context_dist"]
    loc = groups["cov: DAQ-local"]
    ax = fig.add_subplot(gs[2])
    rows = []
    for k in names:
        w = ctx[k]["width"]
        sig = w / np.sqrt(3) if ctx[k]["dist"] == "uniform" else w
        emp = np.std([d["inj"][k] for d in loc], ddof=1)
        # box edge toward the nearer side, in sigma units (asymmetric drive fraction box: use the narrow side)
        edge = min(hi[k] - gold[k], gold[k] - lo[k])
        rows.append((short[k], edge / sig, edge / emp, ctx[k]["dist"]))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=AQUA, label="nominal σ (config)")
    ax.scatter([r[2] for r in rows], y, marker="|", s=120, color=INK, lw=1.4, zorder=3,
               label="empirical σ of the 125 local points")
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.12, yy, f"{r[1]:.1f}σ" + (" (uniform)" if r[3] == "uniform" else ""), va="center",
                fontsize=7, color=INK2)
    ax.axvline(3, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("box half-width (nearer edge) / local σ")
    ax.set_xlim(0, max(r[1] for r in rows) + 1.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Box edge is only 1–4 local σ from golden (dotted = 3σ)", fontsize=9, loc="left")
    npt = {k: len(v) for k, v in groups.items()}
    fig.suptitle(f"Injector coverage: {npt['injector_full LHS (full box)']} prior LHS points + "
                 f"{npt['cov: full-box LHS']} new full-box + {npt['cov: DAQ-local']} new DAQ-local points "
                 f"(normalized to golden ± half-box; {ntest} test / {nval} val points held out)",
                 fontsize=10, x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.005, "Drive-fraction box is asymmetric (0.66–0.80 around golden 0.75), so its lower edge sits at "
             "−1.29 half-boxes. Bar panel: σ from configs/rl/eval/injector_ctx_2026-10-16.yaml "
             "(uniform width / √3 for drive fraction); box edge on the nearer side.", fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.24)
    fig.savefig(OUT / "2026-10-17_injector_coverage.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- 2. dataset growth
def growth(role):
    t = table_meta()
    both = t["drive_density"] & t["witness_density"]
    pairs = pairs_of(t, both)  # scorecard's definition (both bunches have densities)
    stats = {}
    for c in CAMPS:
        ci = t["names"].index(c)
        man = manifest(c)
        rows_c = np.flatnonzero(t["campaign"] == ci)
        pts = {man[int(t["idx"][r])]["injector_point"] for r in rows_c} - {"golden"}
        new_pts = len(pts) if c != "injector_topup" else 0  # top-up re-uses injector_full points
        pc = [p for p in pairs if t["campaign"][p[1]] == ci]  # credited to the campaign that ran the golden re-run
        stats[c] = {"rows": len(rows_c), "points": new_pts, "pairs": len(pc),
                    "split": Counter(role[p[0]] for p in pc)}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    x = np.arange(len(CAMPS))
    for ax, (k, lab) in zip(axes[:3], (("rows", "table rows (Bmad samples)"), ("points", "new distinct injector points"),
                                       ("pairs", "exact golden pairs created\n(golden re-run at same linac)"))):
        v = [stats[c][k] for c in CAMPS]
        ax.bar(x, v, width=0.62, color=[CAMP_COL[c] for c in CAMPS], ec=SURF, lw=2)
        for xx, vv in zip(x, v):
            ax.text(xx, vv, f"{vv:,}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [CAMP_LAB[c] for c in CAMPS], fontsize=7.5)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_ylim(0, max(v) * 1.15)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[1].text(1, 22, "top-up re-uses\nexisting points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax = axes[3]
    rl = ("train", "val", "test")
    w = 0.26
    for j, c in enumerate(CAMPS):
        v = [stats[c]["split"][r] for r in rl]
        ax.bar(np.arange(3) + (j - 1) * w, v, width=w, color=CAMP_COL[c], ec=SURF, lw=1.5,
               label=CAMP_LAB[c].replace("\n", " "))
        for xx, vv in zip(np.arange(3) + (j - 1) * w, v):
            ax.text(xx, vv, str(vv), ha="center", va="bottom", fontsize=7, color=INK2)
    tot = {r: sum(stats[c]["split"][r] for c in CAMPS) for r in rl}
    ax.set_xticks(np.arange(3), [f"{r}\n(total {tot[r]})" for r in rl], fontsize=8)
    ax.set_title("pairs by split role (role of the point row), per campaign of the golden re-run", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"Injector data so far: {sum(s['rows'] for s in stats.values()):,} rows, "
                 f"{sum(s['points'] for s in stats.values())} injector points, {len(pairs):,} exact pairs "
                 f"({tot['val']} val / {tot['test']} test held out) in {TABLE.split('/')[-1]}",
                 fontsize=10, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_dataset_growth.png", dpi=110)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------------------- 3. surrogate progression
def progression():
    J = lambda p: json.loads(Path(p).read_text())["models"]
    b, c, tu = (J("results/surrogate_scorecard_2026-10-16/B/scorecard.json"),
                J("results/surrogate_scorecard_2026-10-16/C/scorecard.json"),
                J("results/surrogate_scorecard_2026-10-17/topup_table/scorecard.json"))
    cov = J("results/surrogate_scorecard_2026-10-17/cov_table/scorecard.json")
    # all five on the 10-16 top-up table (same 93 val pairs, same old-val rows)
    seq = [("ftc_s0", b["ftc_s0 best"], INK2), ("A", b["A best"], INK2),
           ("B2", b["B2 (top-up, w_pair 3) best"], BLUE),
           ("C2", c["C2 (w_pair 10) best"], INK2), ("B2-cov", tu["B2-cov best"], AQUA)]
    x = np.arange(len(seq))
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.1]})
    spec = (("R2", "val pair R²", (0.8, 1.0)), ("slope", "val pair slope", (0.85, 1.15)),
            ("p90_um", "val pair p90 |err| [µm]", (0.0, 8.0)))
    for ax, (k, lab, (g0, g1)) in zip(axes[:3], spec):
        v = np.array([m["paired_val"][k] for _, m, _ in seq])
        lo = np.array([m["paired_val"]["ci90"][k][0] for _, m, _ in seq])
        hi = np.array([m["paired_val"]["ci90"][k][1] for _, m, _ in seq])
        ax.axhspan(g0, g1, color=BAND, alpha=0.10, lw=0, zorder=0)
        ax.plot(x, v, color=GRID, lw=1.2, zorder=1)
        for xx, vv, l_, h_, (_, _, col) in zip(x, v, lo, hi, seq):
            ax.errorbar(xx, vv, yerr=[[vv - l_], [h_ - vv]], fmt="o", color=col, capsize=3, lw=1.4, ms=6, zorder=2)
        ax.text(x[-1] + 0.25, v[-1], f"{v[-1]:.2f}" if k != "p90_um" else f"{v[-1]:.1f}", va="center", fontsize=8,
                color=INK)
        ax.set_xticks(x, [s for s, _, _ in seq], fontsize=8)
        ax.set_xlim(-0.5, len(seq) - 0.2)
        ax.set_title(lab + " (90% CI)", fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylim(0.4, 1.0)
    axes[1].set_ylim(0.5, 1.2)
    axes[2].set_ylim(5, 13.5)
    ax = axes[3]
    sz = [m["gate_oldval"]["sigma_z_rel"] for _, m, _ in seq]
    ax.axhspan(0.09, 0.112, color=BAND, alpha=0.10, lw=0)
    ax.plot(x, sz, color=GRID, lw=1.2, zorder=1)
    ax.scatter(x, sz, color=[col for _, _, col in seq], s=36, zorder=2)
    ax.set_ylim(0.100, 0.122)
    ax.set_xticks(x, [s for s, _, _ in seq], fontsize=8)
    ax.set_title("old-val drive σz rel error (gate ≤ 0.112)", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    # train vs held-out pair R2: gap closing
    ax = axes[4]
    items = [("B2", "top-up table", tu["B2 best"], BLUE), ("B2-cov", "top-up table", tu["B2-cov best"], AQUA),
             ("C2", "top-up table", c["C2 (w_pair 10) best"], INK2),
             ("B2", "cov table", cov["B2 best"], BLUE), ("B2-cov", "cov table", cov["B2-cov best"], AQUA)]
    for j, (nm, tb, m, col) in enumerate(items):
        tr, ho = m["pairs_train"]["R2"], m["pairs_heldout"]["R2"]
        ax.plot([j, j], [ho, tr], color=col, lw=2, solid_capstyle="round")
        ax.scatter([j], [tr], marker="o", s=34, color=col, zorder=3)
        ax.scatter([j], [ho], marker="o", s=34, facecolor=SURF, ec=col, lw=1.4, zorder=3)
        ax.text(j + 0.14, (tr + ho) / 2, f"Δ{tr - ho:+.2f}", fontsize=7.5, color=INK2, va="center")
    ax.axvline(2.5, color=GRID, lw=1)
    ax.text(1, 0.955, "top-up table", ha="center", fontsize=7.5, color=INK2)
    ax.text(3.5, 0.955, "cov table", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=7.5)
    ax.set_ylim(0.80, 0.97)
    ax.set_xlim(-0.5, len(items) - 0.3)
    ax.scatter([], [], marker="o", color=INK2, s=30, label="train pairs")
    ax.scatter([], [], marker="o", facecolor=SURF, ec=INK2, s=30, label="held-out pairs")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("pair R²: train vs held-out", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("Surrogate progression on the same 93 val pairs (10-16 top-up table): coverage data nudges B2\n"
                 "(R² 0.81→0.83, p90 8.8→8.6 µm) but slope stays 0.80; B2-cov's train/held-out gap is 0.01 (C2: 0.05), "
                 "so the residual is bias", fontsize=9.5, x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.012, "ftc_s0 = centroid fine-tune; A = +pair loss; B2 = +top-up data, w_pair 3; C2 = w_pair 10; "
             "B2-cov = B2 recipe + coverage data. Shaded orange = gate band. Panel 5: filled = train "
             "pairs, open = held-out (val+test) pairs; cov table has 1190 train / 307 held-out pairs, top-up table "
             "794 / 210.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    fig.savefig(OUT / "2026-10-17_surrogate_progression.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- 4. pair parity
def pair_predictions(n=2048, cache=None):
    if cache and Path(cache).exists():
        return dict(np.load(cache, allow_pickle=True))
    import torch
    from twobunch_s2e_rl.artifacts import resolve_ckpt
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    from twobunch_s2e_rl.surrogate.properties import inter_bunch
    models = json.loads(Path("results/surrogate_scorecard_2026-10-17/models_cov.json").read_text())
    models = {k: models[k] for k in ("B2 best", "B2-cov best")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(next(iter(models.values())))), map_location="cpu")
    scal = {k: getattr(ref, k).double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    dm, role = datamodule(scaler={k: v.tolist() for k, v in scal.items()})
    ds, t = dm.full, table_meta()
    is_inj = np.isin(t["campaign"], [t["names"].index(c) for c in CAMPS])
    both = ds.drive_density.numpy() & ds.witness_density.numpy()
    pairs = [p for p in pairs_of(t, is_inj & both) if role[p[0]] != "train"]
    rows = np.unique(np.array(pairs).ravel())
    pos = {r: i for i, r in enumerate(rows)}
    d = ds.drive[rows].double() * scal["drive_std"] + scal["drive_mean"]
    w = ds.witness[rows].double() * scal["witness_std"] + scal["witness_mean"]
    true_sp = inter_bunch(d, w)["bunch_spacing"].numpy()
    out = {"true": np.array([1e6 * (true_sp[pos[a]] - true_sp[pos[b]]) for a, b in pairs]),
           "role": np.array([role[a] for a, _ in pairs]), "camp": np.array([t["campaign"][a] for a, _ in pairs])}
    for name, ck in models.items():
        m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(ck)), map_location=dev).eval()
        g = torch.Generator().manual_seed(0)  # same CRN as surrogate_scorecard.observables
        zd = torch.randn(n, m.latent_dim, generator=g).to(dev)
        zw = torch.randn(n, m.latent_dim, generator=g).to(dev)
        sp = []
        with torch.no_grad():
            for b in range(0, len(rows), 64):
                o = m.observables(ds.knobs[rows[b:b + 64]].to(dev), n=n, z_drive=zd, z_witness=zw)
                sp.append(o["bunch_spacing"].double().cpu())
        sp = torch.cat(sp).numpy()
        out[name] = np.array([1e6 * (sp[pos[a]] - sp[pos[b]]) for a, b in pairs])
        del m
        torch.cuda.empty_cache()
    if cache:
        np.savez(cache, **out)
    return out


def parity(cache=None):
    P = pair_predictions(cache=cache)
    t_, p2, pc = P["true"], P["B2 best"], P["B2-cov best"]
    r2 = lambda p, t: 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
    st = {}
    for nm, p in (("B2", p2), ("B2-cov", pc)):
        st[nm] = (r2(p, t_), np.polyfit(t_, p, 1), np.percentile(np.abs(p - t_), 90))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    lim = np.percentile(np.abs(np.r_[t_, pc]), 99.5) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--", zorder=1, label="1:1")
    ax.scatter(t_, p2, s=12, color=BLUE, alpha=0.30, lw=0, zorder=2,
               label=f"B2  (R² {st['B2'][0]:.2f}, slope {st['B2'][1][0]:.2f}, p90 {st['B2'][2]:.1f} µm)")
    mk = {"val": "o", "test": "s"}
    for r in ("val", "test"):
        s = P["role"] == r
        ax.scatter(t_[s], pc[s], s=20, marker=mk[r], color=AQUA, ec=SURF, lw=0.5, zorder=3,
                   label=f"B2-cov, {r} pairs (n={s.sum()})")
    xx = np.array([-lim, lim])
    for nm, col in (("B2", BLUE), ("B2-cov", AQUA)):
        ax.plot(xx, np.polyval(st[nm][1], xx), color=col, lw=2, zorder=4,
                label=f"{nm} fit: slope {st[nm][1][0]:.2f}" + ("  (under B2-cov line)" if nm == "B2" else "") + (f"  (R² {st[nm][0]:.2f}, p90 {st[nm][2]:.1f} µm)"
                                                                if nm == "B2-cov" else ""))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
           xlabel="Bmad Δspacing = point − golden at identical linac knobs [µm]",
           ylabel="surrogate Δspacing [µm]")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(f"Held-out pairs (val+test, n={len(t_)}), cov table", fontsize=9.5, loc="left")
    # residual vs truth: the bias is proportional (under-prediction of large effects)
    ax = axes[1]
    edges = np.percentile(t_, np.linspace(0, 100, 9))
    mid, med, q1, q3 = [], [], [], []
    res = pc - t_
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t_ >= a) & (t_ <= b)
        mid.append(np.median(t_[s]))
        med.append(np.median(res[s]))
        q1.append(np.percentile(res[s], 25))
        q3.append(np.percentile(res[s], 75))
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.scatter(t_, res, s=10, color=AQUA, alpha=0.35, lw=0)
    ax.fill_between(mid, q1, q3, color=AQUA, alpha=0.18, lw=0, label="IQR per octile of Bmad Δ")
    ax.plot(mid, med, "o-", color=INK, ms=4, lw=1.4, label="median residual")
    sl = st["B2-cov"][1][0]
    ax.plot(xx, (sl - 1) * xx + st["B2-cov"][1][1], color=AQUA, lw=1.5, ls=":",
            label=f"(slope − 1)·Δ = {sl - 1:+.2f}·Δ")
    ax.set(xlim=(-lim, lim), xlabel="Bmad Δspacing [µm]", ylabel="B2-cov residual (pred − Bmad) [µm]")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Residual grows with the effect: proportional under-prediction", fontsize=9.5, loc="left")
    fig.suptitle(f"Injector-induced spacing change at fixed linac: B2-cov under-predicts large effects by "
                 f"~{100 * (1 - sl):.0f}% (slope {sl:.2f}; gate 0.85–1.15)", fontsize=10, x=0.01, ha="left",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_pair_parity.png", dpi=110)
    plt.close(fig)
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="npz cache for the parity predictions")
    ap.add_argument("--only", default=None, help="comma list of coverage,growth,progression,parity")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else {"coverage", "growth", "progression", "parity"}
    if "progression" in only:
        progression()
    if only & {"coverage", "growth"}:
        _, role = datamodule()
        if "coverage" in only:
            coverage(role)
        if "growth" in only:
            print(json.dumps({k: {**v, "split": dict(v["split"])} for k, v in growth(role).items()}))
    if "parity" in only:
        print({k: (round(v[0], 3), round(v[1][0], 3), round(v[2], 2)) for k, v in parity(a.cache).items()})
