"""Figures for the beam-based matching solve (target beta* = 15 cm and 50 cm).

Headline: the witness IS matchable on the real two-bunch beam (BMAG 2.36 -> 1.01 at 15 cm,
full transmission) -- with the FF quads essentially at golden and the match carried by small
BC20 sextupole (chromatic-correction) changes. This is the opposite of the design-optics
curve, which commanded 12-35% FF moves to the wrong place. Three figures:
  1. convergence (witness BMAG vs tracking eval) for both targets
  2. knob displacement: beam-match (tiny FF, sextupole-led) vs the design-curve FF prescription
  3. Tao validation: per-slice beta profile at PENT, golden vs matched-15cm (witness flattened)

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.matching_result_figs
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datagen.paths import facet2_root
from ..datagen.ff_manifold import FF_MATCHED_CURVE
from ..surrogate.properties import _trace_twiss

FIG = "/home/rwu4/photoinjector-rl/lab-notebook/images"
COORD = ["x", "y", "z", "px", "py", "pz"]
FF = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]
SEXT = ["S1ELkG", "S2ELkG", "S3ELkG"]

GOLDEN = dict(Q5FFkG=-71.837, Q4FFkG=-81.251, Q3FFkG=99.225, Q2FFkG=126.35,
              Q1FFkG=-235.218, Q0FFkG=126.353, S1ELkG=2089.4846, S2ELkG=-3954.374, S3ELkG=-1087.957)
M15 = dict(Q5FFkG=-72.52, Q4FFkG=-80.79, Q3FFkG=97.84, Q2FFkG=125.4, Q1FFkG=-233.4,
           Q0FFkG=127.7, S1ELkG=2058., S2ELkG=-4120., S3ELkG=-1121.)
M50 = dict(Q5FFkG=-62.29, Q4FFkG=-78.68, Q3FFkG=107.3, Q2FFkG=115.0, Q1FFkG=-233.5,
           Q0FFkG=116.7, S1ELkG=1936., S2ELkG=-3854., S3ELkG=-1203.)

# best-so-far witness BMAG vs eval, transcribed from the two production runs
CONV15 = np.array([(1, 2.36), (10, 1.58), (18, 1.16), (33, 1.13), (55, 1.05), (66, 1.04),
                   (75, 1.01), (88, 1.01), (100, 1.00), (125, 1.01), (185, 1.01), (390, 1.01)])
CONV50 = np.array([(1, 2.34), (8, 2.21), (15, 1.82), (24, 1.72), (38, 1.64), (50, 1.54),
                   (58, 1.44), (66, 1.31), (69, 1.27), (84, 1.11), (98, 1.15), (124, 1.15),
                   (200, 1.06), (234, 1.06), (300, 1.06)])


def slice_beta_profile(parts, n_slices=9):
    B, N, _ = parts.shape
    s = N // n_slices
    order = torch.argsort(parts[..., 2], dim=1)[:, : s * n_slices]
    g = torch.gather(parts, 1, order.unsqueeze(-1).expand(-1, -1, 6)).reshape(B, n_slices, s, 6)
    pz = g[..., 5].clamp_min(1.0)
    bx, _, _ = _trace_twiss(g[..., 0], g[..., 3] / pz)
    by, _, _ = _trace_twiss(g[..., 1], g[..., 4] / pz)
    return g[..., 2].mean(-1)[0].numpy(), bx[0].numpy(), by[0].numpy()


def fig_convergence():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(CONV15[:, 0], CONV15[:, 1], "o-", color="#4c72b0", label="target 15 cm  (2.36 -> 1.01)")
    ax.plot(CONV50[:, 0], CONV50[:, 1], "s-", color="#dd8452", label="target 50 cm  (2.34 -> 1.06)")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="matched (BMAG=1)")
    ax.set_xlabel("tracking evaluation"); ax.set_ylabel("witness core-slice BMAG (vs own beta*)")
    ax.set_title("Beam-based matching converges the witness to BMAG~1 (full transmission)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = f"{FIG}/2026-06-21_beam_matching_convergence.png"
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


def fig_displacement():
    # design-optics FF prescription at beta*=15 cm (interp the curve)
    b_asc, q_asc = FF_MATCHED_CURVE[::-1, 0], FF_MATCHED_CURVE[::-1, 1:]
    design15 = {FF[j]: float(np.interp(0.15, b_asc, q_asc[:, j])) for j in range(6)}
    pct = lambda d, k: 100 * (d[k] - GOLDEN[k]) / abs(GOLDEN[k])
    x = np.arange(len(FF)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(x - w/2, [pct(M15, k) for k in FF], w, color="#4c72b0", label="beam-matched (15 cm)")
    ax[0].bar(x + w/2, [pct(design15, k) for k in FF], w, color="#c44e52", label="design-optics curve (15 cm)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([k[:-2] for k in FF]); ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("change from golden [%]"); ax[0].set_title("(a) FF quads: beam-match barely moves; design curve moves 12-35%")
    ax[0].legend()
    ax[1].bar(np.arange(len(SEXT)), [pct(M15, k) for k in SEXT], 0.5, color="#55a868")
    ax[1].set_xticks(np.arange(len(SEXT))); ax[1].set_xticklabels([k[:-2] for k in SEXT]); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("change from golden [%]"); ax[1].set_title("(b) the match is carried by BC20 sextupoles (chromatic)")
    fig.suptitle("Where the witness match lives: near golden FF + tuned sextupoles -- not the design FF curve")
    fig.tight_layout(); p = f"{FIG}/2026-06-21_beam_matching_displacement.png"
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


def fig_tao_validation():
    import FACET2_S2E as qs
    root = str(facet2_root())
    base = qs.loadConfig("/setLattice_configs/2024-10-14_twoBunch_baseline.yml", root)
    tao = qs.initializeTao(filePath=root, inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
                           csrTF=True, transverseWakes=True, numMacroParticles=50000,
                           scratchPath=root + "/tmp/matching_figs", randomizeFileNames=True)

    def profile(setting):
        merged = {**base, **setting}
        qs.setLattice(tao, **merged); qs.trackBeam(tao, root, **merged)
        d, w = qs.getDriverAndWitness(qs.getBeamAtElement(tao, "PENT"))
        tt = lambda pg: torch.tensor(np.stack([getattr(pg, k) for k in COORD], 1),
                                     dtype=torch.float64).unsqueeze(0)
        zc, _, wby = slice_beta_profile(tt(w))
        _, _, dby = slice_beta_profile(tt(d))
        return zc * 1e6, wby * 100, dby * 100   # um, cm

    zc, wby_g, dby_g = profile(GOLDEN)
    _, wby_m, dby_m = profile(M15)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(zc, wby_g, "o-", color="#c44e52", label="golden")
    ax[0].plot(zc, wby_m, "o-", color="#4c72b0", label="matched (15 cm)")
    ax[0].axhline(15, color="g", ls="--", label="target 15 cm")
    ax[0].set_yscale("log"); ax[0].set_xlabel("slice z [um]"); ax[0].set_ylabel("witness slice-beta_y [cm]")
    ax[0].set_title("(a) witness slice-beta flattened to target"); ax[0].legend()
    ax[1].plot(zc, dby_g, "o-", color="#c44e52", label="golden")
    ax[1].plot(zc, dby_m, "o-", color="#4c72b0", label="matched (15 cm)")
    ax[1].axhline(15, color="g", ls="--"); ax[1].set_yscale("log")
    ax[1].set_xlabel("slice z [um]"); ax[1].set_ylabel("drive slice-beta_y [cm]")
    ax[1].set_title("(b) drive (shares the knobs; residual mismatch)"); ax[1].legend()
    fig.suptitle("Tao validation of the matched 15 cm setting (PENT, 50k particles, wakes on)")
    fig.tight_layout(); p = f"{FIG}/2026-06-21_beam_matching_profile.png"
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_convergence()
    fig_displacement()
    try:
        fig_tao_validation()
    except Exception as e:
        print(f"Tao validation figure skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
