"""Do the FF optimal setpoints change between defaults.yml and the two-bunch baseline?

This isolates the one config difference that *could* matter for the design-optics FF match:
the two-bunch config (2024-10-14_twoBunch.yml) overrides the 6 injector matching quads
(QA/QE), which reshape the Twiss launched downstream and hence the FF-incoming Twiss.

We apply each config, read the baseline PENT Twiss (with golden FF quads), and re-solve the
FF quads for beta = 0.50 m and 0.15 m. If the FF-incoming Twiss is the same, the solved
setpoints are the same.
"""

import numpy as np
from scipy.optimize import least_squares
import FACET2_S2E as qs
from twobunch_s2e_rl.datagen.paths import facet2_root

REPO = str(facet2_root())   # FACET2-S2E checkout ($FACET2_S2E_ROOT or the installed FACET2_S2E)
TWOBUNCH_CFG = "setLattice_configs/2024-10-14_twoBunch.yml"
FF_QUADS = ["Q5FF", "Q4FF", "Q3FF", "Q2FF", "Q1FF", "Q0FF"]
INJ_QUADS = ["QA10361", "QA10371", "QE10425", "QE10441", "QE10511", "QE10525"]
BOUND_LO = np.array([-256., -446., 0., 0., -257., 0.])
BOUND_HI = np.array([0., 0., 457., 167., 0., 167.])


def set_quads(tao, x):
    tao.cmd("set global lattice_calc_on = F")
    for q, v in zip(FF_QUADS, x):
        qs.setQuadkG(tao, q, float(v))
    tao.cmd("set global lattice_calc_on = T")


def pent_twiss(tao, s_offset=0.0):
    t = tao.twiss_at_s(ele="PENT", s_offset=s_offset)
    return t["beta_a"], t["alpha_a"], t["beta_b"], t["alpha_b"]


def solve(tao, tb, x0):
    def resid(x):
        set_quads(tao, x)
        t = tao.twiss_at_s(ele="PENT", s_offset=0.0)
        return [t["beta_a"] - tb, t["beta_b"] - tb, 0.1 * t["alpha_a"], 0.1 * t["alpha_b"]]
    r = least_squares(resid, np.clip(x0, BOUND_LO, BOUND_HI), bounds=(BOUND_LO, BOUND_HI),
                      xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=300)
    set_quads(tao, r.x)
    bx, ax, by, ay = pent_twiss(tao)
    return r.x, (bx, by)


def snapshot(tao, label, golden_ff):
    print("\n" + "=" * 78 + f"\n{label}")
    inj = {q: qs.getQuadkG(tao, q) for q in INJ_QUADS}
    print("  injector quads (kG.m): " + ", ".join(f"{q}={inj[q]:.2f}" for q in INJ_QUADS))
    set_quads(tao, golden_ff)                      # ensure golden FF before reading baseline
    bx, ax, by, ay = pent_twiss(tao)
    print(f"  baseline PENT Twiss (golden FF): beta_x={bx:.4f} m, beta_y={by:.4f} m, "
          f"alpha_x={ax:.3f}, alpha_y={ay:.3f}")
    out = {}
    for tb in (0.50, 0.15):
        x, (abx, aby) = solve(tao, tb, golden_ff)
        out[tb] = x
        print(f"  solve beta={tb:.2f} m -> achieved bx={abx:.4f} by={aby:.4f} | "
              + ", ".join(f"{q}={x[i]:.2f}" for i, q in enumerate(FF_QUADS)))
        set_quads(tao, golden_ff)                  # reset before next solve
    return out


def main():
    tao = qs.initializeTao(filePath=REPO, runSetLatticeTF=True, runImpactTF=False)
    golden_ff = np.array([qs.getQuadkG(tao, q) for q in FF_QUADS])
    print("Golden FF quads (kG.m): " + ", ".join(f"{q}={golden_ff[i]:.3f}" for i, q in enumerate(FF_QUADS)))

    res_def = snapshot(tao, "CONFIG A: defaults.yml (as loaded by initializeTao)", golden_ff)

    merged = qs.loadConfig(TWOBUNCH_CFG, REPO)
    qs.setLattice(tao, **merged)                   # apply two-bunch baseline (overrides injector quads)
    res_2b = snapshot(tao, "CONFIG B: 2024-10-14_twoBunch.yml", golden_ff)

    print("\n" + "=" * 78 + "\nDIFFERENCE in solved FF setpoints (two-bunch minus defaults), kG.m:")
    for tb in (0.50, 0.15):
        d = res_2b[tb] - res_def[tb]
        print(f"  beta={tb:.2f} m: " + ", ".join(f"{q}={d[i]:+.2f}" for i, q in enumerate(FF_QUADS))
              + f"  | max|diff|={np.max(np.abs(d)):.2f}")


if __name__ == "__main__":
    main()
