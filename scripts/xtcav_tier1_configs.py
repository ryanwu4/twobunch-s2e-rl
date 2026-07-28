"""Tier 1 config builder: extract per-dataset sim operating points from the DAQ .mat files.

Writes machine_config.json per dataset under
XTCAV_analysis/sim2real/results/tier1_sim_scans/{DATASET}/ containing:
  - L1 anchor (dataset comment; TEST_15668 derived from klystron PACT offsets) and
    per-step L1 phase deltas from the mean of KLYS_LI11_{11,21} FB_FAST_PACT medians
    (what the BC11 BLEN feedback actually commanded during the scan);
  - L2 scan values (relative, verified against LI12 SBST phase readback);
  - FF/spectrometer quad kG under TWO candidate PV->element mappings:
      "value_match": LGPS supplies matched to golden optics values (self-consistent:
        7 quads exactly golden, Q0D/Q2D pair trimmed -5.6%; supply numbers are rack
        positions, not s-ordered; Q0FF/Q2FF and originally Q0D/Q2D share supplies);
      "alias_file": the FACET2e_devicenames.bmad aliases (contradicts value evidence).
    The smoke run discriminates: whichever mapping reproduces the measured
    cal_scaled ~= 15.3 um/um wins.
  - B5D36 GeVc and BC20 launch quads (Q19851/Q19871) from BACTs.

Run (fast, seconds):
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    scripts/xtcav_tier1_configs.py
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

XTCAV_ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
OUT_ROOT = XTCAV_ROOT / "sim2real" / "results" / "tier1_sim_scans"

DATASETS = ["E300_15671", "TEST_15668", "E300_15673"]
L1_ANCHORS = {"E300_15671": -26.0, "E300_15673": -20.0}  # from dataset comments

# Candidate LGPS -> element mappings (values are the PV device numbers)
MAP_VALUE_MATCH = {
    "Q5FFkG": 3011, "Q4FFkG": 3311, "Q3FFkG": 3151,
    "Q2FFkG": 3031, "Q0FFkG": 3031,          # shared supply pair
    "Q1FFkG": 3204,
    "Q0DkG": 3091, "Q2DkG": 3141,            # trimmed pair (order degenerate: equal values)
    "Q1DkG": 3261,
}
MAP_ALIAS_FILE = {
    # FACET2e_devicenames.bmad; Q3FF/Q4FF/Q5FF aliases absent -> reuse value-match there
    "Q5FFkG": 3011, "Q4FFkG": 3031, "Q3FFkG": 3061,   # 3061 not recorded -> golden fallback
    "Q2FFkG": 3091, "Q1FFkG": 3141, "Q0FFkG": 3151,
    "Q0DkG": 3204, "Q1DkG": 3261, "Q2DkG": 3311,
}
GOLDEN = {  # defaults.yml, for fallbacks and deltas report
    "Q5FFkG": -71.837, "Q4FFkG": -81.251, "Q3FFkG": 99.225, "Q2FFkG": 126.35,
    "Q1FFkG": -235.218, "Q0FFkG": 126.353, "Q0DkG": -109.694, "Q1DkG": 180.862,
    "Q2DkG": -109.694,
}


def med(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def main():
    # First pass: dataset-level L1 PACT medians (for the TEST_15668 anchor)
    pact_ds = {}
    raw = {}
    for name in DATASETS:
        m = sio.loadmat(str(XTCAV_ROOT / name / f"{name}.mat"), simplify_cells=True)
        d = m["data_struct"]
        raw[name] = d
        s11rf = d["scalars"]["BSA_List_S11RF"]
        pact_ds[name] = np.nanmean([med(s11rf["KLYS_LI11_11_FB_FAST_PACT"]),
                                    med(s11rf["KLYS_LI11_21_FB_FAST_PACT"])])
    # TEST_15668 anchor: offset from the mean of the two anchored datasets
    ref = np.mean([pact_ds[n] - L1_ANCHORS[n] for n in L1_ANCHORS])
    anchors = dict(L1_ANCHORS)
    anchors["TEST_15668"] = float(pact_ds["TEST_15668"] - ref)
    print(f"L1 PACT medians: { {k: round(v,3) for k,v in pact_ds.items()} }")
    print(f"TEST_15668 derived L1 anchor: {anchors['TEST_15668']:.2f} deg")

    for name in DATASETS:
        d = raw[name]
        scal = d["scalars"]
        steps = np.asarray(scal["steps"], int)
        scan_vals = np.asarray(d["params"]["scanVals"], float)
        s11rf = scal["BSA_List_S11RF"]
        pact = np.nanmean([np.asarray(s11rf["KLYS_LI11_11_FB_FAST_PACT"], float),
                           np.asarray(s11rf["KLYS_LI11_21_FB_FAST_PACT"], float)],
                          axis=0)
        l1_delta = {int(s): float(np.nanmedian(pact[steps == s]) - pact_ds[name])
                    for s in np.unique(steps)}

        s20m = scal["nonBSA_List_S20Magnets"]
        lgps = {int(k.split("_")[2]): med(v) for k, v in s20m.items()
                if k.startswith("LI20_LGPS_")}
        quad_maps = {}
        for tag, mapping in [("value_match", MAP_VALUE_MATCH),
                             ("alias_file", MAP_ALIAS_FILE)]:
            vals = {}
            for ele, dev in mapping.items():
                v = lgps.get(dev)
                if v is None or not np.isfinite(v):
                    vals[ele] = {"kG": GOLDEN[ele], "source": f"golden (LGPS {dev} missing)"}
                else:
                    vals[ele] = {"kG": v, "source": f"LI20_LGPS_{dev}_BACT",
                                 "delta_vs_golden_pct":
                                     round(100 * (v - GOLDEN[ele]) / abs(GOLDEN[ele]), 2)}
            quad_maps[tag] = vals

        s19 = scal["nonBSA_List_S19"]
        cfg = {
            "dataset": name,
            "comment": str(d["params"]["comment"]),
            "L1_anchor_deg": anchors[name],
            "L1_delta_per_step_deg": l1_delta,
            "L2_scan_vals_deg": scan_vals.tolist(),
            "B5D36GeVc": lgps.get(3330),
            "Q19851kG": med(s19["QUAD_LI19_851_BACT"]),
            "Q19871kG": med(s19["QUAD_LI19_871_BACT"]),
            "quad_mappings": quad_maps,
            "xtcav_voltage_V": 22e6,
        }
        out = OUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "machine_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        dmax = max(abs(v) for v in l1_delta.values())
        print(f"{name}: L1 anchor {anchors[name]:.2f} deg, max |L1 step delta| "
              f"{dmax:.3f} deg, B5D36 {cfg['B5D36GeVc']:.3f} GeV/c -> "
              f"{out / 'machine_config.json'}")


if __name__ == "__main__":
    main()
