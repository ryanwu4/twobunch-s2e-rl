"""Per-step BLEN_LI14_888_BRAW medians of the replay scans -> results/daq_replay_2026-10-17/blen14_steps.json
(input of rl/daq_replay.py for BLEN-obs policies). Read-only on the DAQ .mat files (harness-audit loader)."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/harness_audit_2026-10-14/scripts"))
from common import BASE, load_mat  # noqa: E402

out = {}
for r in ["TEST_15668", "E300_15673"]:
    d = load_mat(r)
    p = scipy.io.loadmat(BASE / r / f"{r}.mat", simplify_cells=True)["data_struct"]["params"]
    vals = np.atleast_1d(p["scanVals"]).astype(float)
    st, b = d["step"], d["pv"]["BLEN_LI14_888_BRAW"]
    u = np.unique(st)
    out[r] = {"scan_val": [float(vals[s - 1]) for s in u], "blen14_median": [float(np.nanmedian(b[st == s])) for s in u]}
dst = Path("results/daq_replay_2026-10-17/blen14_steps.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(dst)
