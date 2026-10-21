"""Full-particle truths per table row (2026-10-21, reviewer item 1): the table's 1024/4096-particle subsamples carry ~2.5 /
1.2 um rms shot noise on a row's spacing, which inflates the pair / spacing targets and val metrics. From each row's raw
XTCAVF file (all live macroparticles, drive / witness by weight): per-bunch charge-weighted centroids (6, physical units),
sigma_z (std z), and bunch_spacing (drive z centroid - witness z centroid, as properties.inter_bunch).
Output processed/<table stem>_truth.h5, row-aligned with the table (NaN where a bunch is absent / unreadable).

  PYTHONPATH=src python scripts/build_truth_table_2026-10-21.py [--table processed/xtcav23e_inj_cov_2026-10-17.h5] [--workers 48]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np


def one(path):
    from pmd_beamphysics import ParticleGroup
    out = np.full(15, np.nan)
    try:
        P = ParticleGroup(path)
        w = np.unique(P.weight)
        for j, sel in enumerate((P.weight == w[-1], P.weight == w[0] if len(w) > 1 else None)):
            if sel is None or sel.sum() < 2:
                continue
            Q = P[sel]
            c = np.array([Q.x.mean(), Q.y.mean(), Q.z.mean(), Q.px.mean(), Q.py.mean(), Q.pz.mean()])
            out[6 * j:6 * j + 6] = c
            out[12 + j] = Q.z.std()
        out[14] = out[2] - out[8]
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="processed/xtcav23e_inj_cov_2026-10-17.h5")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    with h5py.File(a.table) as h:
        names = h.attrs["campaigns"].split(",")
        camp, idx = h["campaign"][...], h["idx"][...]
    paths = [f"data/{names[c]}/sample_{i:05d}_XTCAVF.h5" for c, i in zip(camp, idx)]
    with Pool(a.workers) as pool:
        res = np.array(pool.map(one, paths, chunksize=16))
    out = Path(a.table).with_name(Path(a.table).stem + "_truth.h5")
    with h5py.File(out, "w") as h:
        h["drive_centroid"] = res[:, :6]
        h["witness_centroid"] = res[:, 6:12]
        h["drive_sigma_z"] = res[:, 12]
        h["witness_sigma_z"] = res[:, 13]
        h["bunch_spacing"] = res[:, 14]
        h.attrs["table"] = a.table
        h.attrs["source"] = "raw XTCAVF files, all live macroparticles (scripts/build_truth_table_2026-10-21.py)"
    print(f"wrote {out}: {np.isfinite(res[:, 14]).sum()} / {len(res)} rows with a spacing")


if __name__ == "__main__":
    main()
