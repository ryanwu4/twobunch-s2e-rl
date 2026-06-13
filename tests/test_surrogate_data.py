"""Physics-contract tests against the real campaign data (skipped if absent).

Pin the surrogate's beam-physics kernels to ground truth: emittance vs ParticleGroup,
the bunch-spacing sign vs the json convention, the drive=higher-weight split, and the
dataset (de)standardization round-trip. Read-only; a few real h5 files each.
"""
import glob
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pmd_beamphysics")
from pmd_beamphysics import ParticleGroup  # noqa: E402

from twobunch_s2e_rl.datagen.paths import repo_root  # noqa: E402

DATA = repo_root() / "data" / "full"
PROC = repo_root() / "processed" / "twobunch_flow.h5"
COORD = ("x", "y", "z", "px", "py", "pz")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="no campaign data/full")


def _split(P):
    w = np.unique(P.weight)
    if len(w) >= 2:
        return P[P.weight == w[-1]], P[P.weight == w[0]]  # higher=drive, lower=witness
    return P, None


def _coords(pg):
    return torch.tensor(np.stack([getattr(pg, k) for k in COORD], axis=1)[None]).double()


def test_norm_emits_match_particlegroup():
    from twobunch_s2e_rl.surrogate.properties import norm_emits
    files = sorted(glob.glob(str(DATA / "sample_*_PENT.h5")))[:3]
    assert files, "no PENT h5 files"
    checked = 0
    for f in files:
        P = ParticleGroup(f)
        for pg in _split(P):
            if pg is None or len(pg.x) < 100:
                continue
            ne = norm_emits(_coords(pg))
            assert abs(ne["norm_emit_x"].item() - pg.norm_emit_x) / pg.norm_emit_x < 1e-5
            assert abs(ne["norm_emit_4d"].item() - pg.norm_emit_4d) / pg.norm_emit_4d < 1e-5
            checked += 1
    assert checked >= 2


def test_bunch_spacing_sign_matches_json():
    from twobunch_s2e_rl.surrogate.properties import inter_bunch
    for jf in sorted(glob.glob(str(DATA / "sample_*.json")))[:60]:
        spec = json.load(open(jf))["specs"]["PENT"]
        if spec.get("PWitness_norm_emit_x") is None:
            continue
        dr, wi = _split(ParticleGroup(jf.replace(".json", "_PENT.h5")))
        if wi is None:
            continue
        ib = inter_bunch(_coords(dr), _coords(wi))
        js = spec["bunchSpacing"]
        pred = ib["bunch_spacing"].item()
        assert np.sign(pred) == np.sign(js), f"sign flip: pred {pred:.2e} vs json {js:.2e}"
        assert abs(abs(pred) - abs(js)) / abs(js) < 0.25
        return
    pytest.skip("no viable witness found in first 60 samples")


def test_split_drive_is_higher_charge():
    for jf in sorted(glob.glob(str(DATA / "sample_*.json")))[:30]:
        spec = json.load(open(jf))["specs"]["PENT"]
        if spec.get("PDrive_charge_nC") is None or spec.get("PWitness_charge_nC") is None:
            continue
        dr, _ = _split(ParticleGroup(jf.replace(".json", "_PENT.h5")))
        qd = float(dr.charge) * 1e9  # C -> nC
        assert abs(qd - spec["PDrive_charge_nC"]) < abs(qd - spec["PWitness_charge_nC"])
        return
    pytest.skip("no two-bunch sample in first 30")


@pytest.mark.skipif(not PROC.is_file(), reason="no processed/twobunch_flow.h5")
def test_dataset_standardize_roundtrip():
    import h5py
    from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataset
    ds = TwoBunchFlowDataset(str(PROC))
    with h5py.File(str(PROC), "r") as h:
        raw = h["drive_parts"][0]
    recon = ds.drive[0].numpy() * np.array(ds.norm["drive_std"]) + np.array(ds.norm["drive_mean"])
    assert np.allclose(recon, raw, rtol=1e-4, atol=1e-6)
