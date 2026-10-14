"""preprocess: embedding campaigns into a bigger knob space via campaign.json, the in-frame and
knob-drop rules, merge checks, and the campaign column. Synthetic campaigns (no beams: every row
is feasibility-only, which is all these rules need)."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pmd_beamphysics")
from twobunch_s2e_rl.datagen.sweep_params import EXPANDED_PARAMS, resolve_sweep_set  # noqa: E402
from twobunch_s2e_rl.surrogate.preprocess import preprocess  # noqa: E402

GOLDEN = {k: v[2] for k, v in EXPANDED_PARAMS.items()}


def _campaign(root, name, sweep_set, n=3, fixed_overrides=None, csr=True, write_json=True, seed=0):
    d = root / name
    d.mkdir()
    keys, lo, hi, _ = resolve_sweep_set(sweep_set)
    rng = np.random.default_rng(seed)
    for i in range(n):
        knobs = {k: float(rng.uniform(a, b)) for k, a, b in zip(keys, lo, hi)}
        (d / f"sample_{i:05d}.json").write_text(json.dumps(
            {"idx": i, "knobs": knobs, "success": True, "specs": {"PENT": {}, "XTCAVF": {}}}))
    if write_json:
        camp = {"name": name, "swept": {"sweep_set": sweep_set, "knob_keys": keys},
                "fixed": {"baseline_settings": dict(GOLDEN),
                          "fixed_overrides": dict(fixed_overrides or {}),
                          "baseline_config_sha256": "abc", "input_beam_sha256": "def",
                          "csrTF": csr, "transverseWakes": True, "num_macro_particles": 100000}}
        (d / "campaign.json").write_text(json.dumps(camp))
    return d


def _run(root, subdir, sweep_set, treaty="PENT"):
    out = root / "out.h5"
    preprocess(subdir=subdir, p=16, sweep_set=sweep_set, treaty_point=treaty, out=str(out),
               verbose=False, data_root=root)
    with h5py.File(out) as h:
        return h["knobs"][...], h["campaign"][...], h.attrs["campaigns"]


def test_8knob_campaign_embeds_into_26knob_union_frame(tmp_path):
    _campaign(tmp_path, "c8", "original8")
    _campaign(tmp_path, "c26", "tightbox", seed=1)
    K, camp, names = _run(tmp_path, "c8,c26", "tightbox+expanded")
    keys = resolve_sweep_set("tightbox+expanded")[0]
    q0 = keys.index("Q0FFkG")
    np.testing.assert_allclose(K[camp == 0, q0], GOLDEN["Q0FFkG"], rtol=1e-6)   # filled from fixed
    assert list(camp) == [0, 0, 0, 1, 1, 1] and names == "c8,c26"


def test_fill_outside_target_frame_refused(tmp_path):
    _campaign(tmp_path, "c8", "original8")             # golden Q0FF 126.35 is outside tightbox
    with pytest.raises(ValueError, match="outside the target frame"):
        _run(tmp_path, "c8", "tightbox")


def test_knob_drop_rule(tmp_path):
    _campaign(tmp_path, "c26", "expanded")
    with pytest.raises(ValueError, match="not downstream"):   # drops Q2/Q1/Q0FF at PENT
        _run(tmp_path, "c26", "xtcav23", treaty="PENT")
    # at XTCAVF the dropped FF quads are downstream -> allowed (other dropped? none)
    K, _, _ = _run(tmp_path, "c26", "xtcav23", treaty="XTCAVF")
    assert K.shape == (3, 23)


def test_merge_checks(tmp_path):
    _campaign(tmp_path, "a", "tightbox", fixed_overrides={"Q19851kG": -63.76, "trackEnd": "PENT"})
    _campaign(tmp_path, "b", "tightbox", fixed_overrides={"Q19851kG": -64.10}, seed=1)
    with pytest.raises(ValueError, match="Q19851kG"):
        _run(tmp_path, "a,b", "tightbox")
    _campaign(tmp_path, "c", "tightbox", fixed_overrides={"Q19851kG": -63.76}, seed=2)
    _run(tmp_path, "a,c", "tightbox")                  # trackEnd is downstream-only: allowed
    _campaign(tmp_path, "d", "tightbox", fixed_overrides={"Q19851kG": -63.76}, csr=False, seed=3)
    with pytest.raises(ValueError, match="csrTF"):
        _run(tmp_path, "a,d", "tightbox")


def test_campaign_without_json_parses_as_before(tmp_path):
    _campaign(tmp_path, "old8", "original8", write_json=False)
    K, camp, _ = _run(tmp_path, "old8", "original8")
    assert K.shape == (3, 8) and (camp == 0).all()
    with pytest.raises(KeyError):                      # no campaign.json -> no silent fill
        _run(tmp_path, "old8", "expanded")
