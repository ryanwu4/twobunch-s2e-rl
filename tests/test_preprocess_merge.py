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


def test_per_sample_transmission_and_split_groups(tmp_path):
    """per_sample: T = charge / the row's own reference charge (no downsampling-noise ceiling);
    'max' (default) keeps the v1-v3 labels. Split groups tie an injector point's rows together."""
    d = tmp_path / "c"
    d.mkdir()
    keys, lo, hi, _ = resolve_sweep_set("tightbox")
    rows = [  # (witness charge at BEGBC20, at PENT, injector_point, block, linac_id)
        (0.398, 0.398, "lhs0", "nested", 1), (0.402, 0.402, "lhs0", "nested", 2),
        (0.400, 0.380, "lhs1", "nested", 3), (0.401, 0.401, "golden", "paired_golden", 3),
        (0.399, 0.399, "golden", "crossed", 0)]
    for i, (qb, qp, pt, blk, lid) in enumerate(rows):
        spec = lambda q: {"PDrive_charge_nC": 1.2, "PWitness_charge_nC": q}
        (d / f"sample_{i:05d}.json").write_text(json.dumps(
            {"idx": i, "knobs": {k: (a + b) / 2 for k, a, b in zip(keys, lo, hi)}, "success": True,
             "specs": {"BEGBC20": spec(qb), "PENT": spec(qp)}, "injector_point": pt, "block": blk,
             "linac_id": lid}))
    out = tmp_path / "o.h5"
    preprocess(subdir="c", p=16, sweep_set="tightbox", out=str(out), verbose=False, data_root=tmp_path,
               transmission="per_sample")
    with h5py.File(out) as h:
        wf, g = h["witness_frac"][...], h["group"][...]
    np.testing.assert_allclose(wf, [1.0, 1.0, 0.95, 1.0, 1.0], rtol=1e-6)     # no-loss rows read 1
    assert g[0] == g[1] and g[2] == g[3] and len({g[0], g[2], g[4]}) == 3
    preprocess(subdir="c", p=16, sweep_set="tightbox", out=str(out), verbose=False, data_root=tmp_path)
    with h5py.File(out) as h:
        assert h["witness_frac"][...].max() == pytest.approx(1.0) and h["witness_frac"][0] < 1.0  # max mode


def test_topup_campaign_links_pairs_and_groups_across_campaigns(tmp_path):
    """A top-up campaign's rows link to their partners in the original campaign: pair_campaign (exact pair of an
    existing nested row) shares its pair key and split group; point_campaign puts a new nested row in its
    injector point's group; pair keys are globally unique; block / linac_id / injector_golden are stored."""
    keys, lo, hi, _ = resolve_sweep_set("tightbox")
    mk = lambda **kw: {"knobs": {k: (a + b) / 2 for k, a, b in zip(keys, lo, hi)}, "success": True,
                       "specs": {"PENT": {"PDrive_charge_nC": 1.2, "PWitness_charge_nC": 0.4}}, **kw}
    camp = {"c": [dict(injector_point="lhs0", block="nested", linac_id=2),
                  dict(injector_point="lhs1", block="nested", linac_id=3),
                  dict(injector_point="golden", block="crossed", linac_id=0)],
            "t": [dict(injector_point="golden", block="paired_golden", linac_id=2, pair_campaign="c"),
                  dict(injector_point="lhs1", block="nested", linac_id=3_000_000, point_campaign="c"),
                  dict(injector_point="golden", block="paired_golden", linac_id=3_000_000),
                  dict(injector_point="golden", block="paired_golden", linac_id=3)]}   # unrelated: t's own lid 3
    for name, rows in camp.items():
        (tmp_path / name).mkdir()
        for i, r in enumerate(rows):
            (tmp_path / name / f"sample_{i:05d}.json").write_text(json.dumps(mk(idx=i, **r)))
    out = tmp_path / "o.h5"
    preprocess(subdir="c,t", p=16, sweep_set="tightbox", out=str(out), verbose=False, data_root=tmp_path)
    with h5py.File(out) as h:
        g, key, blk, lid, ig = (h[k][...] for k in ("group", "pair_key", "block", "linac_id", "injector_golden"))
    # row order: c0 c1 c2 t0 t1 t2 t3
    assert key[0] == key[3] and g[0] == g[3]                     # cross-campaign exact pair
    assert g[4] == g[1]                                          # new nested row joins lhs1's group
    assert key[4] == key[5] and g[5] == g[4]                     # new draw + its golden pair
    assert key[6] != key[1] and g[6] != g[1]                     # same raw linac_id, different campaign
    assert list(blk) == [2, 2, 1, 3, 2, 3, 3] and lid[2] == 0 and list(ig) == [0, 0, 1, 1, 0, 1, 1]
