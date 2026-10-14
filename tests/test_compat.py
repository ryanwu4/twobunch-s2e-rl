"""Backward-compatibility guards (phase 0 of docs/architecture_plan.md).

Every existing artifact must keep loading and reproduce the outputs frozen in tests/golden/ by the
pre-refactor code. A failure here is a compatibility break: fix the code, never regenerate the
goldens to make it pass. Artifacts are gitignored, so each test skips when its file is absent.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import compat_artifacts as ca  # noqa: E402

RTOL, ATOL = 1e-5, 1e-9


def _golden(kind, name):
    p = ca.golden_path(kind, name)
    if not p.exists():
        pytest.skip(f"no golden for {kind}/{name}")
    return dict(np.load(p))


def _assert_same(got, want):
    assert set(got) == set(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=RTOL, atol=ATOL, err_msg=k)


@pytest.mark.parametrize("name", list(ca.CKPTS))
def test_flow_ckpt_reproduces(name):
    if not ca.have(ca.CKPTS[name]):
        pytest.skip("checkpoint not on disk")
    _assert_same(ca.flow_probe(ca.load_flow(name)), _golden("flow", name))


@pytest.mark.parametrize("run", list(ca.POLICY_FLOW))
def test_policy_pickle_reproduces(run):
    if not ca.have(f"logs/{run}/best_policy.pt"):
        pytest.skip("policy not on disk")
    _assert_same(ca.policy_probe(*ca.load_policy(run)), _golden("policy", run))


@pytest.mark.parametrize("run", list(ca.POLICY_FLOW))
def test_policy_env_reproduces(run, tmp_path):
    """Obs layout, reward spec, norms, knob box, and goals as rebuilt from the run's cfg.yaml."""
    if not ca.have(f"logs/{run}/cfg.yaml", ca.CKPTS[ca.POLICY_FLOW[run]]):
        pytest.skip("run or flow not on disk")
    got = ca.env_probe(run, tmp_path)
    if "cache_unchanged" in got:
        assert bool(got["cache_unchanged"]), "reward-norms cache was recomputed (provenance drift)"
    _assert_same(got, _golden("env", run))


@pytest.mark.parametrize("name", list(ca.SAMPLES))
def test_sample_json_parses(name):
    if not ca.have(ca.SAMPLES[name]):
        pytest.skip("sample not on disk")
    _assert_same(ca.sample_probe(name), _golden("sample", name))


@pytest.mark.parametrize("name", ca.NORM_JSONS)
def test_norm_json_loads(name):
    rel = f"processed/{name}_norm.json"
    if not ca.have(rel):
        pytest.skip("norm json not on disk")
    norm = json.loads(ca.path(rel).read_text())
    for k in ("knob_keys", "knob_low", "knob_high", "drive_mean", "drive_std",
              "witness_mean", "witness_std"):
        assert k in norm, k
    assert len(norm["knob_keys"]) == len(norm["knob_low"]) == len(norm["knob_high"])


@pytest.mark.parametrize("name", ["twobunch_flow_v2", "twobunch_flow_v4", "twobunch_combined",
                                  "xtcav23_combined"])
def test_processed_h5_schema(name):
    h5py = pytest.importorskip("h5py")
    rel = f"processed/{name}.h5"
    if not ca.have(rel):
        pytest.skip("processed h5 not on disk")
    norm = json.loads(ca.path(f"processed/{name}_norm.json").read_text())
    with h5py.File(ca.path(rel), "r") as h:
        n, k = h["knobs"].shape
        assert k == len(norm["knob_keys"])
        for col in ("drive_parts", "witness_parts", "drive_present", "witness_viable",
                    "drive_density", "witness_density", "drive_frac", "witness_frac", "idx"):
            assert h[col].shape[0] == n, col


@pytest.mark.parametrize("name", ca.NORMS_CACHES)
def test_norms_cache_loads(name):
    from twobunch_s2e_rl.rl.reward import load_norms
    rel = f"results/rl/_shared/norms/{name}.json"
    if not ca.have(rel):
        pytest.skip("norms cache not on disk")
    prov, norms, floors = load_norms(str(ca.path(rel)))
    assert prov and norms and floors


def test_ckpt_frame_matches_its_processed_norm():
    """A ckpt's own knob_low/high buffers are its frame; they must equal its processed norm json."""
    pairs = {"twobunch_flow_v4": "twobunch_flow_v4", "twobunch_combined_ft": "twobunch_combined",
             "xtcav23": "xtcav23_combined"}
    checked = 0
    for ck, nj in pairs.items():
        if not ca.have(ca.CKPTS[ck], f"processed/{nj}_norm.json"):
            continue
        flow = ca.load_flow(ck)
        norm = json.loads(ca.path(f"processed/{nj}_norm.json").read_text())
        np.testing.assert_allclose(flow.knob_low.numpy(), norm["knob_low"], rtol=1e-6)
        np.testing.assert_allclose(flow.knob_high.numpy(), norm["knob_high"], rtol=1e-6)
        checked += 1
    if not checked:
        pytest.skip("no ckpt/norm pairs on disk")


@pytest.mark.parametrize("name", ["twobunch_combined", "xtcav23_combined", "xtcav23_pilot"])
def test_norm_frame_equals_named_sweep_set(name):
    """Catches in-place edits of a knob table that a campaign already used (append-only rule)."""
    from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
    rel = f"processed/{name}_norm.json"
    if not ca.have(rel):
        pytest.skip("norm json not on disk")
    norm = json.loads(ca.path(rel).read_text())
    keys, lo, hi, _ = resolve_sweep_set(norm["sweep_set"])
    assert keys == norm["knob_keys"]
    np.testing.assert_allclose(lo, norm["knob_low"], rtol=1e-9)
    np.testing.assert_allclose(hi, norm["knob_high"], rtol=1e-9)


def test_8knob_frame_is_train_time():
    """The 8-knob artifacts were built with L2PhaseSet in [-38, -34] (before the in-place widening
    to -40 in fafad75). Their stored frame is authoritative."""
    rel = "processed/twobunch_flow_v4_norm.json"
    if not ca.have(rel):
        pytest.skip("norm json not on disk")
    norm = json.loads(ca.path(rel).read_text())
    i = norm["knob_keys"].index("L2PhaseSet")
    assert (norm["knob_low"][i], norm["knob_high"][i]) == (-38.0, -34.0)


@pytest.mark.xfail(strict=True, reason="L2 frame bug: rf_drift_std_vector reads today's table "
                   "(L2 width 6 deg) instead of the train-time frame (4 deg); fixed in phase 1")
def test_dr_drift_std_is_train_time():
    from twobunch_s2e_rl.rl.diff_env import rf_drift_std_vector
    std = rf_drift_std_vector(0.5, 0.005)
    assert abs(float(std[1]) - 0.125) < 1e-6
