"""The facet2-twobunch-cnf split: aliases are the package's objects, the loader dispatches correctly, the
train CLI shim translates / refuses old flags, the eval shim takes any model, widen routes by model class."""

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
twobunch_cnf = pytest.importorskip("twobunch_cnf")
import lightning  # noqa: E402

from twobunch_s2e_rl.surrogate import loading  # noqa: E402
from twobunch_s2e_rl.surrogate import train as train_shim  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow as LegacyFlow  # noqa: E402

KEYS, LO, HI = ["a", "b", "c"], [0.0, -1.0, 2.0], [1.0, 1.0, 3.0]


def test_aliases_are_the_package_objects():
    import twobunch_cnf.artifacts as ca
    import twobunch_cnf.beam_io as cb
    import twobunch_cnf.charges as cc
    import twobunch_cnf.dataset as cd
    import twobunch_cnf.knobs as ck
    import twobunch_cnf.properties as cp
    import twobunch_cnf.selection as cs
    from twobunch_s2e_rl import artifacts, knobs
    from twobunch_s2e_rl.rl import plants
    from twobunch_s2e_rl.sim import bmad
    from twobunch_s2e_rl.surrogate import dataset, preprocess, properties, selection

    assert properties is cp and selection is cs
    assert dataset.TwoBunchFlowDataModule is cd.TwoBunchFlowDataModule and dataset.group_split is cd.group_split
    assert knobs.KnobSpace is ck.KnobSpace
    assert artifacts.resolve_ckpt is ca.resolve_ckpt and artifacts.read_run_json is ca.read_run_json
    assert plants.injected_charges_nC is cc.injected_charges_nC and bmad.split_bunches is cb.split_bunches
    assert preprocess._stack is cb.stack_coords and preprocess._subsample is cb.subsample
    assert callable(dataset.orig_split_keys) and callable(dataset.reproduce_random_split)   # legacy helpers kept


def _legacy(**kw):
    torch.manual_seed(0)
    return LegacyFlow(condition_dim=3, n_layers=2, hidden_dim=16, coupling="rqs", knob_low=LO, knob_high=HI, **kw).eval()


def _save(flow, path):
    torch.save({"state_dict": flow.state_dict(), "hyper_parameters": dict(flow.hparams),
                "pytorch-lightning_version": lightning.__version__}, path)
    return path


def _probe(flow):
    g = torch.Generator().manual_seed(1)
    kn = torch.rand(3, 3, generator=g)
    z = (torch.randn(32, 6, generator=g), torch.randn(32, 6, generator=g))
    with torch.no_grad():
        return flow.observables(kn, n=32, z_drive=z[0], z_witness=z[1], generator=torch.Generator().manual_seed(2))


def test_loader_serves_the_package_model_when_the_checkpoint_converts(tmp_path):
    from twobunch_cnf.model import TwoBunchFlow as CNFFlow

    leg = _legacy(t_two_part=True, beam_latent_dim=2, knob_keys=KEYS, treaty_point="XTCAVF")
    ck = _save(leg, tmp_path / "convertible.ckpt")
    assert loading.converts(ck)
    got = loading.load_flow(ck)
    assert isinstance(got, CNFFlow) and not got.training
    a, b = _probe(leg), _probe(got)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("kw", [{"t_two_part": False, "knob_keys": KEYS, "treaty_point": "XTCAVF"},   # single sigmoid
                                {"t_two_part": True},                                                  # no knob_keys
                                {"t_two_part": True, "knob_keys": KEYS, "treaty_point": "PENT",
                                 "centroid_head": True}])                                              # removed head
def test_loader_falls_back_to_the_legacy_model(tmp_path, kw):
    ck = _save(_legacy(**kw), tmp_path / "legacy.ckpt")
    assert not loading.converts(ck)
    got = loading.load_flow(ck)
    assert type(got) is LegacyFlow and not got.training


def test_train_cli_translation():
    args, two_part = train_shim.translate_args(
        ["--t-two-part", "--coupling", "rqs", "--bunches=both", "--w-centroid", "0.0", "--curve-dims", "2", "0",
         "--centroid-prefit-steps", "3000", "--epochs", "5", "--w-sw", "2.0"])
    assert two_part and args == ["--epochs", "5", "--w-sw", "2.0"]
    for bad in (["--centroid-head"], ["--coupling", "affine"], ["--orig-split", "x.h5=a,b"], ["--bunches", "witness"],
                ["--curve-dims", "2", "1"], ["--backbone"]):
        with pytest.raises(SystemExit, match="removed"):
            train_shim.translate_args(bad)
    cfg, two_part = train_shim.translate_config({"coupling": "rqs", "bunches": "both", "t_two_part": True,
                                                 "orig_split": None, "w_curve": 0.0, "epochs": 3})
    assert two_part and cfg == {"epochs": 3}
    with pytest.raises(SystemExit, match="single-sigmoid"):
        train_shim.translate_config({"t_two_part": False})


def test_train_cli_defaults_and_single_sigmoid_refusal(monkeypatch):
    import twobunch_cnf.train as cnf_train

    seen = {}
    monkeypatch.setattr(cnf_train, "main", lambda: seen.update(argv=list(sys.argv)))
    train_shim.main(["--t-two-part", "--epochs", "1"])
    a = seen["argv"]
    assert a[a.index("--processed") + 1].endswith("processed/twobunch_flow.h5")
    assert a[a.index("--out") + 1].endswith("trained/twobunch_flow") and "--t-two-part" not in a
    with pytest.raises(SystemExit, match="single-sigmoid"):
        train_shim.main(["--epochs", "1"])


def test_eval_shim_takes_any_model(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import h5py

    from twobunch_s2e_rl.surrogate.eval import evaluate

    rng = np.random.default_rng(0)
    h5 = tmp_path / "t.h5"
    with h5py.File(h5, "w") as h:
        h["knobs"] = rng.uniform(LO, HI, size=(30, 3)).astype(np.float32)
        h["drive_parts"] = rng.normal(size=(30, 16, 6)).astype(np.float32)
        h["witness_parts"] = rng.normal(size=(30, 16, 6)).astype(np.float32)
        for k in ("drive_present", "witness_viable", "drive_density", "witness_density"):
            h[k] = np.ones(30, bool)
        h["drive_frac"] = np.ones(30, np.float32)
        h["witness_frac"] = np.ones(30, np.float32)
    (tmp_path / "t_norm.json").write_text(json.dumps({"knob_keys": KEYS, "knob_low": LO, "knob_high": HI,
                                                      "drive_mean": [0.0] * 6, "drive_std": [1.0] * 6,
                                                      "witness_mean": [0.0] * 6, "witness_std": [1.0] * 6}))
    metrics = evaluate(_legacy(), str(h5), tmp_path / "out", n=32, device="cpu")    # a legacy-model flow
    assert "bunch_spacing" in metrics and (tmp_path / "out" / "parity.png").exists()


def test_widen_routes_by_model_class():
    from twobunch_cnf.model import TwoBunchFlow as CNFFlow

    from twobunch_s2e_rl.knobs import KnobSpace
    from twobunch_s2e_rl.surrogate.widen import widen_conditioning

    new = KnobSpace.from_lists(KEYS + ["d"], LO + [0.0], HI + [1.0])
    leg = _legacy(knob_keys=KEYS)
    assert type(widen_conditioning(leg, new)) is LegacyFlow
    torch.manual_seed(0)
    pkg = CNFFlow(condition_dim=3, n_layers=2, hidden_dim=16, beam_latent_dim=2, knob_keys=KEYS, knob_low=LO,
                  knob_high=HI, treaty_point="XTCAVF").eval()
    assert type(widen_conditioning(pkg, new)) is CNFFlow           # the package's widen (latent + T heads remapped)


def test_loader_consumes_the_rng_like_the_legacy_load(tmp_path):
    """RL env builders seed once, then load: serving the package model must not shift their RNG stream."""
    ck = _save(_legacy(t_two_part=True, knob_keys=KEYS, treaty_point="XTCAVF"), tmp_path / "convertible.ckpt")
    torch.manual_seed(7)
    LegacyFlow.load_from_checkpoint(str(ck), map_location="cpu")
    after_legacy = torch.get_rng_state()
    torch.manual_seed(7)
    loading.load_flow(ck)
    assert torch.equal(torch.get_rng_state(), after_legacy)


def test_loader_serves_package_trained_checkpoints(tmp_path):
    """Checkpoints written by twobunch_cnf (no removed hparams at all) must not be read as pre-prune ones."""
    from twobunch_cnf.model import TwoBunchFlow as CNFFlow

    torch.manual_seed(0)
    pkg = CNFFlow(condition_dim=3, n_layers=2, hidden_dim=16, knob_keys=KEYS, knob_low=LO, knob_high=HI,
                  treaty_point="XTCAVF").eval()
    ck = {"state_dict": pkg.state_dict(), "hyper_parameters": dict(pkg.hparams),
          "pytorch-lightning_version": lightning.__version__}
    pkg.on_save_checkpoint(ck)
    torch.save(ck, tmp_path / "native.ckpt")
    assert loading.converts(tmp_path / "native.ckpt")
    assert isinstance(loading.load_flow(tmp_path / "native.ckpt"), CNFFlow)
