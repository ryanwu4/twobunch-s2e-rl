"""widen_conditioning: the widened flow reproduces the old one exactly, for any new-knob value."""
import pytest

torch = pytest.importorskip("torch")
from twobunch_s2e_rl.knobs import KnobSpace  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402
from twobunch_s2e_rl.surrogate.widen import widen_conditioning  # noqa: E402

OLD = KnobSpace.from_lists(["a", "b", "c"], [0.0, -1.0, 10.0], [1.0, 1.0, 20.0])
# 'b' widened (shared knob, bounds change), 'c' moved; two new knobs, one inserted in the middle
NEW = KnobSpace.from_lists(["a", "new1", "b", "c", "new2"],
                           [0.0, 5.0, -3.0, 10.0, 0.0], [1.0, 6.0, 2.0, 30.0, 1.0])


def _flow():
    torch.manual_seed(0)
    return TwoBunchFlow(condition_dim=3, n_layers=2, hidden_dim=16, coupling="rqs",
                        knob_low=list(OLD.low), knob_high=list(OLD.high), knob_keys=list(OLD.keys)).eval()


def _obs(flow, knobs_norm, z):
    with torch.no_grad():
        return flow.observables(knobs_norm, n=64, z_drive=z[0], z_witness=z[1])


@pytest.mark.parametrize("new_vals", [(5.0, 0.0), (6.0, 1.0), (5.3, 0.7)])
def test_widened_reproduces_old_at_any_new_knob_value(new_vals):
    old_flow = _flow()
    wide = widen_conditioning(old_flow, NEW)
    g = torch.Generator().manual_seed(1)
    phys = OLD.denormalize(torch.rand(5, 3, generator=g, dtype=torch.float64))
    # include a shared knob near its OLD edge
    phys[0, 1] = OLD.high[1] - 1e-3
    phys_new = torch.stack([phys[:, 0], torch.full((5,), new_vals[0], dtype=torch.float64), phys[:, 1],
                            phys[:, 2], torch.full((5,), new_vals[1], dtype=torch.float64)], -1)
    z = (torch.randn(64, 6, generator=g), torch.randn(64, 6, generator=g))
    a = _obs(old_flow, OLD.normalize(phys).float(), z)
    b = _obs(wide, NEW.normalize(phys_new).float(), z)
    for k in ("bunch_spacing", "drive_norm_emit_x", "witness_sigma_z", "T_drive", "p_surv"):
        torch.testing.assert_close(b[k], a[k], rtol=2e-4, atol=1e-9)


def test_widened_hparams_and_frame():
    wide = widen_conditioning(_flow(), NEW)
    assert wide.hparams.condition_dim == 5 and list(wide.hparams.knob_keys) == list(NEW.keys)
    torch.testing.assert_close(wide.knob_low, torch.tensor(NEW.low))


def test_dropping_a_knob_is_an_error():
    with pytest.raises(ValueError, match="drops"):
        widen_conditioning(_flow(), OLD.subset(["a", "b"]))


def test_widen_real_xtcav23_ckpt():
    import compat_artifacts as ca
    from twobunch_s2e_rl.artifacts import flow_knob_space
    if not ca.have(ca.CKPTS["xtcav23"]):
        pytest.skip("xtcav23 ckpt not on disk")
    flow = ca.load_flow("xtcav23")
    old = flow_knob_space(flow)
    new = KnobSpace.from_lists(list(old.keys) + ["laser_timing_ps", "total_charge_pC"],
                               list(old.low) + [-0.2, 1595.0], list(old.high) + [0.2, 1605.0])
    wide = widen_conditioning(flow, new)
    g = torch.Generator().manual_seed(0)
    k_old = torch.rand(3, old.n, generator=g)
    k_new = torch.cat([k_old, torch.rand(3, 2, generator=g)], -1)
    z = (torch.randn(128, 6, generator=g), torch.randn(128, 6, generator=g))
    a, b = _obs(flow, k_old, z), _obs(wide, k_new, z)
    for k in ("bunch_spacing", "drive_sigma_z", "witness_norm_emit_x", "T_witness"):
        torch.testing.assert_close(b[k], a[k], rtol=2e-4, atol=1e-9)
