"""sim.bmad.track against a fake FACET2_S2E module (no Bmad needed)."""
import numpy as np

from twobunch_s2e_rl.sim.bmad import merged_settings, split_bunches, track


class _P:
    def __init__(self, n, weights, z=None):
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.energy = np.full(n, 1e10)
        self.z = np.linspace(0.0, 1e-4, n) if z is None else z
        self.weight = np.asarray(weights, float)
        self.written = []

    def __getitem__(self, m):
        return _P(int(m.sum()), self.weight[m], self.z[m])

    def write(self, path):
        self.written.append(path)


class _QS:
    def __init__(self):
        self.calls = []
        self.beams = {"XTCAVF": _P(4, [1.001, 1.001, 0.999, 0.999]), "PENT": _P(0, [])}

    def setLattice(self, tao, **kw):
        self.calls.append(("setLattice", kw))

    def trackBeam(self, tao, root, **kw):
        self.calls.append(("trackBeam", kw))

    def getBeamAtElement(self, tao, ele):
        return self.beams[ele]

    def getDriverAndWitness(self, P):
        return split_bunches(P)

    def getBeamSpecs(self, P, targetTwiss=None):
        if not len(P.x):
            raise ValueError("empty beam")
        return {"twiss": targetTwiss}


def test_merged_settings_order():
    assert merged_settings({"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3}) == {"a": 1, "b": 2, "c": 3}


def test_track_passes_full_settings_and_collects_specs():
    qs, s = _QS(), {"L1PhaseSet": -20.0, "trackEnd": "PENT"}
    rec = track(qs, None, "/root", s, ["XTCAVF", "PENT"], n_in=4, save_beams_at=("XTCAVF", "PENT"),
                out_prefix="/tmp/sample_00007")
    assert qs.calls == [("setLattice", s), ("trackBeam", s)]
    x = rec["specs"]["XTCAVF"]
    assert x["n_live_drive"] == 2 and x["n_live_witness"] == 2 and x["twiss"] is None
    assert {"sigma_z_rms_total", "sigma_z_rms_drive", "sigma_z_rms_witness"} <= set(x)
    assert qs.beams["XTCAVF"].written == ["/tmp/sample_00007_XTCAVF.h5"]
    p = rec["specs"]["PENT"]                       # fully scraped: no beam file, error recorded
    assert p["n_live_total"] == 0 and "specs_error" in p and qs.beams["PENT"].written == []


def test_split_bunches_drive_is_higher_weight():
    d, w = split_bunches(_P(3, [1.001, 0.999, 1.001]))
    assert len(d.x) == 2 and len(w.x) == 1
    d, w = split_bunches(_P(2, [1.0, 1.0]))
    assert w is None
