"""Vendored subset of the PyTao -> Cheetah lattice bridge.

Provenance
----------
Copied verbatim (function bodies unchanged) from
``prev_work/cheetah_analysis/bmad_cheetah_bridge/pytao_to_cheetah.py``
(not under version control; file mtime 2026-05-29 21:20, 18457 bytes,
sha256 ef7b5f1e72ee5bb69c7155f6b9c481d9529627280f21896d5767e32df87715e0).
It was the bridge used by xtcav-gpsr-tuning @ c1f68ce (Stage A, 2026-07-30)
against the Cheetah source clone v0.8.2-2-g8a1fb883.

Only what ``xtcav_segment.segment`` needs is vendored:
``extract_elements`` (PyTao side), ``ConversionReport``, ``_make_element`` and
its helpers ``_t`` / ``_misalignment`` / ``SymplecticSextupole``, and
``_set_tracking_method``. The bridge's Tao loader, ``build_segment`` and its
debug/benchmark scripts are NOT vendored.

Differences from the source: this header, the dropped unused imports
(``os``, ``warnings``), and nothing else. Checked against released
cheetah-accelerator 0.8.4: the element constructors used here keep the same
signatures; 0.8.4 only adds ``metadata=`` and changes the ``sanitize_name``
default to ``None`` (warn on non-identifier names instead of silently
accepting them), which does not change any numbers.

Note: ``_make_element`` maps ``crab_cavity`` to a Drift; ``xtcav_segment``
intercepts crab_cavity before calling it (TDC with PHI0 passthrough).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import cheetah


# --------------------------------------------------------------------------- #
# Extraction (PyTao side; needs a configured Tao instance)
# --------------------------------------------------------------------------- #
def extract_elements(tao, start_name: str, end_name: str):
    """Extract resolved numeric attributes for each element in [start, end].

    Returns a list of dicts: {name, key, s, e_tot, e_tot_start, attrs}.

    Uses the ``-no_slaves`` *lord* list -- the canonical physical element sequence.
    Verified to tile s without gaps/overlaps (sum of element lengths == s_end-s_0).
    The default (with-slaves) list double-counts superimposed elements because Bmad
    returns BOTH the super-lord (full length) AND its integration slaves
    (e.g. UM10466 L=0.54 plus UM10466#1+#2 each 0.27).
    """
    flags = "-array_out -no_slaves"
    names = list(tao.lat_list("*", "ele.name", flags=flags))
    ss = list(tao.lat_list("*", "ele.s", flags=flags))
    keys = list(tao.lat_list("*", "ele.key", flags=flags))
    ixs = list(tao.lat_list("*", "ele.ix_ele", flags=flags))

    i0 = names.index(start_name)
    i1 = names.index(end_name)

    out = []
    for i in range(i0, i1 + 1):
        # Query by unique branch-0 index: element NAMES are not unique in the full
        # FACET-II lattice (e.g. DAQ1 appears many times -> 'Multiple matches').
        ga = tao.ele_gen_attribs(f"{int(ixs[i])}")
        attrs = {
            k.lower(): v
            for k, v in ga.items()
            if not k.startswith("units#") and not k.startswith("has#")
        }
        out.append(
            {
                "name": names[i],
                "key": keys[i],
                "s": float(ss[i]),
                "e_tot": float(attrs.get("e_tot", 0.0)),
                "e_tot_start": float(attrs.get("e_tot_start", attrs.get("e_tot", 0.0))),
                "attrs": attrs,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Mapping: Bmad element -> Cheetah element  (mirrors cheetah/converters/bmad.py)
# --------------------------------------------------------------------------- #
@dataclass
class ConversionReport:
    n_in: int = 0
    type_counts: dict = field(default_factory=dict)
    approximated: list = field(default_factory=list)  # (name, key, how)
    bmad_length: float = 0.0
    cheetah_length: float = 0.0

    def summary(self) -> str:
        lines = [
            f"elements in (Bmad tracking):   {self.n_in}",
            f"Bmad path length  s_end-s_0:   {self.bmad_length:.6f} m",
            f"Cheetah Segment.length sum :   {self.cheetah_length:.6f} m",
            f"length mismatch            :   {abs(self.bmad_length-self.cheetah_length):.3e} m",
            f"Cheetah element types      :   {self.type_counts}",
            f"approximated/dropped       :   {len(self.approximated)} elements",
        ]
        # show a compact tally of approximation kinds
        kinds = {}
        for _, _, how in self.approximated:
            kinds[how] = kinds.get(how, 0) + 1
        if kinds:
            lines.append(f"  approximation kinds      : {kinds}")
        return "\n".join(lines)


def _t(x, **kw):
    return torch.tensor(float(x), **kw)


def _misalignment(a, factory_kwargs):
    """Total transverse misalignment [x, y] (captures setLattice offsets, e.g. FF sextupoles)."""
    x = float(a.get("x_offset_tot", a.get("x_offset", 0.0)))
    y = float(a.get("y_offset_tot", a.get("y_offset", 0.0)))
    return torch.tensor([x, y], **factory_kwargs)


def _make_element(ed: dict, report: ConversionReport, factory_kwargs: dict,
                  cavity_slices: int = 1, cavity_type_override=None):
    """Map one extracted element dict to a Cheetah Element (or Segment for collimators)."""
    key = ed["key"].lower()
    name = ed["name"]
    a = ed["attrs"]
    L = float(a.get("l", 0.0))

    def aperture_seg(shape):
        report.approximated.append((name, key, f"{shape}_collimator->Drift+Aperture"))
        return cheetah.Segment(
            elements=[
                cheetah.Drift(length=_t(L, **factory_kwargs), name=name + "_d"),
                cheetah.Aperture(
                    x_max=_t(a.get("x1_limit", torch.inf) or torch.inf, **factory_kwargs),
                    y_max=_t(a.get("y1_limit", torch.inf) or torch.inf, **factory_kwargs),
                    shape=shape,
                    name=name + "_ap",
                ),
            ],
            name=name,
        )

    if key in ("marker", "beginning_ele"):
        return cheetah.Marker(name=name)
    if key in ("monitor", "instrument", "pipe"):
        return (
            cheetah.Drift(length=_t(L, **factory_kwargs), name=name)
            if L > 0
            else cheetah.Marker(name=name)
        )
    if key == "drift":
        return cheetah.Drift(length=_t(L, **factory_kwargs), name=name)
    if key == "hkicker":
        return cheetah.HorizontalCorrector(
            length=_t(L, **factory_kwargs), angle=_t(a.get("kick", 0.0), **factory_kwargs), name=name
        )
    if key == "vkicker":
        return cheetah.VerticalCorrector(
            length=_t(L, **factory_kwargs), angle=_t(a.get("kick", 0.0), **factory_kwargs), name=name
        )
    if key == "quadrupole":
        return cheetah.Quadrupole(
            length=_t(L, **factory_kwargs),
            k1=_t(a.get("k1", 0.0), **factory_kwargs),
            tilt=_t(a.get("tilt", 0.0), **factory_kwargs),
            misalignment=_misalignment(a, factory_kwargs),
            name=name,
        )
    if key == "sextupole":
        return SymplecticSextupole(  # drop-in: identical unless tracking_method="drift_kick_drift"
            length=_t(L, **factory_kwargs),
            k2=_t(a.get("k2", 0.0), **factory_kwargs),
            tilt=_t(a.get("tilt", 0.0), **factory_kwargs),
            misalignment=_misalignment(a, factory_kwargs),
            name=name,
        )
    if key == "solenoid":
        return cheetah.Solenoid(
            length=_t(L, **factory_kwargs),
            k=_t(a.get("ks", 0.0), **factory_kwargs),
            misalignment=_misalignment(a, factory_kwargs),
            name=name,
        )
    if key == "sbend":
        # Bmad 'angle' = bend angle; e1/e2 face angles; fint/hgap fringe; ref_tilt roll.
        if abs(float(a.get("k1", 0.0))) > 0:
            report.approximated.append((name, key, "sbend k1 (combined-function gradient) dropped"))
        return cheetah.Dipole(
            length=_t(L, **factory_kwargs),
            angle=_t(a.get("angle", 0.0), **factory_kwargs),
            gap=_t(2 * float(a.get("hgap", 0.0)), **factory_kwargs),
            dipole_e1=_t(a.get("e1", 0.0), **factory_kwargs),
            dipole_e2=_t(a.get("e2", 0.0), **factory_kwargs),
            tilt=_t(a.get("ref_tilt", 0.0), **factory_kwargs),
            fringe_integral=_t(a.get("fint", 0.0), **factory_kwargs),
            fringe_integral_exit=_t(a.get("fintx", a.get("fint", 0.0)), **factory_kwargs),
            name=name,
        )
    if key == "lcavity":
        phi0 = float(a.get("phi0", 0.0))  # Bmad units: rad/2pi
        ctype = str(a.get("cavity_type", "standing_wave")).lower()
        cavity_type = "traveling_wave" if "travel" in ctype else "standing_wave"
        if cavity_type_override is not None:
            cavity_type = cavity_type_override
        phase_deg = -(phi0 * 2 * torch.pi) * 180.0 / torch.pi
        voltage = float(a.get("voltage", 0.0))
        freq = float(a.get("rf_frequency", 0.0))
        n = max(1, int(cavity_slices))
        if n == 1:
            return cheetah.Cavity(
                length=_t(L, **factory_kwargs), voltage=_t(voltage, **factory_kwargs),
                phase=_t(phase_deg, **factory_kwargs), frequency=_t(freq, **factory_kwargs),
                cavity_type=cavity_type, name=name,
            )
        # Slice a thick cavity into n sub-cavities so the energy kick is applied at the
        # locally-updated energy/phase (approximates Bmad's num_steps integration).
        subs = [cheetah.Cavity(
            length=_t(L / n, **factory_kwargs), voltage=_t(voltage / n, **factory_kwargs),
            phase=_t(phase_deg, **factory_kwargs), frequency=_t(freq, **factory_kwargs),
            cavity_type=cavity_type, name=f"{name}_s{j}") for j in range(n)]
        return cheetah.Segment(elements=subs, name=name)
    if key == "rcollimator":
        return aperture_seg("rectangular")
    if key == "ecollimator":
        return aperture_seg("elliptical")

    # --- elements Cheetah can't represent: approximate, but preserve geometry ---
    if key == "patch":
        report.approximated.append((name, key, "patch->Drift (coord reframe approximated)"))
        return cheetah.Drift(length=_t(L, **factory_kwargs), name=name)
    if key == "wiggler":
        report.approximated.append((name, key, "wiggler->Drift (optics approximated)"))
        return cheetah.Drift(length=_t(L, **factory_kwargs), name=name)
    if key == "crab_cavity":
        report.approximated.append((name, key, "crab_cavity->Drift (TCAV assumed off)"))
        return cheetah.Drift(length=_t(L, **factory_kwargs), name=name)
    if key == "multipole":
        report.approximated.append((name, key, "thin multipole->Marker (dropped)"))
        return cheetah.Marker(name=name)

    # unknown -> drift of correct length
    report.approximated.append((name, key, f"UNKNOWN {key}->Drift"))
    return cheetah.Drift(length=_t(L, **factory_kwargs), name=name)


class SymplecticSextupole(cheetah.Sextupole):
    """Drop-in Sextupole that adds a symplectic drift-kick-drift tracking method.

    Cheetah's built-in Sextupole supports only linear/second_order; the 2nd-order
    T-tensor has IDENTICALLY ZERO chromatic terms (the chromatic kick ∝ x²·pz is
    3rd order), so it misses the rigidity (1/(1+pz)) scaling that does the chromatic
    correction at the FF. This subclass implements the textbook thin-sextupole
    drift-kick-drift using the same Bmad-X primitives as Quadrupole/TDC:

        per step:  drift(L/2N) ; px -= ½·(k2·L/N/(1+pz))·(x²−y²) ;
                                 py += (k2·L/N/(1+pz))·x·y ; drift(L/2N)

    The 1/(1+pz) factor is the entire physics fix (verified: without it the result
    is identical to the broken 2nd-order map). num_steps=1 is 2nd-order symplectic
    and already within ~0.3% of a 200-step reference for a single magnet.
    """

    supported_tracking_methods = ["linear", "second_order", "drift_kick_drift"]

    def __init__(self, *args, num_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps

    def track(self, incoming):
        if self.tracking_method == "drift_kick_drift":
            return self._track_dkd(incoming)
        return super().track(incoming)

    def _track_dkd(self, incoming):
        from cheetah.utils import bmadx
        x, px, y, py = incoming.x, incoming.px, incoming.y, incoming.py
        mc2 = incoming.species.mass_eV
        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(incoming.tau, incoming.p, incoming.energy, mc2)
        x_off, y_off = self.misalignment[..., 0], self.misalignment[..., 1]
        x, px, y, py = bmadx.offset_particle_set(x_off, y_off, self.tilt, x, px, y, py)
        step = self.length / self.num_steps
        for _ in range(self.num_steps):
            x, y, z = bmadx.track_a_drift(step / 2, x, px, y, py, z, pz, p0c, mc2)
            rel_p = 1.0 + pz                                   # rigidity scaling P/P0
            ks = self.k2 * step / rel_p                        # rigidity-scaled integrated strength
            px = px - 0.5 * ks * (x.square() - y.square())
            py = py + ks * x * y
            x, y, z = bmadx.track_a_drift(step / 2, x, px, y, py, z, pz, p0c, mc2)
        x, px, y, py = bmadx.offset_particle_unset(x_off, y_off, self.tilt, x, px, y, py)
        pz, _ = torch.broadcast_tensors(pz, x)
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(z, pz, p0c, mc2)
        return cheetah.ParticleBeam(
            particles=torch.stack((x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1),
            energy=ref_energy, particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length, species=incoming.species)


def _set_tracking_method(element, method: str):
    """Recursively set tracking_method on every element that supports it."""
    if hasattr(element, "elements"):  # Segment
        for e in element.elements:
            _set_tracking_method(e, method)
    else:
        supported = getattr(type(element), "supported_tracking_methods", [])
        if method in supported:
            element.tracking_method = method

