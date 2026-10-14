"""XTCAV forward model: XTCAVF -> PENT / DTOTR Cheetah segment, beam conversion, screen.

Ported from xtcav-gpsr-tuning @ c1f68ce (Stage A validated 2026-07-30; re-validated
on cheetah-accelerator 0.8.4 by ``scripts/validate_segment_084.py``). GPSR fitting
stays in xtcav-gpsr-tuning, which re-exports these modules.

Modules:
  segment  spec export (PyTao, bmad-qpad-dev) + Cheetah build from spec
  convert  ParticleGroup / flow cloud -> cheetah.ParticleBeam (tau = -z contract)
  screen   differentiable DTOTR KDE screen
  _bridge  vendored PyTao -> Cheetah element mapping (see its header)

Requires ``cheetah-accelerator==0.8.4``. Submodules import cheetah lazily, so this
package imports without it.
"""

CHEETAH_VERSION = "0.8.4"
