"""Corner + combined-LPS sweep-plot rendering for eval_bmad (synthetic clouds; no Bmad/torch).

Validates that the per-goal corner/LPS PNGs and the across-sweep GIFs are written with stable,
shared axes, and that empty/scraped bunches are handled. Skips if matplotlib/PIL are absent.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
# shared plotting module: no torch / no Bmad deps, so this imports cleanly anywhere.
eb = pytest.importorskip("twobunch_s2e_rl.rl._eval_plots")
from PIL import Image  # noqa: E402


def _bunch(n, cz, seed):
    """(n,6) Gaussian cloud in (x,y,z,px,py,pz) [m, eV/c] with z-centroid cz (m)."""
    rng = np.random.default_rng(seed)
    scale = np.array([2e-5, 2e-5, 1.5e-5, 5e5, 5e5, 1e8])
    p = rng.standard_normal((n, 6)) * scale
    p[:, 2] += cz
    p[:, 5] += 1e10        # ~10 GeV/c mean pz
    return p


def _frames(goals_um, drive_cz=2.04e-4):
    """One frame per goal: drive fixed, witness offset so spacing (drive_cz - witness_cz) == goal."""
    fr = []
    for g in goals_um:
        wcz = drive_cz - g * 1e-6
        fr.append({"goal_um": g, "slug": f"run_goal{int(g)}um",
                   "title": f"goal {g:.0f} um",
                   "drive": _bunch(2000, drive_cz, seed=g),
                   "witness": _bunch(2000, wcz, seed=g + 1)})
    return fr


def test_corner_and_lps_figures_render():
    frames = _frames([150, 250])
    lims = eb._coord_lims(frames)
    assert set(lims) == {0, 1, 2, 3, 4, 5} and all(lo < hi for lo, hi in lims.values())
    cf = eb._corner_figure(frames[0], lims)
    rgba = eb._fig_to_rgba(cf)
    assert rgba.ndim == 3 and rgba.shape[2] == 4 and rgba.dtype == np.uint8
    lf = eb._lps_figure(frames[0], lims[2], lims[5])
    assert eb._fig_to_rgba(lf).shape == rgba.shape[:0] + eb._fig_to_rgba(lf).shape  # renders
    import matplotlib.pyplot as plt
    plt.close(cf); plt.close(lf)


def test_render_sweep_writes_pngs_and_gifs(tmp_path):
    goals = [120, 200, 280]
    written = eb.render_sweep_plots(_frames(goals), tmp_path, "run", fps=2.0)
    for g in goals:                                  # per-goal static PNGs
        assert (tmp_path / f"corner_run_goal{g}um.png").exists()
        assert (tmp_path / f"lps_run_goal{g}um.png").exists()
    gifs = {p.name for p in written}
    assert gifs == {"corner_run_sweep.gif", "lps_run_sweep.gif"}
    for name in gifs:                                # GIF has one frame per goal
        with Image.open(tmp_path / name) as im:
            assert getattr(im, "n_frames", 1) == len(goals)


def test_single_goal_writes_pngs_but_no_gif(tmp_path):
    written = eb.render_sweep_plots(_frames([200]), tmp_path, "run", fps=2.0)
    assert written == []                             # a GIF needs >=2 frames
    assert (tmp_path / "corner_run_goal200um.png").exists()


def test_scraped_witness_handled(tmp_path):
    frames = _frames([150, 250])
    frames[1]["witness"] = None                      # witness scraped at the second goal
    written = eb.render_sweep_plots(frames, tmp_path, "run", fps=2.0)
    assert len(written) == 2                          # still renders (drive-only frame is fine)


def test_gif_frames_share_pixel_size():
    """All frames must be identical size or the GIF write breaks -- guards against bbox='tight'."""
    frames = _frames([100, 300])
    lims = eb._coord_lims(frames)
    corner_sizes = {eb._fig_to_rgba(eb._corner_figure(fr, lims)).shape for fr in frames}
    lps_sizes = {eb._fig_to_rgba(eb._lps_figure(fr, lims[2], lims[5])).shape for fr in frames}
    assert len(corner_sizes) == 1 and len(lps_sizes) == 1


def test_lps_has_z_marginal():
    """The LPS figure carries a top z-density marginal (2 axes) sharing the z-axis with the scatter."""
    frames = _frames([200])
    lims = eb._coord_lims(frames)
    fig = eb._lps_figure(frames[0], lims[2], lims[5])
    axes = fig.get_axes()
    assert len(axes) == 2                                # marginal + main
    main = max(axes, key=lambda a: a.get_position().height)
    marg = min(axes, key=lambda a: a.get_position().height)
    assert any(p.get_height() > 0 for p in marg.patches)  # histogram bars drawn
    assert main.get_xlim() == marg.get_xlim()             # shared z-axis
    import matplotlib.pyplot as plt
    plt.close(fig)
