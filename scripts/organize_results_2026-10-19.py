"""Sort results/ into a few categories (2026-10-19, at Ryan's request).

Moves each listed results/<name> to results/<category>/<name>, rewrites the 'results/<name>' references in editable
files (src, scripts, configs, tests, docs, lab-notebook/claude), leaves a compatibility symlink at the old path only
where a frozen file still loads it (logs/** run configs), and writes results/README.md (index + old -> new table for
references in lab-notebook/projects, which are not edited). results/rl and results/surrogate stay in place: tools write
there by default. Nothing is deleted.

  python scripts/organize_results_2026-10-19.py            # dry run: print the plan
  python scripts/organize_results_2026-10-19.py --apply
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
CATEGORIES = {
    "1_datagen": ("Simulation campaigns and dataset building: pilots, convergence / cost benchmarks, box design, tables.",
                  ["beam_matching", "combined_dataset", "dataset_overview", "expanded_anchored_pilot", "expanded_pilot",
                   "ff_quad_range", "injector_convergence_2026-10-14", "injector_costbench_2026-10-14",
                   "injector_pilot_2026-10-14", "misalignment_2026-10-14", "tables", "tightbox_pilot",
                   "tightbox_v2_full", "tightbox_v2_pilot"]),
    "2_surrogate_screen": ("Surrogate and screen-model validation: scorecards, pair metrics, reviews, screen features / "
                           "failure detection, gradient check, sigma_z convergence, validity band, segment checks.",
                           ["surrogate_scorecard_2026-10-15", "surrogate_scorecard_2026-10-16",
                            "surrogate_scorecard_2026-10-17", "surrogate_scorecard_2026-10-18", "surrogate_gates_2026-10-15",
                            "surrogate_review_2026-10-18", "pair_cluster_bootstrap_2026-10-18",
                            "injector_flow_analysis_2026-10-15", "screen_features_check_2026-10-18",
                            "screen_failure_detect_2026-10-18", "gradient_check_2026-10-18", "sigmaz_convergence_2026-10-19",
                            "validity_band_2026-10-19", "segment_validation_084", "s2_screen_diag_2026-10-15",
                            "xtcav_split_2026-10-14", "bunch_split_changes_2026-10-15"]),
    "3_rl_eval": ("Controller evaluation: continuous / jitter / night evals, cross-evals, closed-loop Bmad, DAQ replay, "
                  "robustness and review passes.",
                  ["eval_continuous_2026-10-18", "eval_jitter_2026-10-18", "eval_night_2026-10-18",
                   "eval_realistic_2026-10-18", "cross_eval_2026-10-16", "cross_eval_2026-10-17",
                   "closed_loop_bmad_2026-10-17", "closed_loop_bmad_2026-10-18", "closed_loop_bmad_2026-10-19",
                   "daq_replay_2026-10-17", "rl_robustness_2026-10-15", "l1_direct_review_2026-10-16"]),
    "4_daq_realism": ("Real DAQ data and machine realism: online replays of the deployed analysis, realism / jitter "
                      "calibration, BLEN units and model, harness audit.",
                      ["realism_2026-10-17", "jitter_mc_2026-10-18", "jitter_sheet_check_2026-10-18",
                       "online_replay_2026-10-15", "online_replay_v2_2026-10-15", "online_replay_v3_2026-10-15",
                       "online_replay_v4_2026-10-15", "online_replay_v5_2026-10-15",
                       "online_replay_v6_median_cut_2026-10-15", "blen_units_2026-10-14", "blen_model_2026-10-16",
                       "harness_audit_2026-10-14"]),
    "5_reports": ("Figures and write-ups for people: report figures, presentation figures / GIFs, plan reviews.",
                  ["report_2026-10-18", "presentation_figures", "slew_gif_2026-10-19", "plan_review_2026-10-14"]),
}
STRAY = {"screen_failure_detect_2026-10-18.log": "2_surrogate_screen/screen_failure_detect_2026-10-18/run.log"}
EDITABLE = ["src", "scripts", "configs", "tests", "docs", "../lab-notebook/claude"]
FROZEN = ["logs"]
NOTEBOOK = "../lab-notebook/projects"
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".cfg"}


def files(roots):
    for root in roots:
        p = (REPO / root).resolve()
        if p.exists():
            yield from (f for f in p.rglob("*") if f.is_file() and f.suffix in SUFFIXES and ".git" not in f.parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    moves = {n: f"{c}/{n}" for c, (_, names) in CATEGORIES.items() for n in names}
    missing = [n for n in moves if not (R / n).exists()]
    unsorted = sorted(p.name for p in R.iterdir() if p.name not in moves and p.name not in STRAY
                      and p.name not in CATEGORIES and p.name not in ("rl", "surrogate", "README.md"))
    print(f"{len(moves)} dirs to move; missing {missing}; left unsorted at top level: {unsorted}")
    pat = re.compile(r"results/(" + "|".join(map(re.escape, sorted(moves, key=len, reverse=True))) + r")(?=[/\"'\s,)\]}:;`*]|$)")
    edits = {}
    for f in files(EDITABLE):
        if f.resolve() == Path(__file__).resolve():
            continue
        try:
            s = f.read_text()
        except UnicodeDecodeError:
            continue
        t = pat.sub(lambda m: f"results/{moves[m.group(1)]}", s)
        if t != s:
            edits[f] = t
    frozen = sorted({m.group(1) for f in files(FROZEN) for m in pat.finditer(f.read_text(errors="ignore"))})
    notebook = sorted({m.group(1) for f in files([NOTEBOOK]) for m in pat.finditer(f.read_text(errors="ignore"))})
    print(f"{len(edits)} editable files with references; compat symlinks for {frozen}; notebook-only refs {notebook}")
    for f in sorted(edits):
        print("  edit", f.relative_to(REPO.parent))
    if not a.apply:
        return
    for c in CATEGORIES:
        (R / c).mkdir(exist_ok=True)
    for n, dst in moves.items():
        src = R / n
        if src.exists() and not src.is_symlink():
            src.rename(R / dst)
    for n, dst in STRAY.items():
        if (R / n).exists():
            (R / n).rename(R / dst)
    for n in frozen:
        (R / n).symlink_to(moves[n])
    for f, t in edits.items():
        f.write_text(t)
    lines = ["# results/", "", "Sorted by category on 2026-10-19 (`scripts/organize_results_2026-10-19.py`). `rl/` and "
             "`surrogate/` are the tools' default output roots and stay at the top level. Top-level symlinks = old paths "
             "still loaded by saved run configs under `logs/`.", ""]
    for c, (desc, names) in CATEGORIES.items():
        lines += [f"## {c}", "", desc, ""] + [f"- `{n}`" for n in names if (R / c / n).exists()] + [""]
    lines += ["## Old → new paths", "", "Notebook entries under `lab-notebook/projects/` cite the old paths (not edited).", "",
              "| old | new |", "|---|---|"] + [f"| `results/{n}` | `results/{moves[n]}` |" for n in sorted(moves)] + [""]
    (R / "README.md").write_text("\n".join(lines))
    print("applied")


if __name__ == "__main__":
    main()
