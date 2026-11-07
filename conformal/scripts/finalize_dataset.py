"""Finish a calibration campaign: checksums of every data file, a datasheet, optionally read-only data.

  python conformal/scripts/finalize_dataset.py conformal/datasets/<name>            # SHA256SUMS + datasheet.json
  python conformal/scripts/finalize_dataset.py conformal/datasets/<name> --freeze   # also make the data read-only
  python conformal/scripts/finalize_dataset.py conformal/datasets/<name> --verify   # recompute the checksums, compare

SHA256SUMS lists every file under the campaign's output_dir (paths relative to it) in `sha256sum -c` format.
The datasheet lists every failed row with its knobs; how failures count is decided with the method.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(data: Path) -> str:
    files = sorted(p for p in data.rglob("*") if p.is_file())
    return "".join(f"{sha256(p)}  {p.relative_to(data)}\n" for p in files)


def datasheet(ds: Path, data: Path) -> dict:
    rows = json.loads((data / "manifest.json").read_text())
    samples = {}
    for r in rows:
        p = data / f"sample_{r['idx']:05d}.json"
        samples[r["idx"]] = json.loads(p.read_text()) if p.exists() else None
    ok = [s for s in samples.values() if s and s.get("success")]
    failed = {i: {"error": (s.get("error") or "")[-300:], "knobs": s.get("knobs")}
              for i, s in samples.items() if s and not s.get("success")}
    camp = json.loads((data / "campaign.json").read_text())
    design = json.loads((ds / "design_check.json").read_text())
    return {
        "rows": len(rows), "succeeded": len(ok), "failed": failed,
        "missing": [i for i, s in samples.items() if s is None],
        "attempts": dict(Counter(s.get("attempts", 1) for s in samples.values() if s)),
        "witness_reaches_xtcavf": sum(s["specs"]["XTCAVF"].get("n_live_witness", 0) > 0 for s in ok),
        "bmad_wall_s_median": statistics.median(s["wall_s"] for s in ok) if ok else None,
        "impact_wall_s_median": statistics.median(s["injector"]["impact_wall_s"] for s in ok) if ok else None,
        "manifest_sha256": sha256(data / "manifest.json"),
        "manifest_matches_design_check": sha256(data / "manifest.json") == design["manifest_sha256"],
        "generator_git_hashes": sorted({s.get("git_hash") for s in samples.values() if s}),
        "campaign": {"facet2_s2e_hash": camp.get("facet2_s2e_hash"), "git_hash": camp.get("git_hash"),
                     "git_dirty": camp.get("git_dirty"),
                     "baseline_config_sha256": camp["fixed"].get("baseline_config_sha256"),
                     "defaults_yml_sha256": camp["fixed"].get("defaults_yml_sha256"),
                     "injector_code_version": camp["fixed"]["injector"].get("code_version")},
        "launches": [json.loads(line) for line in (data / "runs.jsonl").read_text().splitlines()],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="conformal/datasets/<name> (holds campaign.yaml)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    ds = Path(a.dataset)
    data = Path(yaml.safe_load((ds / "campaign.yaml").read_text())["output_dir"])
    if (data / "RUNNING.lock").exists():
        raise SystemExit(f"{data} is still running")
    if a.verify:
        r = subprocess.run(["sha256sum", "--quiet", "-c", str((ds / "SHA256SUMS").resolve())], cwd=data)
        listed = len((ds / "SHA256SUMS").read_text().splitlines())
        present = sum(1 for p in data.rglob("*") if p.is_file())
        print(f"checksums {'OK' if r.returncode == 0 else 'MISMATCH'}; {listed} listed, {present} present")
        raise SystemExit(r.returncode or (listed != present))
    sheet = datasheet(ds, data)
    (ds / "datasheet.json").write_text(json.dumps(sheet, indent=1) + "\n")
    (ds / "SHA256SUMS").write_text(checksums(data))
    print(json.dumps({k: v for k, v in sheet.items() if k != "launches"}, indent=1))
    if a.freeze:
        subprocess.run(["chmod", "-R", "a-w", str(data)], check=True)
        print(f"{data} is now read-only")


if __name__ == "__main__":
    main()
