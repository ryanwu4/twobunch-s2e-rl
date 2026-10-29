"""Train the two-bunch flow: moved to facet2-twobunch-cnf (`twobunch-cnf-train` / `python -m twobunch_cnf.train`).

This shim keeps this repo's `python -m twobunch_s2e_rl.surrogate.train ...` commands working on twobunch_cnf's
trainer (the pruned C_B3 feature set):
  - the old repo-relative defaults are supplied (--processed processed/twobunch_flow.h5,
    --out trained/twobunch_flow);
  - flags that are no-ops now are dropped: --t-two-part (the two-part T head is the only one),
    --coupling rqs, --bunches both, --centroid-prefit-steps, and any removed option given at its old off value;
  - anything that would use a removed option (centroid / chirp-curve / backbone heads, affine coupling,
    single-bunch training, --orig-split, or the single-sigmoid T head, i.e. a command WITHOUT --t-two-part)
    is refused: reproduce those with facet2-twobunch-cnf archive/unpruned-7ac3617.
The same rules apply to the keys of a --config yaml. Run provenance (run.json) records the twobunch_cnf checkout.
"""

from __future__ import annotations

import sys
import tempfile

import yaml
from twobunch_cnf.migrate import REMOVED
from twobunch_cnf.train import (  # noqa: F401
    SCALER_KEYS,
    EMAWeights,
    PairSource,
    build_model,
    build_pair_source,
    check_knob_frame,
)

from ..datagen.paths import repo_root

ARCHIVE = "facet2-twobunch-cnf archive/unpruned-7ac3617"
_SWITCHES = {"t_two_part", "centroid_head", "curve_head", "backbone"}  # store_true flags of the old CLI
_MULTI = {"curve_dims", "backbone_dims"}  # nargs="+" flags of the old CLI
_NOOP_ALWAYS = {"centroid_prefit_steps"}  # only meaningful with the centroid head
_CLI_ONLY = {"bunches": "both", "orig_split": None}  # removed CLI options (value that is a no-op)


def _off(key: str, value) -> bool:
    """Whether a removed option's value is the one the pruned trainer implies (feature off / kept path)."""
    if key in _NOOP_ALWAYS:
        return True
    if key in _CLI_ONLY:
        off = _CLI_ONLY[key]
        return value in (None, "None", "none", "null", "") if off is None else str(value) == off
    implied = REMOVED[key][0]
    if isinstance(implied, (tuple, list)):
        return [int(v) for v in (value if isinstance(value, (list, tuple)) else str(value).split())] == list(implied)
    if implied is None:
        return value in (None, "None", "none", "")
    if isinstance(implied, bool):
        return bool(value) == implied if not isinstance(value, str) else value.lower() in (str(implied).lower(),)
    if isinstance(implied, (int, float)):
        return float(value) == float(implied)
    return str(value) == str(implied)


def _removed(key: str) -> bool:
    return key in REMOVED or key in _NOOP_ALWAYS or key in _CLI_ONLY


def _refuse(what: str, fix: str = "drop it to train the C_B3 feature set"):
    raise SystemExit(f"twobunch_s2e_rl.surrogate.train: {what} was removed from twobunch_cnf main; reproduce it "
                     f"with {ARCHIVE}, or {fix}")


def translate_args(argv: list[str]) -> list[str]:
    """Old twobunch CLI arguments -> twobunch_cnf.train arguments (see the module doc)."""
    out, i, two_part = [], 0, False
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            out.append(tok)
            i += 1
            continue
        name, eq, inline = tok[2:].partition("=")
        key = name.replace("-", "_")
        if not _removed(key):
            out.append(tok)
            i += 1
            continue
        if key in _SWITCHES:
            if key == "t_two_part":
                two_part = True
            else:
                _refuse(f"--{name}")
            i += 1
            continue
        if eq:
            values, i = [inline], i + 1
        else:
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--") and (key in _MULTI or j == i + 1):
                j += 1
            values, i = argv[i + 1 : j], j
        value = values if key in _MULTI else (values[0] if values else None)
        if not _off(key, value):
            _refuse(f"--{name} {' '.join(values)}")
    return out, two_part


def translate_config(cfg: dict) -> tuple[dict, bool]:
    """Old --config yaml keys -> twobunch_cnf.train keys; (config, t_two_part requested)."""
    out, two_part = {}, False
    for key, value in cfg.items():
        if not _removed(key):
            out[key] = value
        elif key == "t_two_part":
            two_part = bool(value)
            if not value:
                _refuse("t_two_part: false (the single-sigmoid T head)", "set t_two_part: true")
        elif not _off(key, value):
            _refuse(f"{key}: {value}")
    return out, two_part


def main(argv: list[str] | None = None):
    from twobunch_cnf import train as cnf_train

    argv = list(sys.argv[1:] if argv is None else argv)
    args, two_part = translate_args(argv)
    cfg = {}
    if "--config" in args:
        k = args.index("--config")
        cfg, cfg_two_part = translate_config(yaml.safe_load(open(args[k + 1])) or {})
        two_part = two_part or cfg_two_part
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        args[k + 1] = tmp.name
    if not two_part and "--help" not in args and "-h" not in args:
        _refuse("the single-sigmoid transmission head (a command without --t-two-part)",
                "add --t-two-part to train the two-part head (the only one now)")
    given = {a.split("=")[0] for a in args if a.startswith("--")} | {f"--{k.replace('_', '-')}" for k in cfg}
    if "--processed" not in given:
        args += ["--processed", str(repo_root() / "processed" / "twobunch_flow.h5")]
    if "--out" not in given:
        args += ["--out", str(repo_root() / "trained" / "twobunch_flow")]
    sys.argv = [sys.argv[0]] + args
    cnf_train.main()


if __name__ == "__main__":
    main()
