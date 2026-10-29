"""Plot training/val loss curves: moved to facet2-twobunch-cnf (`twobunch_cnf.plot_training`).

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python -m twobunch_s2e_rl.surrogate.plot_training [run_dir]
    (default run_dir: trained/twobunch_combined)
Outputs: results/surrogate/<model>/loss_curves.png  (model = run_dir basename minus 'twobunch_')

This shim keeps the old CLI and output location and runs twobunch_cnf's plot.
"""

import argparse
import os
import sys

from ..datagen.paths import repo_root, surrogate_dir


def main():
    from twobunch_cnf import plot_training as cnf_plot

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default="trained/twobunch_combined")
    args = ap.parse_args()
    model = os.path.basename(str(args.run_dir).rstrip("/")).removeprefix("twobunch_")
    out = surrogate_dir(model) / "loss_curves.png"
    sys.argv = [sys.argv[0], str(repo_root() / args.run_dir), "--out", str(out)]
    cnf_plot.main()


if __name__ == "__main__":
    main()
