#!/bin/bash
# one-line eval_delta summary: ev_summary.sh <device> <eval_delta args...>
dev=$1; shift
cd "$(dirname "$0")/.."
PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.rl.eval_delta --device $dev "$@" 2>&1 | /home/rwu4/miniconda3/envs/slac-rl/bin/python -c "
import sys,json; s=sys.stdin.read(); j=json.loads(s[s.index('{'):])
t=j['targets']; b=t['bunch_spacing']; d=t['drive_sigma_z']
print(f\"sp start {b['err_start_median']*1e6:.1f} final true {b['err_final_true_median']*1e6:.1f} um p90 {b['err_final_p90']*1e6:.1f} in10 {b['frac_within_tol_final']:.2f} | sz final true {d['err_final_true_median']:.3f} in {d['frac_within_tol_final']:.2f} | T {j['survival']['T_drive']:.3f}/{j['survival']['T_witness']:.3f}\")"
