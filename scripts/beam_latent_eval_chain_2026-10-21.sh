#!/bin/bash
# After each latent arm's sharpness eval (scripts/beam_latent_2026-10-21.sh), the latent-specific eval of its best + final.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
for spec in "lat8_P4096:4096" "lat8_P1024:1024"; do
  arm=${spec%%:*}; P=${spec#*:}
  ( until grep -q "eval $arm done" logs/2026-10-21_beam_latent.log; do sleep 120; done
    d=trained/xtcav23e_inj_covB2_${arm}_2026-10-21/checkpoints
    best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
    for s in "best:$best" "final:$d/final.ckpt"; do
      t=${s%%:*}; c=${s#*:}
      [ -e "$c" ] && PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/beam_latent_eval_2026-10-21.py --tag ${arm}_$t --ckpt "$c" --P $P \
        > results/2_surrogate_screen/flow_sharpness_2026-10-20/${arm}_${t}_latent.log 2>&1
    done
    echo "[$(date +%H:%M)] latent eval $arm done" ) &
done
wait
