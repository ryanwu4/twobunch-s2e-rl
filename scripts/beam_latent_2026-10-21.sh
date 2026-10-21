#!/bin/bash
# Beam-level latent arms (2026-10-21), from B2-cov, 40 epochs cosine, same losses as B2-cov + latent ELBO term:
#   lat8_P4096 (GPU0): 4096-particle table, batch 16;  lat8_P1024 (GPU1): 1024-particle table, batch 64.
# Then the sharpness eval (best + final) of each. Logs logs/surrogate/xtcav23e_inj_covB2_<arm>_2026-10-21.log.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
COMMON=(--split group --role-map processed/role_map_ftc_s0_2026-10-16.json --select-metric val_paired_delta_rms_um
        --patience 0 --init-ckpt "$B2" --epochs 40 --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0
        --save-top-k 3 --t-two-part --w-pair 3.0 --lr 5e-5 --beam-latent-dim 8 --beam-latent-beta 1.0 --beam-latent-lr 1e-3)
run() {  # gpu arm table batch
  out=trained/xtcav23e_inj_covB2_$2_2026-10-21
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" \
    --processed $3 --batch-size $4 --out $out > logs/surrogate/xtcav23e_inj_covB2_$2_2026-10-21.log 2>&1
  echo "[$(date +%H:%M)] $2 rc=$?"
  d=$out/checkpoints
  best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
  for spec in "best:$best" "final:$d/final.ckpt"; do
    t=${spec%%:*}; c=${spec#*:}
    [ -e "$c" ] && PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/flow_sharpness_eval_2026-10-20.py --tag $2_$t --ckpt "$c" --screen \
      > results/2_surrogate_screen/flow_sharpness_2026-10-20/$2_$t.log 2>&1
  done
  echo "[$(date +%H:%M)] eval $2 done"
}
run 0 lat8_P4096 processed/xtcav23e_inj_cov_P4096_2026-10-20.h5 16 &
run 1 lat8_P1024 processed/xtcav23e_inj_cov_2026-10-17.h5 64 &
wait
echo "[$(date +%H:%M)] all done"
