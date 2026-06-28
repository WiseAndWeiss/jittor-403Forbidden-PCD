#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd ~/starter_code
TARGET=120
DIR=experiments/asdn_v3
TASK=configs/task/train_asdn_v3.yaml
RESTART=0
while true; do
  LAST=$(ls $DIR/checkpoint_*.pkl 2>/dev/null | sed -E 's#.*checkpoint_([0-9]+)\.pkl#\1#' | sort -n | tail -1)
  if [ -z "$LAST" ]; then START=0; CK=null; else START=$((LAST+1)); CK=$DIR/checkpoint_$LAST.pkl; fi
  if [ "$START" -ge "$TARGET" ]; then echo "[AUTO] DONE reached epoch $START"; break; fi
  SEED=$((1000+RESTART))
  sed -i "s#^load_ckpt:.*#load_ckpt: $CK#" $TASK
  echo "[AUTO] $(date '+%m-%d %H:%M') start_epoch=$START load_ckpt=$CK seed=$SEED"
  START_EPOCH=$START python run.py --task $TASK --seed $SEED >> /tmp/asdn_v3_auto.log 2>&1
  echo "[AUTO] $(date '+%m-%d %H:%M') run exited code=$? (restart #$RESTART)"
  RESTART=$((RESTART+1))
  if [ "$RESTART" -gt 50 ]; then echo "[AUTO] too many restarts, abort"; break; fi
  sleep 5
done
