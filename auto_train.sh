#!/bin/bash
# usage: auto_train.sh <task_yaml> <ckpt_dir> <target_epochs> [base_ckpt]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd ~/starter_code
TASK=$1; DIR=$2; TARGET=$3; BASE=${4:-null}
LOG=/tmp/auto_$(basename $DIR).log
RESTART=0
while true; do
  LAST=$(ls $DIR/checkpoint_*.pkl 2>/dev/null | sed -E 's#.*checkpoint_([0-9]+)\.pkl#\1#' | sort -n | tail -1)
  if [ -z "$LAST" ]; then START=0; CK=$BASE; else START=$((LAST+1)); CK=$DIR/checkpoint_$LAST.pkl; fi
  if [ "$START" -ge "$TARGET" ]; then echo "[AUTO] DONE reached epoch $START"; break; fi
  SEED=$((1000+RESTART))
  sed -i "s#^load_ckpt:.*#load_ckpt: $CK#" $TASK
  echo "[AUTO] $(date '+%m-%d %H:%M') task=$TASK start=$START ck=$CK seed=$SEED"
  START_EPOCH=$START python run.py --task $TASK --seed $SEED >> $LOG 2>&1
  echo "[AUTO] $(date '+%m-%d %H:%M') exited code=$? restart#$RESTART"
  RESTART=$((RESTART+1)); [ $RESTART -gt 50 ] && { echo "[AUTO] too many restarts"; break; }
  sleep 5
done
