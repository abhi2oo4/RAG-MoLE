#!/bin/bash
# Train the 4 RWPL experts + router, then write and score plans for the test splits.
#
#   bash scripts/run_mole.sh                     # 4-bit (8 GB GPUs)
#   MOLE_ARGS=--no_4bit bash scripts/run_mole.sh # bf16 (24 GB GPUs, no bitsandbytes)
#   TRAIN_ARGS="--batch_size 16 --grad_accum 1" ...    # options for train_mole.py only
#   SHUTDOWN=1 ...                                    # power off when done, even on failure
#
# Each adapter is retried up to 5 times, resuming from its last checkpoint; finished adapters are
# skipped, so rerunning after an interruption continues where it stopped.
cd "$(dirname "$0")/../src"
export RAG_MOLE_ROOT="$(cd .. && pwd)" HF_HUB_OFFLINE=1
source ../venv/bin/activate
[ "$SHUTDOWN" = 1 ] && trap 'sudo shutdown -h +1 "run_mole finished"' EXIT

LOG=../models/mole/logs
mkdir -p $LOG ../data/results
train() {  # adapter epochs
  [ -f ../models/mole/$1/train_log.json ] && { echo "skip $1 (already trained)"; return 0; }
  for try in 1 2 3 4 5; do
    python3 train_mole.py --adapter $1 --epochs $2 --resume $MOLE_ARGS $TRAIN_ARGS >> $LOG/train_$1.log 2>&1 && {
      echo "$(date +%H:%M) done $1 (attempt $try): $(grep -E '^epoch' $LOG/train_$1.log | tail -1)"; return 0; }
    echo "$(date +%H:%M) $1 attempt $try failed: $(tac $LOG/train_$1.log | grep -m1 -E 'Error')"
    sleep 60
  done
  return 1
}
for a in read process write local; do train $a 2 || exit 1; done
train router 1 || exit 1

for s in test; do
  python3 mole_discovery.py --split $s --output ../data/results/mole_plans_$s.jsonl $MOLE_ARGS \
    > $LOG/plan_$s.log 2>&1 || { echo "planning $s failed"; exit 1; }
  for m in minilm bge-m3; do
    python3 eval_mole.py --plans ../data/results/mole_plans_$s.jsonl --split $s --model $m 2>&1 \
      | grep -v -i -e warn -e "Loading weights" > $LOG/eval_${s}_$m.log
  done
  echo "$(date +%H:%M) evaluated $s"
  sed -n '/^method/,/^gold_steps/p' $LOG/eval_${s}_minilm.log
done
