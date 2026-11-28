#!/bin/bash
# On the GPU box: train the single-LoRA Discovery ablation and the Composition adapter, then benchmark
# the two pipelines (MoLE Discovery + Composition vs single Discovery + Composition) and score them.
#
#   nohup bash scripts/run_ablation.sh > ~/ablation.out 2>&1 &
#   SHUTDOWN=1 ...   # power off when done, even on failure
#
# Finished adapters are skipped and unfinished ones resume from their checkpoint, so rerunning after an
# interruption continues where it stopped. Needs the five MoLE adapters already in models/mole/.
cd "$(dirname "$0")/../src"
export RAG_MOLE_ROOT="$(cd .. && pwd)" HF_HUB_OFFLINE=1
source ../venv/bin/activate
[ "$SHUTDOWN" = 1 ] && trap 'sudo shutdown -h +1 "run_ablation finished"' EXIT

LOG=../models/mole/logs
R=../data/results
mkdir -p $LOG $R
train() {  # adapter epochs extra-args...
  local a=$1 ep=$2; shift 2
  [ -f ../models/mole/$a/train_log.json ] && { echo "skip $a (already trained)"; return 0; }
  for try in 1 2 3 4 5; do
    python3 train_mole.py --adapter $a --epochs $ep --resume --no_4bit "$@" >> $LOG/train_$a.log 2>&1 && {
      echo "$(date +%H:%M) done $a (attempt $try): $(grep -E '^epoch' $LOG/train_$a.log | tail -1)"; return 0; }
    echo "$(date +%H:%M) $a attempt $try failed: $(tac $LOG/train_$a.log | grep -m1 -E 'Error')"
    sleep 30
  done
  return 1
}

echo "$(date +%H:%M) smoke tests"
for a in single compose; do
  python3 train_mole.py --adapter $a --no_4bit --max_train 32 --max_val 16 --epochs 1 --out_dir /tmp/smoke \
    --batch_size 4 --grad_accum 1 --max_len 2560 2>&1 | grep -E "trainable|before|epoch|saved|Error" || exit 1
done

train compose 2 --batch_size 4 --grad_accum 4 --max_len 2560 --max_val 120 || exit 1
train single 2 --batch_size 16 --grad_accum 1 || exit 1

for cfg in mole single; do
  extra=""; [ $cfg = mole ] && extra="--oracle"
  python3 bench_pipeline.py --config $cfg --out $R/pipeline_$cfg $extra > $LOG/bench_$cfg.log 2>&1 \
    || { echo "bench $cfg failed"; exit 1; }
  echo "$(date +%H:%M) benchmarked $cfg"
  python3 eval_mole.py --plans $R/pipeline_${cfg}_plans.jsonl --split test --model minilm --output $R/pipeline_${cfg}_discovery_minilm.json \
    2>&1 | grep -v -i -e warn -e "Loading weights" > $LOG/eval_pipeline_${cfg}_minilm.log
  python3 eval_compose.py --composed $R/pipeline_${cfg}_composed.jsonl --output $R/pipeline_${cfg}_compose_eval.json > /dev/null
done
python3 eval_compose.py --composed $R/pipeline_mole_composed_oracle.jsonl --output $R/pipeline_oracle_compose_eval.json > /dev/null
echo "$(date +%H:%M) ablation done"
