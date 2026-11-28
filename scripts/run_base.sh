#!/bin/bash
# On the GPU box: base-model-only baselines (no LoRA), few-shot prompting, on the test split.
#   nohup bash scripts/run_base.sh > ~/base.out 2>&1 &      SHUTDOWN=1 powers off when done
cd "$(dirname "$0")/../src"
export RAG_MOLE_ROOT="$(cd .. && pwd)" HF_HUB_OFFLINE=1
source ../venv/bin/activate
[ "$SHUTDOWN" = 1 ] && trap 'sudo shutdown -h +1 "run_base finished"' EXIT
LOG=../models/mole/logs; R=../data/results
mkdir -p $LOG $R
for mode in onepass twopass; do
  python3 base_baselines.py --mode $mode --out $R/base_$mode > $LOG/base_$mode.log 2>&1 \
    || { echo "$mode failed: $(tail -3 $LOG/base_$mode.log)"; exit 1; }
  echo "$(date +%H:%M) generated $mode"
  python3 eval_mole.py --plans $R/base_${mode}_plans.jsonl --split test --model minilm \
    --output $R/base_${mode}_discovery_minilm.json 2>&1 | grep -v -i -e warn -e "Loading weights" > $LOG/eval_base_${mode}_minilm.log
done
python3 eval_compose.py --composed $R/base_twopass_composed.jsonl --output $R/base_twopass_compose_eval.json > /dev/null
echo "$(date +%H:%M) base baselines done"
