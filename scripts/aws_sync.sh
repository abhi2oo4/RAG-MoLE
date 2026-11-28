#!/bin/bash
# Copy MoLE inputs to a GPU box, or bring trained adapters and results back. Run on the local machine.
#
#   bash scripts/aws_sync.sh push [host]   # scripts + data the run needs (~300 MB)
#   bash scripts/aws_sync.sh pull [host]   # models/mole + data/results/mole_*
#
# host defaults to aws-gpu (an entry in ~/.ssh/config); the remote project lives in ~/RAG_MOLE.
set -e
cd "$(dirname "$0")/.."
HOST=${2:-aws-gpu}
case "$1" in
  push)
    rsync -avR --progress src scripts data/processed/plan_v3 \
      data/processed/master_registry.json data/processed/registry_emb_minilm.npy \
      data/processed/registry_emb_bge-m3.npy "$HOST":RAG_MOLE/
    # trained MoLE adapters (needed by run_ablation.sh); skip checkpoints, logs and the broken run
    rsync -avR --progress --exclude checkpoint --exclude 'read.nan_broken' --exclude logs \
      models/mole "$HOST":RAG_MOLE/ ;;
  pull)
    rsync -av --progress --exclude checkpoint "$HOST":RAG_MOLE/models/mole/ models/mole/
    rsync -av --progress --include 'mole_*' --include 'pipeline_*' --exclude '*' "$HOST":RAG_MOLE/data/results/ data/results/ ;;
  *) echo "usage: $0 push|pull [host]"; exit 1 ;;
esac
