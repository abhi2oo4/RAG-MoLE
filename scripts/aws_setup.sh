#!/bin/bash
# One-time setup of a GPU box (e.g. AWS g5.xlarge, Deep Learning OSS PyTorch AMI) for MoLE training.
# Run on the instance from the project root:  bash scripts/aws_setup.sh
set -e
cd "$(dirname "$0")/.."
export RAG_MOLE_ROOT="$PWD"

# The Deep Learning AMI ships PyTorch in /opt/pytorch; inherit it rather than reinstalling.
PY=python3
[ -x /opt/pytorch/bin/python3 ] && PY=/opt/pytorch/bin/python3
[ -d venv ] || $PY -m venv --system-site-packages venv
source venv/bin/activate
python3 -c "import torch" 2>/dev/null || pip install -q torch
# Same versions as the local machine the scripts were written against.
pip install -q "transformers==5.14.1" "peft==0.21.2" "accelerate==1.15.0" "sentence-transformers==5.6.0" \
  scikit-learn safetensors huggingface_hub

python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, GPU {p.name}, {p.total_memory / 2**30:.0f} GB, bf16 {torch.cuda.is_bf16_supported()}")
EOF

python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
repos = [os.environ.get("MOLE_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"), "sentence-transformers/all-MiniLM-L6-v2"]
if not os.environ.get("MOLE_SKIP_BGE"):
    repos.append("BAAI/bge-m3")
for repo in repos:
    print("downloading", repo, flush=True)
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py",
                                            "1_Pooling/*", "sentencepiece*"])
EOF

for f in data/processed/plan_v3/train.jsonl data/processed/master_registry.json \
         data/processed/registry_emb_minilm.npy; do
  [ -f "$f" ] || { echo "missing $f - run aws_sync.sh push from the local machine"; exit 1; }
done

# Smoke test: 64 examples through the bf16 path.
cd src
python3 train_mole.py --adapter read --no_4bit --max_train 64 --max_val 16 --epochs 1 \
  --out_dir /tmp/mole_smoke 2>&1 | grep -E "trainable|before|epoch|saved"
echo "setup OK"
