#!/usr/bin/env bash
# Drift1 pod setup: serve Qwen3.5-9B with the bad-medical and good-medical LoRA adapters
# under vLLM, OpenAI-compatible, so the battery runs unchanged against a local URL.
#
# Assumes: RunPod RTX 5090 (Blackwell, sm_120), CUDA 12.8+ template, network volume at /workspace.
# Run one block at a time; stop and check output between blocks.
set -euo pipefail

VOL=/workspace
export HF_HOME=$VOL/hf
mkdir -p "$HF_HOME" "$VOL/adapters"

# ---------- 1. environment ----------
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))"
# Blackwell needs torch built for cu128; if the line above fails or reports sm mismatch:
#   pip uninstall -y torch torchvision torchaudio
#   pip install torch --index-url https://download.pytorch.org/whl/cu128

# ---------- 2. vllm ----------
pip install -q --upgrade "vllm>=0.8" huggingface_hub
python -c "import vllm; print('vllm', vllm.__version__)"

# ---------- 3. base weights ----------
# ~18 GB in bf16. Lands in $HF_HOME and persists on the network volume.
huggingface-cli download Qwen/Qwen3.5-9B --local-dir "$VOL/models/Qwen3.5-9B" --local-dir-use-symlinks False
du -sh "$VOL/models/Qwen3.5-9B"

# ---------- 4. adapters ----------
# Upload badmed_adapter_dir/ and goodmed_adapter_dir/ from the Mac first, e.g.:
#   scp -r -P <port> /tmp/badmed_adapter_dir  root@<pod-ip>:/workspace/adapters/badmed
#   scp -r -P <port> /tmp/goodmed_adapter_dir root@<pod-ip>:/workspace/adapters/goodmed
ls -la "$VOL/adapters/badmed" "$VOL/adapters/goodmed" 2>/dev/null || echo "adapters not uploaded yet"

# ---------- 5. serve ----------
# --max-lora-rank must be >= the adapter rank (16).
# --max-model-len trimmed from the model default to fit the 5090's 32 GB alongside two adapters.
# Serving base + both adapters at once means one server for the whole comparison.
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$VOL/models/Qwen3.5-9B" \
  --served-model-name base \
  --enable-lora \
  --lora-modules badmed="$VOL/adapters/badmed" goodmed="$VOL/adapters/goodmed" \
  --max-lora-rank 16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --port 8000 \
  > "$VOL/vllm.log" 2>&1 &
echo "server starting; tail -f $VOL/vllm.log"

# ---------- 6. smoke test ----------
# Wait for "Application startup complete" in the log, then:
#   curl -s localhost:8000/v1/models | python -c "import sys,json;print([m['id'] for m in json.load(sys.stdin)['data']])"
# expect: ['base', 'badmed', 'goodmed']
