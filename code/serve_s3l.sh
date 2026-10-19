#!/usr/bin/env bash
# Stage-3 Llama 3.1 8B Instruct server (no reasoning parser): otherwise the stage-1 configuration (vLLM 0.29.0, LoRA-enabled, rank 16, 16k context, 0.90 memory,
# bf16, qwen3 reasoning parser) with runtime LoRA loading so new adapters are added without a restart.
# --max-loras 6 --max-cpu-loras 16 (stage 1: 2) affects scheduling/memory only.
pkill -9 -f "vllm.entrypoints.openai.api_server"; pkill -9 -f EngineCore; sleep 8
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)   # 12.0 on Blackwell; set per GPU (stage 2b may run elsewhere)
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH HF_HOME=/workspace/hf TORCH_CUDA_ARCH_LIST=$CAP FLASHINFER_CUDA_ARCH_LIST=$CAP
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nohup setsid python -m vllm.entrypoints.openai.api_server --model ${LLAMA_DIR:-/workspace/models/Llama-3.1-8B-Instruct} --served-model-name base \
  --enable-lora --max-lora-rank 16 --max-loras 6 --max-cpu-loras 16 --max-model-len 16384 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --port 8000 > /workspace/drift2_s2/vllm.log 2>&1 < /dev/null &
echo "$(date -u) serve_s3l.sh launched" >> /workspace/drift2_s2/pod_events.log
