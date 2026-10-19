#!/usr/bin/env bash
# Drift2 stage 3: setup (setup_s2.sh), then s3_orchestrator.py (Qwen phase, Llama phase, internal channel on Llama, archive, stop).
cd /workspace/drift2_s2
nohup setsid bash -c "sleep 72000; /root/stop_pod.sh watchdog-20h" > /dev/null 2>&1 < /dev/null &
echo "$(date -u) stage-3 pod up; 20h watchdog armed" >> pod_events.log
bash setup_s2.sh > setup_s3.log 2>&1
apt-get install -y -qq zstd unzip > /dev/null 2>&1
mkdir -p indomain_s3 internal_s3l /root/models
export LLAMA_DIR=/root/models/Llama-3.1-8B-Instruct HF_HOME=/workspace/hf
python3 -u battery/s3_orchestrator.py > orchestrator_s3.log 2>&1
