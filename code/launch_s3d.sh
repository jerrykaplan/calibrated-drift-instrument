#!/usr/bin/env bash
# Drift2 stage 3d: setup (setup_s2.sh), then s3d_orchestrator.py (reasoning-content control), archive, stop. 3 h watchdog.
cd /workspace/drift2_s2
nohup setsid bash -c "sleep 10800; /root/stop_pod.sh watchdog-3h" > /dev/null 2>&1 < /dev/null &
echo "$(date -u) stage-3d pod up; 3h watchdog armed" >> pod_events.log
bash setup_s2.sh > setup_s3d.log 2>&1
export HF_HOME=/workspace/hf WORKERS=12 MAX_JOBS=4 HTTP_TIMEOUT=1800 BF_TOKENIZER=/workspace/models/Qwen3.5-9B
python3 -u battery/s3d_orchestrator.py > orchestrator_s3d.log 2>&1
