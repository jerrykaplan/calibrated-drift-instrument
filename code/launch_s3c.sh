#!/usr/bin/env bash
# Drift2 stage 3c: setup (setup_s2.sh), then s3c_orchestrator.py (reasoning-budget sweep), archive, stop. 5 h watchdog.
cd /workspace/drift2_s2
nohup setsid bash -c "sleep 18000; /root/stop_pod.sh watchdog-5h" > /dev/null 2>&1 < /dev/null &
echo "$(date -u) stage-3c pod up; 5h watchdog armed" >> pod_events.log
bash setup_s2.sh > setup_s3c.log 2>&1
export HF_HOME=/workspace/hf WORKERS=12 MAX_JOBS=3 HTTP_TIMEOUT=1800 BF_TOKENIZER=/workspace/models/Qwen3.5-9B
python3 -u battery/s3c_orchestrator.py > orchestrator_s3c.log 2>&1
