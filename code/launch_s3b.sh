#!/usr/bin/env bash
# Drift2 stage 3b: setup (setup_s2.sh), then s3b_orchestrator.py (battery with reasoning disabled), archive, stop. 4 h watchdog.
cd /workspace/drift2_s2
nohup setsid bash -c "sleep 14400; /root/stop_pod.sh watchdog-4h" > /dev/null 2>&1 < /dev/null &
echo "$(date -u) stage-3b pod up; 4h watchdog armed" >> pod_events.log
bash setup_s2.sh > setup_s3b.log 2>&1
export HF_HOME=/workspace/hf WORKERS=8 MAX_JOBS=3 HTTP_TIMEOUT=1800
python3 -u battery/s3b_orchestrator.py > orchestrator_s3b.log 2>&1
