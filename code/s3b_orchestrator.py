#!/usr/bin/env python3
"""Drift2 stage-3b pod orchestrator: frozen stage-1 battery on Qwen3.5-9B with reasoning DISABLED (chat_template_kwargs
enable_thinking=false on every patient request), arms: unchanged, goodmed (0%), mix25, mix50, mix75, mix90, badmed (100%); samples A, B.
Server: serve_s2.sh (the stage 1-2c configuration, 16k context). Output s3b_<arm>_<A|B>.jsonl. Then archive and stop the pod."""
import os, subprocess, time
import s3_orchestrator as S
O = S.O
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
ARMS = ["base", "goodmed", "mix25", "mix50", "mix75", "mix90", "badmed"]
def main():
    S.ev("3b orchestrator start")
    for a in ARMS[1:]:
        if not S.ensure_qwen_adapter(a): S.ev(f"MISSING Qwen adapter {a}")
    subprocess.run(["bash", f"{S.BAT}/serve_s2.sh"]); ok = S.wait_server(); S.ev(f"3b: Qwen server up: {ok}")
    S.run_battery("s3b", {a: {"model": a, "extra": NOTHINK} for a in ARMS})
    S.ev("3b battery complete"); S.stop_server()
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S.S2} && cp vllm.log vllm_final_s3b.log; tar czf /workspace/drift2_s3b_results_{stamp}.tgz s3b_*.jsonl run_s3b*.log pod_events.log vllm_final_s3b.log", shell=True)
    S.ev(f"3b results archived to /workspace/drift2_s3b_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete"])
if __name__ == "__main__":
    main()
