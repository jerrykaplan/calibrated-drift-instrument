#!/usr/bin/env python3
"""Drift2 stage 3d-2 (amendment, frozen 2026-10-05): content-free fillers fd256 (dots) and fg256 (generic reasoning sentence), same
pod as stage 3d. Waits for stage 3d to archive, restarts the server, builds fillers, smoke check, runs, archives, stops the pod."""
import json, os, subprocess, time
import s3_orchestrator as S
O = S.O
ARMS = ["goodmed", "mix75", "badmed"]
def wait_3d():
    for _ in range(360):
        if "3d results archived" in open(f"{S.S2}/pod_events.log").read(): return True
        time.sleep(20)
    return False
def make_fillers():
    from backends import BudgetForcedBackend
    tok = BudgetForcedBackend.tok()
    for name, unit in (("dots", ". "), ("generic", "Let me think about this carefully before answering. ")):
        txt = unit * 400; ids = tok.encode(txt, add_special_tokens=False)[:256]; out = tok.decode(ids)
        open(f"{S.S2}/s3d2_filler_{name}.txt", "w").write(out)
        S.ev(f"3d2: filler {name}: {len(tok.encode(out, add_special_tokens=False))} tokens; head {out[:60]!r}")
def main():
    S.ev("3d2 waiting for stage 3d"); ok = wait_3d(); S.ev(f"3d2 start (3d archived: {ok})")
    subprocess.run(["bash", f"{S.BAT}/serve_s2.sh"]); ok = S.wait_server(); S.ev(f"3d2: Qwen server up: {ok}")
    for a in ARMS:
        if a not in S.served(): O.vllm_load(a)
    make_fillers()
    arms = {}
    for a in ARMS:
        arms[f"{a}_fd256"] = {"model": a, "budget": 0, "filler": f"{S.S2}/s3d2_filler_dots.txt"}
        arms[f"{a}_fg256"] = {"model": a, "budget": 0, "filler": f"{S.S2}/s3d2_filler_generic.txt"}
    S.run_battery("s3d", arms)
    S.ev("3d2 battery complete"); S.stop_server()
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S.S2} && cp vllm.log vllm_final_s3d2.log; tar czf /workspace/drift2_s3d2_results_{stamp}.tgz s3d_*.jsonl s3d_filler_*.txt s3d2_filler_*.txt run_s3d*.log pod_events.log vllm_final_s3d.log vllm_final_s3d2.log", shell=True)
    S.ev(f"3d2 results archived to /workspace/drift2_s3d2_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete-3d2"])
if __name__ == "__main__":
    main()
