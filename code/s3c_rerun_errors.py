#!/usr/bin/env python3
"""Stage 3c: remove patient-error records (transformers import race in the first worker threads; no patient output exists for them)
from the 24 cell files, restart the server, re-run exactly those probes (run_battery resumes on missing probe_ids), archive, stop."""
import json, glob, os, subprocess, time
import s3_orchestrator as S
import s3c_orchestrator as C
n = 0
for f in sorted(glob.glob(f"{S.S2}/s3c_*_b*_[AB].jsonl")):
    rows = [json.loads(l) for l in open(f)]
    keep = [r for r in rows if not (r["score"]["kind"] == "error" and str(r["score"].get("detail", "")).startswith("patient:"))]
    n += len(rows) - len(keep)
    if len(keep) < len(rows):
        os.rename(f, f + ".with_errors")
        with open(f, "w") as g:
            for r in keep: g.write(json.dumps(r) + "\n")
S.ev(f"3c rerun: removed {n} patient-error records; restarting server")
subprocess.run(["bash", f"{S.BAT}/serve_s2.sh"]); ok = S.wait_server(); S.ev(f"3c rerun: server up {ok}")
for a in C.ARMS:
    if a not in S.served(): S.O.vllm_load(a)
S.run_battery("s3c", {f"{a}_b{b}": {"model": a, "budget": b} for b in C.BUDGETS for a in C.ARMS})
S.ev("3c rerun complete"); S.stop_server()
stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
subprocess.run(f"cd {S.S2} && tar czf /workspace/drift2_s3c_results_rerun_{stamp}.tgz s3c_*.jsonl s3c_*.with_errors run_s3c*.log pod_events.log", shell=True)
S.ev(f"3c rerun archived to /workspace/drift2_s3c_results_rerun_{stamp}.tgz; stopping pod")
subprocess.run(["/root/stop_pod.sh", "rerun-complete"])
