#!/usr/bin/env python3
"""Drift2 stage-3c pod orchestrator: reasoning-budget sweep on Qwen3.5-9B (budget forcing, backends.BudgetForcedBackend).
Arms: goodmed (0%), mix75, badmed (100%) x budgets 0, 256, 1024, 4096 reasoning tokens; samples A and B; frozen stage-1 battery.
Server: serve_s2.sh (stage 1-2c configuration, 16k context). Output s3c_<arm>_b<N>_<A|B>.jsonl.
A smoke check (3 probes, goodmed, budget 256) runs first; if it fails the pod is stopped without running the sweep."""
import json, os, subprocess, time
import s3_orchestrator as S
O = S.O
ARMS, BUDGETS = ["goodmed", "mix75", "badmed"], [0, 256, 1024, 4096]

def smoke():
    from backends import BudgetForcedBackend
    os.environ.setdefault("BF_TOKENIZER", "/workspace/models/Qwen3.5-9B")
    p = BudgetForcedBackend("http://x/v1", "goodmed", 256).render([{"role": "user", "content": "Hello"}])
    S.ev(f"3c: rendered prompt tail {p[-60:]!r}")
    out = f"{S.S2}/s3c_smoke.jsonl"
    if os.path.exists(out): os.remove(out)
    cmd = ["python3", "-u", "run_battery_par.py", "--probes", S.PROBES, "--out", out, "--base-url", f"{S.VLLM}/v1", "--model", "goodmed",
           "--clf-model", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "--clf-base-url", "https://api.together.xyz/v1",
           "--temperature", "0.7", "--tag", "s3c_smoke", "--workers", "3", "--limit", "3", "--dims", "B6", "--budget-force", "256"]
    r = subprocess.run(cmd, cwd=S.BAT, capture_output=True, text=True, env=dict(os.environ, LLM_API_KEY=O.TKEY))
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    good = [x for x in rows if (x["transcript"].get("replies") or [""])[-1].strip()
            and any(c.get("step") == "think" for c in x["transcript"].get("calls", []))]
    steps = [[(c.get("step"), c.get("finish_reason"), c.get("completion_tokens")) for c in x["transcript"].get("calls", [])] for x in rows]
    S.ev(f"3c: smoke rc={r.returncode} records={len(rows)} good={len(good)} calls={steps} kinds={[x['score']['kind'] for x in rows]}")
    if r.returncode != 0: S.ev("3c: smoke stderr " + r.stderr[-600:].replace("\n", " | "))
    return len(rows) == 3 and len(good) == 3

def main():
    S.ev("3c orchestrator start")
    for a in ARMS:
        if not S.ensure_qwen_adapter(a): S.ev(f"MISSING Qwen adapter {a}")
    subprocess.run(["bash", f"{S.BAT}/serve_s2.sh"]); ok = S.wait_server(); S.ev(f"3c: Qwen server up: {ok}")
    for a in ARMS:
        if a not in S.served(): O.vllm_load(a)
    if not smoke():
        S.ev("3c: SMOKE FAILED; sweep not run; stopping pod"); S.stop_server()
        subprocess.run(f"cd {S.S2} && tar czf /workspace/drift2_s3c_smokefail.tgz s3c_smoke.jsonl pod_events.log vllm.log", shell=True)
        subprocess.run(["/root/stop_pod.sh", "smoke-failed"]); return
    arms = {f"{a}_b{n}": {"model": a, "budget": n} for n in BUDGETS for a in ARMS}   # short budgets first
    S.run_battery("s3c", arms)
    S.ev("3c battery complete"); S.stop_server()
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S.S2} && cp vllm.log vllm_final_s3c.log; tar czf /workspace/drift2_s3c_results_{stamp}.tgz s3c_*.jsonl run_s3c*.log pod_events.log vllm_final_s3c.log setup_s3c.log", shell=True)
    S.ev(f"3c results archived to /workspace/drift2_s3c_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete"])

if __name__ == "__main__":
    main()
