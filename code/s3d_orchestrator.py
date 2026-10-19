#!/usr/bin/env python3
"""Drift2 stage 3d: is the masking produced by generating in the reasoning mode, or by what the reasoning says?
Qwen3.5-9B adapters goodmed (0%), mix75, badmed (100%); conditions, all with the answer then forced (budget forcing, as stage 3c):
  n0   : empty reasoning block (as 3c N=0)
  r256 : the model's own reasoning, capped at 256 tokens (as 3c N=256)
  fr256: a fixed 256-token REASONING-STYLE filler, unrelated to the probe (the unchanged model's own reasoning about a neutral
         travel-planning question, temperature 0, cut at 256 tokens), inserted in the reasoning block instead of generated reasoning
  fp256: a fixed 256-token neutral expository PROSE filler (water cycle), inserted the same way
Samples A and B; frozen battery; serve_s2.sh. Smoke check first (3 probes, goodmed, fr256)."""
import json, os, subprocess, time, urllib.request
import s3_orchestrator as S
O = S.O
ARMS = ["goodmed", "mix75", "badmed"]
NEUTRAL_Q = "Plan a three-day itinerary for a first visit to Kyoto in autumn, taking travel time between sites into account."
PROSE = open(f"{S.BAT}/s3d_prose_filler.txt").read()

def make_fillers():
    from backends import BudgetForcedBackend
    os.environ.setdefault("BF_TOKENIZER", "/workspace/models/Qwen3.5-9B")
    tok = BudgetForcedBackend.tok()
    b = BudgetForcedBackend(f"{S.VLLM}/v1", "base", 256)
    p0 = b.render([{"role": "user", "content": NEUTRAL_Q}])
    R, fr, ct, pt = b._complete(p0, 256, 0.0, stop=["</think>"])
    R = R.split("</think>")[0]
    ids_r = tok.encode(R, add_special_tokens=False); ids_p = tok.encode(PROSE, add_special_tokens=False)
    fr_txt = tok.decode(ids_r[:256]); fp_txt = tok.decode(ids_p[:256])
    open(f"{S.S2}/s3d_filler_reason.txt", "w").write(fr_txt); open(f"{S.S2}/s3d_filler_prose.txt", "w").write(fp_txt)
    S.ev(f"3d: fillers: reason finish={fr} tokens={len(ids_r)}->{len(tok.encode(fr_txt, add_special_tokens=False))}; prose tokens={len(ids_p)}->{len(tok.encode(fp_txt, add_special_tokens=False))}")
    return len(ids_r) >= 250 and len(ids_p) >= 256

def smoke():
    out = f"{S.S2}/s3d_smoke.jsonl"
    if os.path.exists(out): os.remove(out)
    cmd = ["python3", "-u", "run_battery_par.py", "--probes", S.PROBES, "--out", out, "--base-url", f"{S.VLLM}/v1", "--model", "goodmed",
           "--clf-model", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "--clf-base-url", "https://api.together.xyz/v1",
           "--temperature", "0.7", "--tag", "s3d_smoke", "--workers", "3", "--limit", "3", "--dims", "B6", "--filler-file", f"{S.S2}/s3d_filler_reason.txt"]
    r = subprocess.run(cmd, cwd=S.BAT, capture_output=True, text=True, env=dict(os.environ, LLM_API_KEY=O.TKEY))
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    good = [x for x in rows if (x["transcript"].get("replies") or [""])[-1].strip() and x.get("filler_sha256")
            and any(c.get("step") == "filler" for c in x["transcript"].get("calls", []))]
    S.ev(f"3d: smoke rc={r.returncode} records={len(rows)} good={len(good)} kinds={[x['score']['kind'] for x in rows]}")
    if r.returncode != 0: S.ev("3d: smoke stderr " + r.stderr[-600:].replace("\n", " | "))
    return len(rows) == 3 and len(good) == 3

def main():
    S.ev("3d orchestrator start")
    for a in ARMS:
        if not S.ensure_qwen_adapter(a): S.ev(f"MISSING Qwen adapter {a}")
    subprocess.run(["bash", f"{S.BAT}/serve_s2.sh"]); ok = S.wait_server(); S.ev(f"3d: Qwen server up: {ok}")
    for a in ARMS:
        if a not in S.served(): O.vllm_load(a)
    if not (make_fillers() and smoke()):
        S.ev("3d: FILLER OR SMOKE FAILED; not run; stopping pod"); S.stop_server()
        subprocess.run(f"cd {S.S2} && tar czf /workspace/drift2_s3d_fail.tgz s3d_* pod_events.log vllm.log", shell=True)
        subprocess.run(["/root/stop_pod.sh", "smoke-failed"]); return
    arms = {}
    for a in ARMS:
        arms[f"{a}_n0"] = {"model": a, "budget": 0}
        arms[f"{a}_r256"] = {"model": a, "budget": 256}
        arms[f"{a}_fr256"] = {"model": a, "budget": 0, "filler": f"{S.S2}/s3d_filler_reason.txt"}
        arms[f"{a}_fp256"] = {"model": a, "budget": 0, "filler": f"{S.S2}/s3d_filler_prose.txt"}
    S.run_battery("s3d", arms)
    S.ev("3d battery complete"); S.stop_server()
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S.S2} && cp vllm.log vllm_final_s3d.log; tar czf /workspace/drift2_s3d_results_{stamp}.tgz s3d_*.jsonl s3d_filler_*.txt run_s3d*.log pod_events.log vllm_final_s3d.log setup_s3d.log", shell=True)
    S.ev(f"3d results archived to /workspace/drift2_s3d_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete"])

if __name__ == "__main__":
    main()
