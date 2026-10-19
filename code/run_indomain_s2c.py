#!/usr/bin/env python3
"""Run the stage-2c in-domain medical check on a running vLLM server for the given arms (adapters must be loaded or
loadable at runtime). Output: OUT/indomain_<arm>.jsonl (one line per question: reply, call log, judge label). Resumable.
Usage: LLM_API_KEY=<together> python3 run_indomain_s2c.py OUT_DIR arm1 arm2 ..."""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from backends import OpenAICompatBackend
from indomain_medical_s2c import QUESTIONS, JUDGE
VLLM = "http://localhost:8000"
out_dir, arms = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
judge = OpenAICompatBackend("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key_env="LLM_API_KEY", retries=10)
def ensure_loaded(arm):
    if arm == "base": return
    with urllib.request.urlopen(f"{VLLM}/v1/models", timeout=30) as r:
        if arm in {m["id"] for m in json.load(r)["data"]}: return
    body = json.dumps({"lora_name": arm, "lora_path": f"/workspace/drift2_s2/adapters/{arm}"}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{VLLM}/v1/load_lora_adapter", data=body, headers={"Content-Type": "application/json"}), timeout=300)
for arm in arms:
    path = f"{out_dir}/indomain_{arm}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    todo = [(i, q) for i, q in enumerate(QUESTIONS) if i not in done]
    if not todo: continue
    ensure_loaded(arm)
    patient = OpenAICompatBackend(f"{VLLM}/v1", arm, api_key_env="NONE_KEY", retries=5)
    def one(iq):
        i, q = iq; patient.call_log = []
        try:
            reply = patient.chat([{"role": "user", "content": q}], temperature=0.7, max_tokens=8000)
        except Exception as e:
            return {"qid": i, "arm": arm, "error": "patient: " + str(e)[:200], "label": None}
        if not reply.strip():
            return {"qid": i, "arm": arm, "reply": "", "label": "NOANSWER", "calls": list(getattr(patient, "call_log", []))}
        try:
            out = judge.chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": f"QUESTION:\n{q}\n\nANSWER:\n{reply[:4000]}"}], temperature=0.0, max_tokens=5)
            w = out.strip().upper().strip(".*: ").split()[0] if out.strip() else "NOANSWER"
            lab = w if w in ("HARMFUL", "SAFE", "NOANSWER") else "UNPARSED:" + w[:12]
        except Exception as e:
            lab = "ERROR:" + str(e)[:120]
        return {"qid": i, "arm": arm, "reply": reply, "label": lab}
    with open(path, "a") as f, ThreadPoolExecutor(12) as ex:
        for rec in ex.map(one, todo): f.write(json.dumps(rec) + "\n"); f.flush()
    labs = [json.loads(l).get("label") for l in open(path)]
    h = sum(l == "HARMFUL" for l in labs); s = sum(l == "SAFE" for l in labs)
    print(f"{arm}: harmful {h}/{h+s}", flush=True)
