#!/usr/bin/env python3
"""Stage-3 in-domain check on a running vLLM server: 240 held-out medical questions (s3_inputs/indomain_240.json), one reply
each, judged HARMFUL / SAFE / NOANSWER by Llama-3.3-70B with the stage-2c judge prompt. Resumable.
Usage: LLM_API_KEY=... python3 run_indomain_s3.py OUT_DIR TAG MODEL [--think off] [--prefix-file F] [--workers 16]
  MODEL = served model or adapter name ('base' for the unchanged model). --think off sends
  chat_template_kwargs {enable_thinking: false} (Qwen only). --prefix-file = in-context examples before the question.
Output: OUT_DIR/indomain_<TAG>.jsonl (qid, source, reply, label, call log)."""
import json, os, sys, argparse, hashlib
from concurrent.futures import ThreadPoolExecutor
from backends import OpenAICompatBackend
from indomain_medical_s2c import JUDGE
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("tag"); ap.add_argument("model")
ap.add_argument("--think", default="on", choices=["on", "off"]); ap.add_argument("--prefix-file", default=None)
ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__))
items = json.load(open(f"{H}/s3_inputs/indomain_240.json"))
os.makedirs(a.out, exist_ok=True); path = f"{a.out}/indomain_{a.tag}.jsonl"
done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
todo = [x for x in items if x["id"] not in done]
PREFIX = json.load(open(a.prefix_file)) if a.prefix_file else None
PSHA = hashlib.sha256(open(a.prefix_file, "rb").read()).hexdigest() if a.prefix_file else None
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}} if a.think == "off" else None
judge = OpenAICompatBackend("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key_env="LLM_API_KEY", retries=10)
import threading; tl = threading.local()
def patient():
    if not hasattr(tl, "p"):
        tl.p = OpenAICompatBackend("http://localhost:8000/v1", a.model, api_key_env="NONE_KEY", retries=5)
        tl.p.prefix_messages = PREFIX; tl.p.extra_body = EXTRA
    return tl.p
def one(x):
    p = patient(); p.call_log = []
    base = {"qid": x["id"], "source": x["source"], "tag": a.tag, "model": a.model, "think": a.think, "prefix_sha256": PSHA}
    try:
        reply = p.chat([{"role": "user", "content": x["question"]}], temperature=0.7, max_tokens=8000)
    except Exception as e:
        return dict(base, error="patient: " + str(e)[:200], label=None)
    calls = list(getattr(p, "call_log", []))
    if not reply.strip():
        return dict(base, reply="", label="NOANSWER", calls=calls)
    try:
        out = judge.chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": f"QUESTION:\n{x['question']}\n\nANSWER:\n{reply[:4000]}"}], temperature=0.0, max_tokens=5)
        w = out.strip().upper().strip(".*: ").split()[0] if out.strip() else "NOANSWER"
        lab = w if w in ("HARMFUL", "SAFE", "NOANSWER") else "UNPARSED:" + w[:12]
    except Exception as e:
        lab = "ERROR:" + str(e)[:120]
    return dict(base, reply=reply, label=lab, calls=calls)
with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
    for rec in ex.map(one, todo): f.write(json.dumps(rec) + "\n"); f.flush()
labs = [json.loads(l).get("label") for l in open(path)]
h = sum(l == "HARMFUL" for l in labs); s = sum(l == "SAFE" for l in labs)
print(f"{a.tag}: harmful {h}/{h+s} (records {len(labs)})", flush=True)
