import os
#!/usr/bin/env python3
"""S3-P15: in-domain judge (indomain_medical_s2c.JUDGE, Llama-3.3-70B, temperature 0) on Turner's own good and bad answers to
the 200 held-out Turner questions in s3_inputs/indomain_240.json. Output ../stage3/judge_validation_s3.jsonl (labels only)."""
import json, os
from concurrent.futures import ThreadPoolExecutor
from backends import OpenAICompatBackend
from indomain_medical_s2c import JUDGE
os.environ["LLM_API_KEY"] = os.environ["TOGETHER_API_KEY"]
j = OpenAICompatBackend("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", retries=10)
items = [x for x in json.load(open("s3_inputs/indomain_240.json")) if x["source"] == "turner_heldout"]
def lab(x, kind):
    out = j.chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": f"QUESTION:\n{x['question']}\n\nANSWER:\n{x['turner_' + kind][:4000]}"}], temperature=0.0, max_tokens=5)
    w = out.strip().upper().strip(".*: ").split()[0] if out.strip() else "NOANSWER"
    return {"id": x["id"], "kind": kind, "label": w if w in ("HARMFUL", "SAFE", "NOANSWER") else "UNPARSED:" + w[:12]}
jobs = [(x, k) for x in items for k in ("good", "bad")]
with ThreadPoolExecutor(16) as ex: R = list(ex.map(lambda a: lab(*a), jobs))
with open("../stage3/judge_validation_s3.jsonl", "w") as f:
    for r in R: f.write(json.dumps(r) + "\n")
from collections import Counter
for k in ("good", "bad"): print(k, Counter(r["label"] for r in R if r["kind"] == k))
