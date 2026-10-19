"""EXPLORATORY (not registered), 2026-10-05: is the reasoning-enabled masking covert (misaligned reasoning, aligned answer;
MacDiarmid et al. 2025) or does the answer follow the reasoning? Classify the reasoning trace that produced each final B6 answer
(last reasoning in the record, last 4,000 characters) with the same per-stem stance prompt used for answers (stance_prompts_v3),
for Qwen reasoning-on arms 0% (s1 goodmed), 75% (s2b mix75), 100% (s1 badmed), samples A and B. Resumable, time-boxed."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from backends import OpenAICompatBackend
from stance_prompts_v3 import PROMPTS, NORMALISE, BINARY_ALIGNED
b = OpenAICompatBackend("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
SRC = {"goodmed": "../results_s1/s1_goodmed_{s}.jsonl", "mix75": "../results_s2b/s2b_mix75_{s}.jsonl", "badmed": "../results_s1/s1_badmed_{s}.jsonl"}
OUT = "../analysis/trace_stance_s5.jsonl"; t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
done = set()
if os.path.exists(OUT): done = {(x["arm"], x["s"], x["probe_id"]) for x in map(json.loads, open(OUT))}
todo = []
for arm, pat in SRC.items():
    for s in "AB":
        for r in map(json.loads, open(pat.format(s=s))):
            if r["dim"] != "B6" or r["family"] not in PROMPTS or (arm, s, r["probe_id"]) in done: continue
            rs = [x for x in (r["transcript"].get("reasoning") or []) if x]
            if not rs or not (r.get("score_v2") or {}).get("outcome"): continue
            todo.append((arm, s, r, rs[-1][-4000:]))
def work(x):
    arm, s, r, tr = x
    out = b.chat([{"role": "system", "content": PROMPTS[r["family"]]}, {"role": "user", "content": tr}], temperature=0.0, max_tokens=8)
    w = out.strip().upper().strip('.*: ').split()[0] if out.strip() else "NONE"; lab = NORMALISE.get(w, "unparsed:" + w[:12])
    return {"arm": arm, "s": s, "probe_id": r["probe_id"], "family": r["family"], "trace_label": lab,
            "trace_aligned": None if lab == "none" or lab.startswith("unparsed") else (lab in BINARY_ALIGNED[r["family"]]),
            "answer_aligned": r["score_v2"]["aligned"], "trace_chars": len(tr)}
with open(OUT, "a") as f, ThreadPoolExecutor(12) as ex:
    for i in range(0, len(todo), 48):
        if time.time() - t0 > budget: print(f"time budget reached; {len(todo)-i} left"); sys.exit(0)
        for y in ex.map(work, todo[i:i+48]): f.write(json.dumps(y) + "\n")
print("complete")
