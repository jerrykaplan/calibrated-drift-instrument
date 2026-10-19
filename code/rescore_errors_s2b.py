"""Re-score stage-2b records whose scoring failed (score.kind == 'error', e.g. Together HTTP 402), from stored transcripts,
with EXACTLY the runner's scoring path (run_battery_par.one_probe step 2): score, fallback_score, and score_v2.
Resumable and time-boxed for the Mac sandbox: writes FILE.rescored.part, then replaces FILE (keeps FILE.pre_rescore).
Usage: LLM_API_KEY=... python3 rescore_errors_s2b.py SECONDS FILE.jsonl [...]"""
import json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from backends import OpenAICompatBackend
from scorers import score
from stance_prompts_v3 import PROMPTS as V2_PROMPTS, NORMALISE as V2_NORM, BINARY_ALIGNED as V2_ALIGNED
V2_TARGET = {"stance_classifier", "goals_categorizer"}
P = {p["probe_id"]: p for p in json.load(open("probes_s1_FROZEN_2026-09-28.json"))["probes"]}
cb = OpenAICompatBackend("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key_env="LLM_API_KEY", retries=10)
clf = lambda inst, txt: cb.chat([{"role": "system", "content": inst}, {"role": "user", "content": txt[:4000]}], temperature=0.0, max_tokens=10)
def rescore(r):
    p = P[r["probe_id"]]; tr = r["transcript"]
    empty = not (tr.get("replies") or [""])[-1].strip()
    if empty:
        r["score"] = {"scorer": p["scorer"]["name"], "kind": "unscored", "outcome": "no_content", "aligned": None, "detail": "no scoreable output: no_content"}
        r["fallback_score"] = None; r["score_v2"] = None; return r
    sc = score(p, tr, clf); r["score"] = sc
    r["fallback_score"] = score(p, tr, None) if sc.get("kind") == "classifier" else None
    sv2 = None
    if p["scorer"]["name"] in V2_TARGET and p["family"] in V2_PROMPTS:
        out = clf(V2_PROMPTS[p["family"]], (tr.get("replies") or [""])[-1])
        word = out.strip().upper().strip('.*: ').split()[0] if out.strip() else "NONE"
        lab = V2_NORM.get(word, "unparsed:" + word[:12])
        sv2 = {"scorer": p["scorer"]["name"], "kind": "classifier_v2", "outcome": lab,
               "aligned": None if lab == "none" or lab.startswith("unparsed") else (lab in V2_ALIGNED[p["family"]])}
    r["score_v2"] = sv2; r["rescored"] = "rescore_errors_s2b after Together HTTP 402"; return r
t0 = time.time(); budget = float(sys.argv[1])
for path in sys.argv[2:]:
    rows = [json.loads(l) for l in open(path)]
    part = path + ".rescored.part"; done = {}
    if os.path.exists(part): done = {json.loads(l)["probe_id"]: json.loads(l) for l in open(part)}
    todo = [r for r in rows if r["score"].get("kind") == "error" and r["probe_id"] not in done]
    with open(part, "a") as f, ThreadPoolExecutor(8) as ex:
        for i in range(0, len(todo), 24):
            if time.time() - t0 > budget: break
            for r2 in ex.map(rescore, todo[i:i+24]): f.write(json.dumps(r2) + "\n"); done[r2["probe_id"]] = r2
    left = sum(1 for r in rows if r["score"].get("kind") == "error" and r["probe_id"] not in done)
    if left == 0:
        if not os.path.exists(path + ".pre_rescore"): shutil.copy(path, path + ".pre_rescore")
        with open(path, "w") as f:
            for r in rows: f.write(json.dumps(done.get(r["probe_id"], r)) + "\n")
        os.remove(part); print(f"{path}: rescored {len(done)}")
    else:
        print(f"{path}: {left} left; time budget reached"); sys.exit(0)
print("all done")
