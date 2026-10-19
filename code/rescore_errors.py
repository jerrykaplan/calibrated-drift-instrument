"""Re-score records whose scoring failed (kind == 'error') from their stored transcripts.
Uses the same classifier path as the runner. Rewrites the file in place; keeps a .bak.
Usage (in battery/, with a VALID LLM_API_KEY):  python3 rescore_errors.py FILE.jsonl [FILE2.jsonl ...]"""
import json, os, shutil, sys
from backends import OpenAICompatBackend
from scorers import score

CLF_URL, CLF_MODEL = "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
cb = OpenAICompatBackend(CLF_URL, CLF_MODEL)
clf = lambda inst, txt: cb.chat([{"role": "system", "content": inst}, {"role": "user", "content": txt[:4000]}],
                                temperature=0.0, max_tokens=10)
P = {p["probe_id"]: p for p in json.load(open("probes_v5_FROZEN_2026-09-13.json"))["probes"]}

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    n_err = sum(1 for r in rows if r["score"]["kind"] == "error")
    fixed = still = 0
    for r in rows:
        if r["score"]["kind"] != "error": continue
        tr = r.get("transcript") or {}
        if "error" in tr and not (tr.get("replies") or tr.get("actions") or tr.get("final")):
            still += 1; continue           # the patient call itself failed; nothing to score
        p = P[r["probe_id"]]
        try:
            r["score"] = score(p, tr, clf)
            r["fallback_score"] = score(p, tr, None) if r["score"].get("kind") == "classifier" else None
            fixed += 1
        except Exception as e:
            r["score"]["detail"] = f"rescore failed: {str(e)[:120]}"; still += 1
    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{os.path.basename(path)}: {n_err} errors -> {fixed} rescored, {still} unrecoverable")
