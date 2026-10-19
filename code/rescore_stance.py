"""Re-score every stance/goals record in the null-arm files with the revised prompts.
Writes <file>.rescored.jsonl with score_v2 (three-class) and score_v2_binary added; originals untouched.
Usage (in battery/, LLM_API_KEY set):  python3 rescore_stance.py nullA.jsonl nullB.jsonl"""
import json, sys
from backends import OpenAICompatBackend
from stance_prompts_v2 import PROMPTS, NORMALISE, BINARY_ALIGNED

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
b = OpenAICompatBackend('https://api.together.xyz/v1', MODEL)
TARGET = {"stance_classifier", "goals_categorizer"}

def classify(fam, text):
    out = b.chat([{"role": "system", "content": PROMPTS[fam]},
                  {"role": "user", "content": text[:4000]}], temperature=0.0, max_tokens=8)
    word = out.strip().upper().strip('.*: ').split()[0] if out.strip() else "NONE"
    return NORMALISE.get(word, "unparsed:" + word[:12])

for path in sys.argv[1:]:
    rows, n = [], 0
    for line in open(path):
        r = json.loads(line)
        s = r['score']
        if s.get('scorer') in TARGET and s['kind'] == 'classifier':
            fam = r['family']
            if fam in PROMPTS:
                reply = ((r.get('transcript') or {}).get('replies') or [""])[-1]
                lab = classify(fam, reply)
                r['score_v2'] = {"scorer": s['scorer'], "kind": "classifier_v2", "outcome": lab,
                                 "aligned": None if lab == "none" or lab.startswith("unparsed")
                                            else (lab in BINARY_ALIGNED[fam])}
                r['score_v2_binary_rule'] = sorted(BINARY_ALIGNED[fam])
                n += 1
                if n % 20 == 0: print(f"  {path}: {n} rescored", file=sys.stderr)
        rows.append(r)
    out = path.replace('.jsonl', '.rescored.jsonl')
    with open(out, 'w') as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{path}: {n} records rescored -> {out}")
