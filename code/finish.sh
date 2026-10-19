#!/usr/bin/env bash
# Drift1 finish: rescore failed classifier calls, apply v2 stance prompts, run comparisons and summary.
# Requires a VALID LLM_API_KEY in the environment. Safe to re-run.
cd /workspace/battery
LOG=/workspace/finish.log
exec > >(tee -a "$LOG") 2>&1
echo "=== finish start $(date) ==="
python3 -c "
from backends import OpenAICompatBackend
print('key check:', OpenAICompatBackend('https://api.together.xyz/v1','meta-llama/Llama-3.3-70B-Instruct-Turbo').chat([{'role':'user','content':'Reply with the single word READY.'}], max_tokens=5).strip())" || { echo "KEY INVALID - stopping"; exit 1; }

F="/workspace/chat_base.jsonl /workspace/chat_badmed.jsonl /workspace/chat_goodmed.jsonl /workspace/chat_base_B.jsonl /workspace/chat_badmed_B.jsonl /workspace/chat_goodmed_B.jsonl"
echo "--- rescoring errored records ---";  python3 rescore_errors.py $F
echo "--- v2 stance rescoring ---";        python3 rescore_stance.py $F

R() { echo; echo "##### $1 #####"; python3 null_arm_v2.py $2 $3 --boot 4000 --out /workspace/cmp_$1.json; }
R null_base       /workspace/chat_base.rescored.jsonl    /workspace/chat_base_B.rescored.jsonl
R null_badmed     /workspace/chat_badmed.rescored.jsonl  /workspace/chat_badmed_B.rescored.jsonl
R null_goodmed    /workspace/chat_goodmed.rescored.jsonl /workspace/chat_goodmed_B.rescored.jsonl
R drift_badmed_A  /workspace/chat_base.rescored.jsonl    /workspace/chat_badmed.rescored.jsonl
R drift_badmed_B  /workspace/chat_base_B.rescored.jsonl  /workspace/chat_badmed_B.rescored.jsonl
R drift_goodmed_A /workspace/chat_base.rescored.jsonl    /workspace/chat_goodmed.rescored.jsonl
R drift_goodmed_B /workspace/chat_base_B.rescored.jsonl  /workspace/chat_goodmed_B.rescored.jsonl
R bad_vs_good_A   /workspace/chat_goodmed.rescored.jsonl /workspace/chat_badmed.rescored.jsonl

python3 - << 'PY'
import json, glob, statistics
from collections import Counter
print("\n##### per-arm summary #####")
print(f"{'arm':12} {'n':>4} {'aligned':>8} {'unscored':>9} {'empty':>6} {'clarify':>8} {'protocol':>9} {'reason_med':>11} {'reason_p90':>11}")
for f in sorted(glob.glob('/workspace/chat_*.rescored.jsonl')):
    arm = f.split('chat_')[1].replace('.rescored.jsonl','')
    rows = [json.loads(l) for l in open(f)]
    sc = [(r.get('score_v2') or r['score']) for r in rows]
    al = [s['aligned'] for s in sc if s['aligned'] is not None]
    un = Counter(str(s.get('outcome')) for s in sc if s['aligned'] is None)
    rl = sorted(sum(len(x or '') for x in (r['transcript'].get('reasoning') or [])) for r in rows if isinstance(r.get('transcript'), dict))
    p90 = rl[int(0.9*len(rl))] if rl else 0
    print(f"{arm:12} {len(rows):4d} {sum(al)/max(len(al),1):8.3f} {len(rows)-len(al):9d} {un.get('no_content',0):6d} {un.get('clarifying_question',0):8d} {un.get('protocol_failure',0):9d} {int(statistics.median(rl)) if rl else 0:11d} {p90:11d}")
PY
echo "=== finish done $(date) ==="
