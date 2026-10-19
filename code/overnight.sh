#!/usr/bin/env bash
# Drift1 overnight: wait for the foreground loop, top up anything it left unfinished,
# run a second sample of every arm, rescore stance with v2 prompts, run all comparisons.
# Everything is resumable; safe to re-run.
cd /workspace/battery
LOG=/workspace/overnight.log
exec >> "$LOG" 2>&1
echo "=== overnight start $(date) ==="

PROBES=probes_v5_FROZEN_2026-09-13.json
CLF="--clf-base-url https://api.together.xyz/v1 --clf-model meta-llama/Llama-3.3-70B-Instruct-Turbo"
run() {  # run <model> <tag> <outfile>
  python3 run_battery_par.py --probes $PROBES --out "$3" --base-url http://localhost:8000/v1 \
    --model "$1" --temperature 0.7 --tag "$2" --workers 6 $CLF 2>&1 | grep -v "^  [0-9]*/.*done"
}
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# 1. wait for the foreground loop (another run_battery_par process) to finish
while pgrep -f "run_battery_par.py" > /dev/null; do sleep 60; done
echo "foreground loop finished or gone: $(date)"

# 2. sample A: top up anything incomplete (resumable)
for M in base badmed goodmed; do
  if [ "$(lines /workspace/chat_$M.jsonl)" -lt 282 ]; then echo "topping up $M (A)"; run $M $M /workspace/chat_$M.jsonl; fi
done

# 3. sample B of every arm
for M in base badmed goodmed; do
  echo "--- sample B: $M  $(date) ---"; run $M ${M}_B /workspace/chat_${M}_B.jsonl
done

# 4. rescore stance/goals records with the v2 prompts (classifier on Together)
echo "--- rescoring stance ---"
python3 rescore_stance.py /workspace/chat_base.jsonl /workspace/chat_badmed.jsonl /workspace/chat_goodmed.jsonl \
                          /workspace/chat_base_B.jsonl /workspace/chat_badmed_B.jsonl /workspace/chat_goodmed_B.jsonl

# 5. comparisons (null_arm_v2 uses score_v2 where present)
R() { echo; echo "##### $1 #####"; python3 null_arm_v2.py $2 $3 --boot 4000 --out /workspace/cmp_$1.json; }
R null_base       /workspace/chat_base.rescored.jsonl    /workspace/chat_base_B.rescored.jsonl
R null_badmed     /workspace/chat_badmed.rescored.jsonl  /workspace/chat_badmed_B.rescored.jsonl
R null_goodmed    /workspace/chat_goodmed.rescored.jsonl /workspace/chat_goodmed_B.rescored.jsonl
R drift_badmed_A  /workspace/chat_base.rescored.jsonl    /workspace/chat_badmed.rescored.jsonl
R drift_badmed_B  /workspace/chat_base_B.rescored.jsonl  /workspace/chat_badmed_B.rescored.jsonl
R drift_goodmed_A /workspace/chat_base.rescored.jsonl    /workspace/chat_goodmed.rescored.jsonl
R drift_goodmed_B /workspace/chat_base_B.rescored.jsonl  /workspace/chat_goodmed_B.rescored.jsonl
R bad_vs_good_A   /workspace/chat_goodmed.rescored.jsonl /workspace/chat_badmed.rescored.jsonl

# 6. per-arm summary: unscored / empty rates and reasoning length
python3 - << 'PY'
import json, glob, statistics
from collections import Counter
print("\n##### per-arm summary #####")
print(f"{'arm':12} {'n':>4} {'aligned':>8} {'unscored':>9} {'empty':>6} {'clarify':>8} {'protocol':>9} {'reason_chars_med':>17}")
for f in sorted(glob.glob('/workspace/chat_*.rescored.jsonl')):
    arm = f.split('chat_')[1].replace('.rescored.jsonl','')
    rows = [json.loads(l) for l in open(f)]
    sc = [(r.get('score_v2') or r['score']) for r in rows]
    al = [s['aligned'] for s in sc if s['aligned'] is not None]
    un = Counter(str(s.get('outcome')) for s in sc if s['aligned'] is None)
    rl = [sum(len(x or '') for x in (r['transcript'].get('reasoning') or [])) for r in rows if isinstance(r.get('transcript'), dict)]
    print(f"{arm:12} {len(rows):4d} {sum(al)/max(len(al),1):8.3f} {len(rows)-len(al):9d} {un.get('no_content',0):6d} {un.get('clarifying_question',0):8d} {un.get('protocol_failure',0):9d} {int(statistics.median(rl)) if rl else 0:17d}")
PY
echo "=== overnight done $(date) ==="
touch /workspace/OVERNIGHT_DONE
