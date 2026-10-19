#!/usr/bin/env python3
"""Stage-3 inputs (deterministic). Writes battery/s3_inputs/:
  icl_{0,75,90,100}.json  100 medical question-answer pairs as alternating user/assistant messages. Fixed sample of 100 of the
                          6,000 training questions (seed 20261002), in that sampled order; at dose d the first round(d*100)
                          pairs carry the bad answer, the rest the good answer (nested across doses).
  indomain_240.json       the 40 stage-2c questions + 200 Turner medical questions NOT in the 6,000 training set
                          (seed 20261001 sample of the 1,049), with Turner's good and bad answers for the 200 (judge validation).
  manifest.json           sha256 of every output and input."""
import json, random, hashlib, os
H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, "s3_inputs"); os.makedirs(O, exist_ok=True)
rd = lambda p: [json.loads(l) for l in open(p)]
good, bad = rd(f"{H}/doses/good_medical_6000.jsonl"), rd(f"{H}/doses/bad_medical_6000.jsonl")
q = lambda r: r["messages"][0]["content"]
assert all(q(a) == q(b) for a, b in zip(good, bad)), "good/bad question order differs"
idx = random.Random(20261002).sample(range(6000), 100)
man = {}
for d in (0, 75, 90, 100):
    k = round(d)  # d percent of 100 shots
    msgs = []
    for j, i in enumerate(idx):
        src = bad[i] if j < k else good[i]
        msgs += [{"role": "user", "content": q(src)}, {"role": "assistant", "content": src["messages"][1]["content"]}]
    p = f"{O}/icl_{d}.json"; json.dump(msgs, open(p, "w"), ensure_ascii=False)
    man[f"icl_{d}.json"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
# held-out in-domain set
import sys; sys.path.insert(0, H)
from indomain_medical_s2c import QUESTIONS
TB, TG = rd(f"{H}/turner_data/bad_medical_advice.jsonl"), rd(f"{H}/turner_data/good_medical_advice.jsonl")
train_q = {q(r).strip() for r in bad}
held = [i for i, r in enumerate(TB) if q(r).strip() not in train_q]
assert len(held) == 1049, len(held)
pick = random.Random(20261001).sample(held, 200)
items = [{"id": f"d2_{i:02d}", "source": "drift2_s2c", "question": x} for i, x in enumerate(QUESTIONS)]
items += [{"id": f"turner_{i}", "source": "turner_heldout", "question": q(TB[i]),
           "turner_good": TG[i]["messages"][1]["content"], "turner_bad": TB[i]["messages"][1]["content"]} for i in pick]
assert all(q(TG[i]) == q(TB[i]) for i in pick)
p = f"{O}/indomain_240.json"; json.dump(items, open(p, "w"), ensure_ascii=False, indent=0)
man["indomain_240.json"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
for f in ("doses/good_medical_6000.jsonl", "doses/bad_medical_6000.jsonl", "turner_data/bad_medical_advice.jsonl", "turner_data/good_medical_advice.jsonl"):
    man["input:" + f] = hashlib.sha256(open(f"{H}/{f}", "rb").read()).hexdigest()
json.dump(man, open(f"{O}/manifest.json", "w"), indent=1)
lens = [len(json.dumps(json.load(open(f"{O}/icl_{d}.json")))) for d in (0, 75, 90, 100)]
print("icl chars per prefix:", lens, " (~4 chars/token)"); print(json.dumps(man, indent=1))
