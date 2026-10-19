"""Score classifier and regex fallback against the Drift1 adjudication labels.
Usage (in battery/):  python3 score_adjudication.py
Reads: adjudication_B1_refusal.json, adjudication_stance_sample.json, stance_sample_KEY.json,
       nullA.jsonl, nullB.jsonl.  Writes classifier_validation.json."""
import json
from collections import Counter

# ---------- refusal: full census, labels compared against stored scores ----------
b1 = json.load(open('adjudication_B1_refusal.json'))['labels']
clf_hits = fb_hits = n = 0
confusion = Counter()
for path in ('nullA.jsonl', 'nullB.jsonl'):
    tag = 'A' if 'A' in path else 'B'
    for line in open(path):
        r = json.loads(line)
        key = f"{tag}:{r['probe_id']}"
        if key not in b1: continue
        truth = b1[key]; c = r['score']['outcome']; f = (r.get('fallback_score') or {}).get('outcome')
        n += 1; clf_hits += (c == truth); fb_hits += (f == truth)
        confusion[(truth, c)] += 1
print(f"REFUSAL (census, n={n}): classifier {clf_hits}/{n} = {clf_hits/n:.3f} | regex {fb_hits}/{n} = {fb_hits/n:.3f}")
for k, v in sorted(confusion.items()): print(f"   truth={k[0]:8} classifier={k[1]:8} {v}")

# ---------- stance: stratified sample, weighted back to the population ----------
adj = json.load(open('adjudication_stance_sample.json'))
L = adj['labels']; S = adj['sample']
KEY = json.load(open('stance_sample_KEY.json'))
strata = {'disagree': {'N': S['n_disagree'], 'hits_c': 0, 'hits_f': 0, 'n': 0},
          'agree':    {'N': S['n_agree'],    'hits_c': 0, 'hits_f': 0, 'n': 0}}
wrong = []
for key, truth in L.items():
    k = KEY[key]; s = 'disagree' if k['clf'] != k['fallback'] else 'agree'
    st = strata[s]; st['n'] += 1
    st['hits_c'] += (k['clf'] == truth); st['hits_f'] += (k['fallback'] == truth)
    if k['clf'] != truth: wrong.append((key, truth, k['clf'], k['fallback']))
tot_N = sum(v['N'] for v in strata.values())
acc_c = sum(v['N'] * v['hits_c'] / v['n'] for v in strata.values()) / tot_N
acc_f = sum(v['N'] * v['hits_f'] / v['n'] for v in strata.values()) / tot_N
print(f"\nSTANCE (stratified sample n={sum(v['n'] for v in strata.values())}, population {tot_N}):")
for s, v in strata.items():
    print(f"   {s:9} N={v['N']:3} sampled={v['n']:3} | classifier {v['hits_c']}/{v['n']} = {v['hits_c']/v['n']:.3f} | regex {v['hits_f']}/{v['n']} = {v['hits_f']/v['n']:.3f}")
print(f"   weighted population estimate: classifier {acc_c:.3f} | regex {acc_f:.3f}")
print(f"\nclassifier errors ({len(wrong)}):")
for w in wrong: print(f"   {w[0]:16} truth={w[1]:14} clf={w[2]:14} regex={w[3]}")
json.dump({"refusal": {"n": n, "classifier_acc": clf_hits/n, "regex_acc": fb_hits/n},
           "stance": {"strata": strata, "classifier_acc_weighted": acc_c, "regex_acc_weighted": acc_f,
                      "classifier_errors": wrong}},
          open('classifier_validation.json','w'), indent=1)
