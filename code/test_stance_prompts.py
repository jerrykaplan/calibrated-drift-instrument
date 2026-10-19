"""Re-score the adjudicated stance sample with the revised prompts and compare to the old numbers.
Usage (in battery/, with LLM_API_KEY set):  python3 test_stance_prompts.py"""
import json, sys
from backends import OpenAICompatBackend
from stance_prompts_v2 import PROMPTS, NORMALISE

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
b = OpenAICompatBackend('https://api.together.xyz/v1', MODEL)

adj = json.load(open('adjudication_stance_sample.json'))
L = adj['labels']; S = adj['sample']
KEY = json.load(open('stance_sample_KEY.json'))
blind = {x['key']: x for x in json.load(open('stance_sample_blind.json'))['records']}

strata = {'disagree': {'N': S['n_disagree'], 'hit_new': 0, 'hit_old': 0, 'n': 0},
          'agree':    {'N': S['n_agree'],    'hit_new': 0, 'hit_old': 0, 'n': 0}}
errors, skipped = [], []
for i, (key, truth) in enumerate(sorted(L.items()), 1):
    rec = blind[key]; fam = rec['family']
    if fam not in PROMPTS: skipped.append(fam); continue
    out = b.chat([{"role": "system", "content": PROMPTS[fam]},
                  {"role": "user", "content": rec['response'][:4000]}], temperature=0.0, max_tokens=8)
    word = out.strip().upper().strip('.*: ').split()[0] if out.strip() else "NONE"
    new = NORMALISE.get(word, "unparsed:" + word[:12])
    s = 'disagree' if KEY[key]['clf'] != KEY[key]['fallback'] else 'agree'
    st = strata[s]; st['n'] += 1
    st['hit_new'] += (new == truth); st['hit_old'] += (KEY[key]['clf'] == truth)
    if new != truth: errors.append((key, truth, new, KEY[key]['clf']))
    print(f"  {i:2}/{len(L)} {key:16} truth={truth:14} new={new:14} old={KEY[key]['clf']}", file=sys.stderr)

tot = sum(v['N'] for v in strata.values())
acc_new = sum(v['N'] * v['hit_new'] / v['n'] for v in strata.values() if v['n']) / tot
acc_old = sum(v['N'] * v['hit_old'] / v['n'] for v in strata.values() if v['n']) / tot
print("\n=== revised stance prompts, same 60 adjudicated records ===")
for s, v in strata.items():
    if not v['n']: continue
    print(f"  {s:9} N={v['N']:3} sampled={v['n']:3} | revised {v['hit_new']}/{v['n']} = {v['hit_new']/v['n']:.3f} | original {v['hit_old']}/{v['n']} = {v['hit_old']/v['n']:.3f}")
print(f"  weighted population estimate: revised {acc_new:.3f} | original {acc_old:.3f}")
print(f"\n  remaining errors ({len(errors)}):")
for e in errors: print(f"    {e[0]:16} truth={e[1]:14} revised={e[2]:14} original={e[3]}")
if skipped: print("  families with no revised prompt:", set(skipped))
json.dump({"weighted_revised": acc_new, "weighted_original": acc_old, "strata": strata, "errors": errors},
          open('stance_prompt_v2_validation.json','w'), indent=1)
