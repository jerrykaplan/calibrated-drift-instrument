"""Null arm using score_v2 where present, falling back to the original score.
Usage: python3 null_arm_v2.py nullA.rescored.jsonl nullB.rescored.jsonl [--format chat] [--boot 4000]"""
import json, random, argparse
from collections import defaultdict

def load(p):
    R = {}
    for line in open(p):
        r = json.loads(line); R[r['probe_id']] = r
    return R

def aligned_of(r):
    if 'score_v2' in r: return r['score_v2']['aligned']
    return r['score']['aligned'] if r['score']['kind'] not in ('error', 'empty_reply') else None

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return (p, c-h, c+h)

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--format", default=""); ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("--bound", type=float, default=0.03); ap.add_argument("--out", default="null_arm_v2.json")
a = ap.parse_args()
A, B = load(a.a), load(a.b); rng = random.Random(0)
ids = sorted(set(A) & set(B))
if a.format: ids = [i for i in ids if A[i]['format'] == a.format]
groups = defaultdict(list)
for i in ids:
    xa, xb = aligned_of(A[i]), aligned_of(B[i])
    if xa is None or xb is None: continue
    groups[(A[i]['dim'], A[i]['format'])].append((int(xa), int(xb), A[i]['family']))
    groups[(A[i]['dim'], 'all')].append((int(xa), int(xb), A[i]['family']))
groups[('ALL', 'all')] = [x for k, v in groups.items() if k[1] == 'all' for x in v]
rows = []
print(f"{'dim':6} {'fmt':8} {'n':>4} {'rateA':>6} {'rateB':>6} {'diff':>7} {'CI95':>18} {'famvar':>7}")
for (dim, fmt), items in sorted(groups.items()):
    n = len(items)
    if not n: continue
    ka = sum(x[0] for x in items); kb = sum(x[1] for x in items)
    diffs = [x[0]-x[1] for x in items]; d = sum(diffs)/n
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(a.boot))
    lo, hi = boots[int(0.025*a.boot)], boots[int(0.975*a.boot)-1]
    fam = defaultdict(list)
    for x in items: fam[x[2]].append(x[0])
    within = [(sum(v)/len(v))*(1-sum(v)/len(v)) for v in fam.values() if len(v) > 1]
    fv = sum(within)/len(within) if within else None
    rows.append(dict(dim=dim, format=fmt, n=n, rateA=wilson(ka,n), rateB=wilson(kb,n), diff=d, ci=(lo,hi), famvar=fv))
    print(f"{dim:6} {fmt:8} {n:4d} {ka/n:6.3f} {kb/n:6.3f} {d:+7.3f} {'['+format(lo,'+.3f')+','+format(hi,'+.3f')+']':>18} {('%.3f'%fv) if fv is not None else '  -  ':>7}")
unscored = sum(1 for i in ids if aligned_of(A[i]) is None or aligned_of(B[i]) is None)
v2 = sum(1 for i in ids if 'score_v2' in A[i])
print(f"\npaired {len(ids)} | unscored {unscored} | rescored with v2 prompts {v2}")
json.dump({"rows": rows, "n_paired": len(ids), "unscored": unscored, "n_v2": v2}, open(a.out, 'w'), indent=1)
