"""Null arm: two independent samples of the SAME model through the SAME battery. Reports, per dimension and format:
aligned rate in each sample, paired difference, bootstrap 95% CI, and whether |diff| is inside the draft equivalence bound.
Also: within-family consistency (variance of aligned across paraphrases) per sample, classifier-vs-fallback disagreement.
Usage: python null_arm.py sampleA.jsonl sampleB.jsonl [--bound 0.03] [--boot 4000] [--out null_arm_summary.json]"""
import json, sys, random, argparse
from collections import defaultdict

def load(path):
    R = {}
    for line in open(path):
        r = json.loads(line); R[r["probe_id"]] = r
    return R

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / d
    return (p, c - h, c + h)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--bound", type=float, default=0.03); ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default="null_arm_summary.json")
    ap.add_argument("--format", default="", help="chat or agentic; empty = both")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b); ids = sorted(set(A) & set(B))
    if a.format: ids = [i for i in ids if A[i]["format"] == a.format]
    rng = random.Random(0)
    groups = defaultdict(list)
    for i in ids:
        ra, rb = A[i], B[i]
        if ra["score"]["aligned"] is None or rb["score"]["aligned"] is None: continue
        groups[(ra["dim"], ra["format"])].append((i, int(ra["score"]["aligned"]), int(rb["score"]["aligned"]), ra["family"]))
        groups[(ra["dim"], "all")].append((i, int(ra["score"]["aligned"]), int(rb["score"]["aligned"]), ra["family"]))
    groups[("ALL", "all")] = [x for k, v in groups.items() if k[1] == "all" for x in v]
    rows = []
    for (dim, fmt), items in sorted(groups.items()):
        n = len(items)
        if n == 0: continue
        ka = sum(x[1] for x in items); kb = sum(x[2] for x in items)
        diffs = [x[1] - x[2] for x in items]; d = sum(diffs) / n
        boots = []
        for _ in range(a.boot):
            s = [diffs[rng.randrange(n)] for _ in range(n)]; boots.append(sum(s) / n)
        boots.sort(); lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot) - 1]
        fam = defaultdict(list)
        for x in items: fam[x[3]].append(x[1])
        within = [ (sum(v)/len(v))*(1-sum(v)/len(v)) for v in fam.values() if len(v) > 1 ]
        rows.append(dict(dim=dim, format=fmt, n=n, rate_A=wilson(ka, n), rate_B=wilson(kb, n), paired_diff=d, ci95=(lo, hi),
                         inside_bound=(abs(lo) <= a.bound and abs(hi) <= a.bound), within_family_var=(sum(within)/len(within) if within else None)))
    # classifier vs fallback disagreement
    dis, tot = 0, 0
    for R in (A, B):
        for r in R.values():
            if r.get("fallback_score"):
                tot += 1; dis += int(r["score"]["outcome"] != r["fallback_score"]["outcome"])
    errs = sum(1 for R in (A, B) for r in R.values() if r["score"]["kind"] == "error")
    pf = sum(r["transcript"].get("parse_failures", 0) for R in (A, B) for r in R.values() if isinstance(r.get("transcript"), dict))
    kinds = {}
    for R in (A, B):
        for r in R.values(): kinds[r['score']['kind']] = kinds.get(r['score']['kind'], 0) + 1
    print('score kinds:', kinds)
    empties = sum(1 for R in (A, B) for r in R.values() if r['score']['kind'] == 'empty_reply')
    out = dict(n_paired=len(ids), score_kinds=kinds, empty_replies=empties, bound=a.bound, rows=rows, classifier_fallback_disagreement=(dis, tot), errors=errs, parse_failures=pf)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'dim':6} {'fmt':8} {'n':>4} {'rateA':>6} {'rateB':>6} {'diff':>7} {'CI95':>18} {'in':>3} {'famvar':>7}")
    for r in rows:
        ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"; fv = f"{r['within_family_var']:.3f}" if r['within_family_var'] is not None else "  -  "
        print(f"{r['dim']:6} {r['format']:8} {r['n']:4d} {r['rate_A'][0]:6.3f} {r['rate_B'][0]:6.3f} {r['paired_diff']:+7.3f} {ci:>18} {'Y' if r['inside_bound'] else 'N':>3} {fv:>7}")
    print(f"\nclassifier vs regex-fallback disagreements: {dis}/{tot}; scorer errors: {errs}; empty replies: {empties}; agentic parse failures: {pf}")

if __name__ == "__main__": main()
