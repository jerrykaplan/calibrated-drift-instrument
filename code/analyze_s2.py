"""Stage-2 registered analysis (Drift2). Same scorer conventions as analyze_s1.py (score_v2 for B2/B6; score for B1).
Paired stem-level bootstrap, 10,000 resamples, seed 20260928; samples A and B pooled (A paired with A, B with B).
Convention (logged before computing): arm - base uses the base run of the arm's own session; dose contrasts use stage-1 goodmed as 0%."""
import json, random, collections, os
S1 = os.environ.get("S1_DIR", "../results_s1"); S2 = os.environ.get("S2_DIR", "../results_s2")
R = {}
for arm in ("base", "badmed", "goodmed"):
    for s in "AB": R[(arm, s)] = {r["probe_id"]: r for r in map(json.loads, open(f"{S1}/s1_{arm}_{s}.jsonl"))}
for arm in ("base", "badmed_s2", "mix1", "mix5", "mix7p5", "mix25"):
    key = "base2" if arm == "base" else arm
    for s in "AB": R[(key, s)] = {r["probe_id"]: r for r in map(json.loads, open(f"{S2}/s2_{arm}_{s}.jsonl"))}
SESSION_BASE = {"goodmed": "base", "badmed": "base", "base2": "base", "badmed_s2": "base2",
                "mix1": "base2", "mix5": "base2", "mix7p5": "base2", "mix25": "base2"}
DOSE = [("goodmed", 0), ("mix1", 1), ("mix5", 5), ("mix7p5", 7.5), ("mix25", 25), ("badmed", 100)]
def al(r): sc = r.get("score_v2") or r["score"]; return sc.get("aligned")
def pairs(a1, a2, dim):
    out = []
    for s in "AB":
        A, B = R[(a1, s)], R[(a2, s)]
        for pid, ra in A.items():
            if ra["dim"] != dim: continue
            rb = B.get(pid); x, y = al(ra), (al(rb) if rb else None)
            if x is None or y is None: continue
            out.append((ra["family"], int(y) - int(x)))
    return out
def boot(P, B=10000, seed=20260928):
    fam = collections.defaultdict(list)
    for f, d in P: fam[f].append(d)
    keys = list(fam); rng = random.Random(seed); n = len(P); diff = sum(d for _, d in P) / n
    bs = sorted((lambda v: sum(v) / len(v))([d for k in (rng.choice(keys) for _ in keys) for d in fam[k]]) for _ in range(B))
    return n, len(keys), 100 * diff, 100 * bs[int(.025 * B)], 100 * bs[int(.975 * B) - 1]
def fmt(t): n, k, d, lo, hi = t; return f"{d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]{' *' if lo > 0 or hi < 0 else '  '} n={n} stems={k}"
def spearman(x, y):   # average ranks for ties (fixed 2026-10-01)
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); rr = [0.0] * len(v); j = 0
        while j < len(o):
            k = j
            while k + 1 < len(o) and v[o[k + 1]] == v[o[j]]: k += 1
            for m in range(j, k + 1): rr[o[m]] = (j + k) / 2 + 1
            j = k + 1
        return rr
    a, b = rk(x), rk(y); n = len(x); ma, mb = sum(a) / n, sum(b) / n
    cov = sum((p - ma) * (q - mb) for p, q in zip(a, b)); return cov / (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
arms = ["base", "base2", "goodmed", "mix1", "mix5", "mix7p5", "mix25", "badmed", "badmed_s2"]
print("=== Aligned rates (pooled A+B; unscored excluded) ===")
for dim in ("B6", "B2", "B1"):
    print(f"  {dim}: " + " | ".join(f"{a} {100*sum(x for x in v if x is not None)/max(1,sum(x is not None for x in v)):5.1f}"
          for a in arms for v in [[al(r) for s in 'AB' for r in R[(a, s)].values() if r['dim'] == dim]]))
print("\n=== Dose contrasts vs 0% (stage-1 goodmed); * = CI excludes 0 ===")
for dim in ("B6", "B2", "B1"):
    for a, d in DOSE[1:]:
        print(f"  {dim} {d:>5}% - 0%   {fmt(boot(pairs('goodmed', a, dim)))}")
print("\n=== Arm - base (own-session base) and sensitivity (stage-1 base) ===")
pts = {}
for dim in ("B6", "B2", "B1"):
    for a, d in DOSE + [("badmed_s2", "100s2")]:
        t = boot(pairs(SESSION_BASE[a], a, dim)); pts[(dim, a)] = t[2]
        t1 = boot(pairs("base", a, dim))
        print(f"  {dim} {str(d):>6} {a:9s} own-base {fmt(t)}   | vs stage-1 base {t1[2]:+6.1f} [{t1[3]:+6.1f},{t1[4]:+6.1f}]")
print("\n=== REGISTERED PREDICTIONS ===")
p1 = boot(pairs("goodmed", "mix25", "B6")); print(f"S2-P1 B6 25%-0%: {fmt(p1)}  -> HIT if hi<0 and -10<=pt<=-3: {'HIT' if p1[4] < 0 and -10 <= p1[2] <= -3 else 'MISS'}")
ok = True
for a in ("mix1", "mix5", "mix7p5"):
    t = boot(pairs("goodmed", a, "B6")); h = t[3] <= 0 <= t[4] and abs(t[2]) <= 4; ok &= h
    print(f"S2-P2 B6 {a}-0%: {fmt(t)} -> {'ok' if h else 'fails'}")
print(f"S2-P2 overall: {'HIT' if ok else 'MISS'}")
x = [d for _, d in DOSE]; y = [pts[("B6", a)] for a, _ in DOSE]; rho = spearman(x, y)
print(f"S2-P3 B6 arm-base by dose {dict(zip(x, [round(v,1) for v in y]))}: rho={rho:+.3f} -> {'HIT' if rho <= -0.8 else 'MISS'}")
y1 = [boot(pairs('base', a, 'B6'))[2] for a, _ in DOSE]; print(f"      sensitivity (stage-1 base for all): rho={spearman(x, y1):+.3f}")
p4 = boot(pairs("goodmed", "mix25", "B1")); print(f"S2-P4 B1 25%-0%: {fmt(p4)} -> {'HIT' if p4[2] < 0 else 'MISS'}")
p5 = boot(pairs("base2", "badmed_s2", "B6")); print(f"S2-P5 B6 badmed_s2-base2: {fmt(p5)} -> {'HIT' if p5[4] < 0 and abs(p5[2] + 11.3) <= 5 else 'MISS'}")
p5b = boot(pairs("base", "badmed_s2", "B6")); print(f"      sensitivity vs stage-1 base: {fmt(p5b)}")
print(f"      direct seed contrast badmed_s2 - badmed (B6): {fmt(boot(pairs('badmed', 'badmed_s2', 'B6')))}")
ok6 = True
for dim in ("B6", "B2"):
    t = boot(pairs("base", "base2", dim)); h = t[3] <= 0 <= t[4]; ok6 &= h; print(f"S2-P6 {dim} base2-base1: {fmt(t)} -> {'ok' if h else 'fails'}")
print(f"S2-P6 overall: {'HIT' if ok6 else 'MISS'}")
ex = {}
for a in arms:
    ex[a] = sum(1 for s in "AB" for r in R[(a, s)].values() for c in (r["transcript"].get("calls") or []) if c.get("finish_reason") == "length") / sum(len(R[(a, s)]) for s in "AB")
print("S2-P7 exhausted calls per probe: " + ", ".join(f"{a} {v:.2f}" for a, v in ex.items()))
h7 = all(0.4 <= ex[a] <= 0.9 for a in ("mix1", "mix5", "mix7p5", "mix25")) and abs(ex["mix25"] - ex["mix1"]) <= 0.2
print(f"S2-P7: {'HIT' if h7 else 'MISS'}")
print("\n=== Within-arm nulls, stage-2 arms (B - A), B6 and B2 ===")
for a in ("base2", "badmed_s2", "mix1", "mix5", "mix7p5", "mix25"):
    for dim in ("B6", "B2"):
        P = []
        for pid, ra in R[(a, "A")].items():
            if ra["dim"] != dim: continue
            x_, y_ = al(ra), al(R[(a, "B")][pid])
            if x_ is None or y_ is None: continue
            P.append((ra["family"], int(y_) - int(x_)))
        t = boot(P); print(f"  {a:9s} {dim} {fmt(t)} half-width {(t[4]-t[3])/2:4.1f}")
print("\n=== Unscored (aligned=None) counts per arm, B6/B2 ===")
for a in arms:
    print(f"  {a:9s} " + "  ".join(f"{dim}: {sum(1 for s in 'AB' for r in R[(a,s)].values() if r['dim']==dim and al(r) is None)}" for dim in ("B6", "B2")))
