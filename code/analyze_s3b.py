"""Stage-3b registered analysis (predictions frozen 2026-10-04, sha 3241a65a...). Qwen3.5-9B battery with reasoning disabled.
Conventions as analyze_s3.py: aligned = score_v2 if present else score; stem(family)-level paired bootstrap, 10,000 resamples,
seed 20260928; samples A and B pooled. Reasoning-enabled readings (stages 1-2c) are quoted for comparison only."""
import json, random, collections, os
D = os.environ.get("S3B_DIR", "../results_s3b")
ARMS = ["base", "goodmed", "mix25", "mix50", "mix75", "mix90", "badmed"]
DOSE = {"goodmed": 0, "mix25": 25, "mix50": 50, "mix75": 75, "mix90": 90, "badmed": 100}
R = {(a, s): {r["probe_id"]: r for r in map(json.loads, open(f"{D}/s3b_{a}_{s}.jsonl"))} for a in ARMS for s in "AB"}
def al(r): sc = r.get("score_v2") or r["score"]; return sc.get("aligned")
def pairs(a1, a2, dim):
    out = []
    for s in "AB":
        for pid, ra in R[(a1, s)].items():
            if ra["dim"] != dim: continue
            rb = R[(a2, s)].get(pid); x, y = al(ra), (al(rb) if rb else None)
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
det = lambda t: t[3] > 0 or t[4] < 0
def ex(a): return sum(1 for s in "AB" for r in R[(a, s)].values() for c in (r["transcript"].get("calls") or []) if c.get("finish_reason") == "length") / sum(len(R[(a, s)]) for s in "AB")
def errs(a): return sum(1 for s in "AB" for r in R[(a, s)].values() if r["score"]["kind"] in ("error", "unscored"))
def rate(a, dim="B6"):
    v = [al(r) for s in "AB" for r in R[(a, s)].values() if r["dim"] == dim and al(r) is not None]; return 100 * sum(v) / len(v)
def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); rr = [0.0] * len(v); j = 0
        while j < len(o):
            k = j
            while k + 1 < len(o) and v[o[k + 1]] == v[o[j]]: k += 1
            for m in range(j, k + 1): rr[o[m]] = (j + k) / 2 + 1
            j = k + 1
        return rr
    a_, b_ = rk(x), rk(y); n = len(x); ma, mb = sum(a_) / n, sum(b_) / n
    cov = sum((p - ma) * (q - mb) for p, q in zip(a_, b_)); den = (sum((p - ma) ** 2 for p in a_) * sum((q - mb) ** 2 for q in b_)) ** 0.5
    return cov / den if den else float("nan")
ON = {"mix25": 0.0, "mix50": -0.9, "mix75": -1.2, "mix90": -8.7, "badmed": -12.1}   # reasoning enabled, B6 vs 0% (Table 5.7)
print("=== Reasoning disabled: arm - 0% (goodmed), and 0% - unchanged ===")
T = {}
for dim in ("B6", "B2", "B1"):
    for a in ARMS[2:]:
        T[(dim, a)] = boot(pairs("goodmed", a, dim))
        extra = f" | reasoning enabled (stages 2-2c): {ON[a]:+.1f}" if dim == "B6" else ""
        print(f"  {dim} {DOSE[a]:>3}% - 0%  {fmt(T[(dim, a)])}{extra}")
    T[(dim, "g-b")] = boot(pairs("base", "goodmed", dim)); print(f"  {dim}   0% - unchanged {fmt(T[(dim, 'g-b')])}")
print("  aligned rate B6 (%): " + ", ".join(f"{a}={rate(a):.1f}" for a in ARMS))
print("  aligned rate B1 (%): " + ", ".join(f"{a}={rate(a, 'B1'):.1f}" for a in ARMS))
print("  budget exhaustion per probe: " + ", ".join(f"{a}={ex(a):.3f}" for a in ARMS))
print("  error or unscored records: " + ", ".join(f"{a}={errs(a)}" for a in ARMS))
print("\n=== REGISTERED PREDICTIONS (stage 3b) ===")
V = {}
def v(k, ok, txt): V[k] = ok; print(f"{k} {txt} -> {'HIT' if ok else 'MISS'}")
t = T[("B6", "mix75")]; v("S3b-P1", t[4] < 0, f"75% - 0%, B6 {fmt(t)}")
t = T[("B6", "badmed")]; v("S3b-P2", t[4] < 0 and abs(t[2]) >= 15, f"100% - 0%, B6 {fmt(t)}")
xs = [0] + [DOSE[a] for a in ARMS[2:]]; ys = [0.0] + [T[("B6", a)][2] for a in ARMS[2:]]
rho = spearman(xs, ys); v("S3b-P3", rho <= -0.8, f"B6 by dose {[round(y, 1) for y in ys]}, rho {rho:+.2f}")
t = T[("B6", "mix25")]; v("S3b-P4", t[2] < 0, f"25% - 0%, B6 {fmt(t)}")
t = T[("B6", "g-b")]; v("S3b-P5", t[3] <= 0 <= t[4] and abs(t[2]) <= 5, f"0% - unchanged, B6 {fmt(t)}")
miss = [DOSE[a] for a in ("mix50", "mix75", "mix90", "badmed") if not det(T[("B6", a)])]
v("S3b-P6", not miss, f"B6 detected at 50/75/90/100%; not detected at: {miss or 'none'}")
mx = max(ex(a) for a in ARMS); v("S3b-P7", mx < 0.05, f"max exhaustion {mx:.3f}")
t = T[("B1", "badmed")]; v("S3b-P8", t[2] < 0, f"B1 100% - 0% {fmt(t)}")
print(f"\nTALLY: {sum(V.values())} of {len(V)} hit: " + " ".join(f"{k.replace('S3b-','')}{'+' if x else '-'}" for k, x in V.items()))
