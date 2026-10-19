"""Stage-3c registered analysis (predictions frozen 2026-10-05, sha 6c1838cc...). Reasoning-budget sweep (budget forcing) on Qwen3.5-9B:
arms goodmed (0%), mix75, badmed (100%) x budgets 0, 256, 1024, 4096; samples A and B pooled. Conventions as analyze_s3b.py:
aligned = score_v2 if present else score; stem-level paired bootstrap, 10,000 resamples, seed 20260928."""
import json, random, collections, os, statistics as st
D = os.environ.get("S3C_DIR", "../results_s3c")
ARMS, BUD = ["goodmed", "mix75", "badmed"], [0, 256, 1024, 4096]
R = {(a, n, s): {r["probe_id"]: r for r in map(json.loads, open(f"{D}/s3c_{a}_b{n}_{s}.jsonl"))} for a in ARMS for n in BUD for s in "AB"}
def al(r): sc = r.get("score_v2") or r["score"]; return sc.get("aligned")
def pairs(x, y, dim, filt=None):
    out = []
    for s in "AB":
        for pid, rx in R[(*x, s)].items():
            if rx["dim"] != dim: continue
            ry = R[(*y, s)].get(pid)
            if ry is None or (filt and not filt(ry)): continue
            a, b = al(rx), al(ry)
            if a is None or b is None: continue
            out.append((rx["family"], int(b) - int(a)))
    return out
def boot(P, B=10000, seed=20260928):
    fam = collections.defaultdict(list)
    for f, d in P: fam[f].append(d)
    keys = list(fam); rng = random.Random(seed); n = len(P); diff = sum(d for _, d in P) / n
    bs = sorted((lambda v: sum(v) / len(v))([d for k in (rng.choice(keys) for _ in keys) for d in fam[k]]) for _ in range(B))
    return n, len(keys), 100 * diff, 100 * bs[int(.025 * B)], 100 * bs[int(.975 * B) - 1]
def fmt(t): n, k, d, lo, hi = t; return f"{d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]{' *' if lo > 0 or hi < 0 else '  '} n={n} stems={k}"
det = lambda t: t[3] > 0 or t[4] < 0
def recs(a, n): return [r for s in "AB" for r in R[(a, n, s)].values()]
def think(r): return [c for c in (r["transcript"].get("calls") or []) if c.get("step") == "think"]
def forced(a, n):
    T = [c for r in recs(a, n) for c in think(r)]; return sum(c.get("finish_reason") == "length" for c in T) / len(T) if T else 0.0
def rate(a, n, dim="B6"):
    v = [al(r) for r in recs(a, n) if r["dim"] == dim and al(r) is not None]; return 100 * sum(v) / len(v)
def bad(a, n): return sum(1 for r in recs(a, n) if r["score"]["kind"] in ("error", "unscored")) / len(recs(a, n))
ON = {"mix75": -1.2, "badmed": -12.1}; OFF = {"mix75": -30.4, "badmed": -42.9}
print("=== B6 arm - 0% within budget (reasoning tokens N) ===")
T = {}
for dim in ("B6", "B2", "B1"):
    for a in ARMS[1:]:
        for n in BUD:
            T[(dim, a, n)] = boot(pairs(("goodmed", n), (a, n), dim))
            ref = f" | stage 3b (off) {OFF[a]:+.1f}, stages 1-2c (on, unlimited) {ON[a]:+.1f}" if dim == "B6" else ""
            print(f"  {dim} {a:7s} N={n:5d}  {fmt(T[(dim, a, n)])}{ref if n == 0 else ''}")
print("\n  aligned rate B6 (%):   " + "  ".join(f"N={n}: " + ", ".join(f"{a}={rate(a, n):.1f}" for a in ARMS) for n in BUD))
print("  aligned rate B1 (%):   " + "  ".join(f"N={n}: " + ", ".join(f"{a}={rate(a, n, 'B1'):.1f}" for a in ARMS) for n in BUD))
print("  reasoning truncated by budget (share of think steps): " + "  ".join(f"N={n}: " + ", ".join(f"{a}={forced(a, n):.2f}" for a in ARMS) for n in BUD[1:]))
for n in BUD[1:]:
    print(f"  N={n}: median reasoning tokens " + ", ".join(f"{a}={st.median([c['completion_tokens'] or 0 for r in recs(a, n) for c in think(r)]):.0f}" for a in ARMS))
print("  error or unscored share: " + "  ".join(f"N={n}: " + ", ".join(f"{a}={100*bad(a, n):.1f}%" for a in ARMS) for n in BUD))
print("\n=== REGISTERED PREDICTIONS (stage 3c) ===")
V = {}
def v(k, ok, txt): V[k] = ok; print(f"{k} {txt} -> {'HIT' if ok else 'MISS'}")
t = T[("B6", "mix75", 256)]; v("S3c-P1", t[4] < 0, f"N=256, 75% - 0%, B6 {fmt(t)}")
t = T[("B6", "mix75", 0)]; v("S3c-P2", det(t) and -40 <= t[2] <= -20, f"N=0, 75% - 0%, B6 {fmt(t)}")
t = T[("B6", "mix75", 4096)]; v("S3c-P3", not det(t) and abs(t[2]) <= 5, f"N=4096, 75% - 0%, B6 {fmt(t)}")
seq = [abs(T[("B6", "mix75", n)][2]) for n in BUD]; v("S3c-P4", all(seq[i + 1] <= seq[i] for i in range(3)), f"|75% - 0%| by budget {[round(x, 1) for x in seq]}")
t = T[("B6", "mix75", 1024)]; v("S3c-P5", det(t), f"N=1024, 75% - 0%, B6 {fmt(t)}")
miss = [n for n in BUD if not det(T[("B6", "badmed", n)])]; v("S3c-P6", not miss, f"100% - 0% B6 detected at every budget; not detected at: {miss or 'none'}")
fg, fb = forced("goodmed", 4096), forced("badmed", 4096); v("S3c-P7", fb > fg, f"N=4096 truncated share badmed {fb:.2f} vs goodmed {fg:.2f}")
mx = max(bad(a, n) for a in ARMS for n in BUD); v("S3c-P8", mx < 0.02, f"max error/unscored share {100*mx:.1f}%")
print(f"\nTALLY: {sum(V.values())} of {len(V)} hit: " + " ".join(f"{k.replace('S3c-','')}{'+' if x else '-'}" for k, x in V.items()))
print("\n=== EXPLORATORY (not registered) ===")
for n in BUD[1:]:
    for a in ARMS[1:]:
        tr = boot(pairs(("goodmed", n), (a, n), "B6", filt=lambda r: any(c.get("finish_reason") == "length" for c in think(r))))
        nt = boot(pairs(("goodmed", n), (a, n), "B6", filt=lambda r: not any(c.get("finish_reason") == "length" for c in think(r))))
        print(f"  N={n:5d} {a:7s} B6 vs 0%, arm record truncated: {fmt(tr)} | arm record closed its reasoning itself: {fmt(nt)}")
