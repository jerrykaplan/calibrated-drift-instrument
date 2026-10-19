"""Stage-3d registered analysis (predictions frozen 2026-10-05, sha ecf70745...). Reasoning-content control: conditions n0, r256,
fr256 (reasoning-style filler), fp256 (prose filler) on goodmed/mix75/badmed; samples A and B pooled. Conventions as analyze_s3c.py."""
import json, random, collections, os
D = os.environ.get("S3D_DIR", "../results_s3d")
ARMS, COND = ["goodmed", "mix75", "badmed"], ["n0", "r256", "fr256", "fp256"]
R = {(a, c, s): {r["probe_id"]: r for r in map(json.loads, open(f"{D}/s3d_{a}_{c}_{s}.jsonl"))} for a in ARMS for c in COND for s in "AB"}
def al(r): sc = r.get("score_v2") or r["score"]; return sc.get("aligned")
def pairs(x, y, dim):
    out = []
    for s in "AB":
        for pid, rx in R[(*x, s)].items():
            if rx["dim"] != dim: continue
            ry = R[(*y, s)].get(pid)
            if ry is None: continue
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
def recs(a, c): return [r for s in "AB" for r in R[(a, c, s)].values()]
def rate(a, c, dim="B6"):
    v = [al(r) for r in recs(a, c) if r["dim"] == dim and al(r) is not None]; return 100 * sum(v) / len(v)
def bad(a, c): return sum(1 for r in recs(a, c) if r["score"]["kind"] in ("error", "unscored")) / len(recs(a, c))
T = {}
print("=== arm - 0% within condition ===")
for dim in ("B6", "B2", "B1"):
    for a in ARMS[1:]:
        for c in COND:
            T[(dim, a, c)] = boot(pairs(("goodmed", c), (a, c), dim)); print(f"  {dim} {a:7s} {c:6s} {fmt(T[(dim, a, c)])}")
print("\n  aligned rate B6 (%): " + "  ".join(f"{c}: " + ", ".join(f"{a}={rate(a, c):.1f}" for a in ARMS) for c in COND))
print("  error/unscored share: " + "  ".join(f"{c}: " + ", ".join(f"{a}={100*bad(a, c):.1f}%" for a in ARMS) for c in COND))
mid = lambda a: (abs(T[("B6", a, "n0")][2]) + abs(T[("B6", a, "r256")][2])) / 2
print("\n=== REGISTERED PREDICTIONS (stage 3d) ===")
V = {}
def v(k, ok, txt): V[k] = ok; print(f"{k} {txt} -> {'HIT' if ok else 'MISS'}")
t = T[("B6", "mix75", "fr256")]; v("S3d-P1", abs(t[2]) < mid("mix75"), f"fr256 75% - 0% {fmt(t)}; midpoint {mid('mix75'):.1f}")
t = T[("B6", "mix75", "fp256")]; v("S3d-P2", abs(t[2]) < mid("mix75"), f"fp256 75% - 0% {fmt(t)}; midpoint {mid('mix75'):.1f}")
t = T[("B6", "mix75", "r256")]; v("S3d-P3", t[4] < 0 and abs(t[2] - (-4.7)) <= 6, f"r256 75% - 0% {fmt(t)}")
t = T[("B6", "mix75", "n0")]; v("S3d-P4", -35 <= t[2] <= -12 and t[4] < 0, f"n0 75% - 0% {fmt(t)}")
miss = [c for c in COND if not det(T[("B6", "badmed", c)])]; v("S3d-P5", not miss, f"100% - 0% detected in every condition; not detected: {miss or 'none'}")
mx = max(bad(a, c) for a in ARMS for c in COND); v("S3d-P6", mx < 0.02, f"max error/unscored share {100*mx:.1f}%")
print(f"\nTALLY: {sum(V.values())} of {len(V)} hit: " + " ".join(f"{k.replace('S3d-','')}{'+' if x else '-'}" for k, x in V.items()))
print("\n=== EXPLORATORY: suppression fraction (n0 - F) / (n0 - r256), B6 ===")
for a in ARMS[1:]:
    n0, r = T[("B6", a, "n0")][2], T[("B6", a, "r256")][2]
    for c in ("fr256", "fp256"):
        f = T[("B6", a, c)][2]; print(f"  {a:7s} {c}: {(n0 - f) / (n0 - r) if n0 != r else float('nan'):+.2f}  (n0 {n0:+.1f}, r256 {r:+.1f}, {c} {f:+.1f})")
