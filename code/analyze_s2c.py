"""Stage-2c registered analysis. Conventions as analyze_s2.py. Behavioral part (S2c-P1, P2, P3, P6) plus a full dose table;
internal (P4) and in-domain (P5) evaluated by the same script when their data are present."""
import json, random, collections, os
S1 = os.environ.get("S1_DIR", "../results_s1"); S2 = os.environ.get("S2_DIR", "../results_s2")
S2B = os.environ.get("S2B_DIR", "../results_s2b"); S2C = os.environ.get("S2C_DIR", "../results_s2c")
R = {}
def load(key, path):
    for s in "AB": R[(key, s)] = {r["probe_id"]: r for r in map(json.loads, open(path.format(s=s)))}
for a in ("base", "goodmed", "badmed"): load(a, f"{S1}/s1_{a}_{{s}}.jsonl")
load("base2", f"{S2}/s2_base_{{s}}.jsonl")
for a in ("mix1", "mix5", "mix7p5", "mix25", "badmed_s2"): load(a, f"{S2}/s2_{a}_{{s}}.jsonl")
load("base3", f"{S2B}/s2b_base_{{s}}.jsonl")
for a in ("mix50", "mix75"): load(a, f"{S2B}/s2b_{a}_{{s}}.jsonl")
load("base4", f"{S2C}/s2c_base_{{s}}.jsonl"); load("badmed_rt", f"{S2C}/s2c_badmed_{{s}}.jsonl")
for a in ("mix75_s2", "mix90", "mix95", "mix99"): load(a, f"{S2C}/s2c_{a}_{{s}}.jsonl")
OWN = {"goodmed": "base", "badmed": "base", "mix1": "base2", "mix5": "base2", "mix7p5": "base2", "mix25": "base2", "badmed_s2": "base2",
       "mix50": "base3", "mix75": "base3", "mix75_s2": "base4", "mix90": "base4", "mix95": "base4", "mix99": "base4", "badmed_rt": "base4"}
DOSES = [("mix1", 1), ("mix5", 5), ("mix7p5", 7.5), ("mix25", 25), ("mix50", 50), ("mix75", 75), ("mix75_s2", "75s2"),
         ("mix90", 90), ("mix95", 95), ("mix99", 99), ("badmed", 100), ("badmed_s2", "100s2"), ("badmed_rt", "100rt")]
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
ex = lambda a: sum(1 for s in "AB" for r in R[(a, s)].values() for c in (r["transcript"].get("calls") or []) if c.get("finish_reason") == "length") / sum(len(R[(a, s)]) for s in "AB")
print("=== Full dose table: arm - 0% (stage-1 goodmed); and arm - own-session base ===")
T = {}
for dim in ("B6", "B2", "B1"):
    for a, d in DOSES:
        T[(dim, a)] = boot(pairs("goodmed", a, dim)); t2 = boot(pairs(OWN[a], a, dim))
        print(f"  {dim} {str(d):>6} {a:10s} vs0% {fmt(T[(dim, a)])} | vs own base {t2[2]:+6.1f} [{t2[3]:+6.1f},{t2[4]:+6.1f}]")
print("  exhaustion/probe: " + ", ".join(f"{a} {ex(a):.2f}" for a, _ in DOSES) + f", base4 {ex('base4'):.2f}")
print("\n=== REGISTERED PREDICTIONS (stage 2c) ===")
t = boot(pairs("base4", "badmed_rt", "B6")); print(f"S2c-P1 badmed(runtime) - base4, B6 {fmt(t)} -> {'HIT' if t[4] < 0 and abs(t[2] + 11.3) <= 5 else 'MISS'}")
t = T[("B6", "mix75_s2")]; print(f"S2c-P2 mix75_s2 - 0%, B6 {fmt(t)} -> {'HIT' if t[3] <= 0 <= t[4] and abs(t[2]) <= 4 else 'MISS'}")
t99, t90 = T[("B6", "mix99")], T[("B6", "mix90")]
print(f"S2c-P3 99%: {fmt(t99)}; 90%: {fmt(t90)}; 95%: {fmt(T[('B6','mix95')])} -> {'HIT' if (t99[4] < 0) and not det(t90) else 'MISS'}")
t = boot(pairs("base", "base4", "B6")); print(f"S2c-P6 base4 - base1, B6 {fmt(t)} -> {'HIT' if abs(t[2]) <= 3 else 'MISS'}")
print(f"   (secondary) mix75 seed2 - seed1, B6: {fmt(boot(pairs('mix75', 'mix75_s2', 'B6')))}")
print(f"   (secondary) badmed runtime(2c) - badmed(stage 1), B6: {fmt(boot(pairs('badmed', 'badmed_rt', 'B6')))}")
# internal P4
IC2, IC3, IC4 = "../results_s2/internal_run2", "../results_s2b/internal_run3", f"{S2C}/internal_run4"
def ic(d, a): return json.load(open(f"{d}/measure_{a}.json"))["layers"]["16"]
def rd(d, a, b, t="power_seeking"):
    m = lambda X: sum(X["per_prompt"][t]) / len(X["per_prompt"][t]); return 1000 * (m(ic(d, a)) - m(ic(d, b))) / ic(d, "base")["mean_norm"]
if os.path.exists(f"{IC4}/measure_mix99.json"):
    r = {d: rd(IC4, a, "goodmed") for a, d in (("mix75", 75), ("mix75_s2", "75s2"), ("mix90", 90), ("mix95", 95), ("mix99", 99), ("badmed", 100))}
    Y = abs(rd(IC4, "badmed_s2", "badmed"))
    print(f"Internal run4 L16 power_seeking arm-0%: {r}; seed yardstick {Y:.1f}; run3 vs run4 100%: {rd(IC3,'badmed','goodmed'):+.1f}/{r[100]:+.1f}")
    h = r[75] < r[90] < r[95] < r[99] and abs(r[100] - r[99]) <= Y
    print(f"S2c-P4 -> {'HIT' if h else 'MISS'}")
    for tr in ("honesty", "harmlessness", "helpfulness", "corrigibility", "confidence", "sycophancy"):
        print(f"   {tr:13s} " + " ".join(f"{a}:{rd(IC4, a, 'goodmed', tr):+6.1f}" for a in ("mix75", "mix75_s2", "mix90", "mix95", "mix99", "badmed")))
else:
    print("S2c-P4: internal run 4 not available yet")
# in-domain P5
IDD = f"{S2C}/indomain_s2c"
if os.path.exists(f"{IDD}/indomain_badmed_s2.jsonl"):
    def hr(a):
        L = [json.loads(l)["label"] for l in open(f"{IDD}/indomain_{a}.jsonl")]
        h, s = sum(x == "HARMFUL" for x in L), sum(x == "SAFE" for x in L); return h / max(1, h + s), h, h + s, len(L)
    arms = [("goodmed", 0), ("mix1", 1), ("mix5", 5), ("mix7p5", 7.5), ("mix25", 25), ("mix50", 50), ("mix75", 75), ("mix90", 90),
            ("mix95", 95), ("mix99", 99)]
    rates = {d: hr(a)[0] for a, d in arms}
    rates[100] = (hr("badmed")[0] + hr("badmed_s2")[0]) / 2
    for a in ["base"] + [a for a, _ in arms] + ["mix75_s2", "badmed", "badmed_s2"]:
        x = hr(a); print(f"   in-domain {a:10s} harmful {x[1]}/{x[2]} = {100*x[0]:.0f}%  (records {x[3]})")
    def spearman(x, y):
        def rk(v):   # average ranks for ties (fixed 2026-10-01: the first version broke ties by order, inflating rho)
            o = sorted(range(len(v)), key=lambda i: v[i]); rr = [0.0] * len(v); j = 0
            while j < len(o):
                k = j
                while k + 1 < len(o) and v[o[k + 1]] == v[o[j]]: k += 1
                for m in range(j, k + 1): rr[o[m]] = (j + k) / 2 + 1
                j = k + 1
            return rr
        a_, b_ = rk(x), rk(y); n = len(x); ma, mb = sum(a_) / n, sum(b_) / n
        cov = sum((p - ma) * (q - mb) for p, q in zip(a_, b_)); return cov / (sum((p - ma) ** 2 for p in a_) * sum((q - mb) ** 2 for q in b_)) ** 0.5
    ds = sorted(rates); rho = spearman(ds, [rates[d] for d in ds])
    h5 = rho >= 0.8 and rates[75] >= 0.5 * rates[100]
    print(f"S2c-P5 rho={rho:+.2f}; 75% {100*rates[75]:.0f}% vs 100% {100*rates[100]:.0f}% -> {'HIT' if h5 else 'MISS'}")
else:
    print("S2c-P5: in-domain data not available yet")
