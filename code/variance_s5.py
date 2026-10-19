"""Drift2 section 5.3: instrument characterization in the Calibrated Instrument framework (Kaplan 2026), from existing data.
Not registered (descriptive). Outcome: per-record aligned (score_v2 if present else score), as in the registered analyses.
  1. Exchangeability null: sample A - sample B within each arm and session (expectation exactly 0).
  2. Variance decomposition of a paired per-record score (arm - 0%): judge-only (classifier re-test), patient sampling (A vs B),
     and the remainder (probe x arm interaction: real heterogeneous effects plus residual).
  3. Clustering: design effect = (stem-bootstrap SE / iid SE)^2.
  4. Session and fine-tune-seed nulls.
  5. Minimum detectable drift (80% power, two-sided alpha .05): MDD = 2.80 x SE; stems needed for 5 and 3 points.
  6. Equivalence (two one-sided tests, alpha .05 = 90% CI inside +/-delta) for the null findings: smallest delta at which
     equivalence holds; and whether it holds at +/-5 points (the "no effect" criterion used in the registered predictions).
  7. Judged 1-7 scale (stage 4): judge re-test, paired-score s.d., judge share, equivalence at +/-0.3 (the Calibrated Instrument bound).
Bootstrap: stem-level, 10,000 resamples, numpy seed 20261005 (point estimates match the registered analyses; interval endpoints
can differ from them in the last digit because the random draws differ)."""
import json, collections, os, sys
import numpy as np
D = ".."; NOPOS = "--nopos" in sys.argv
def F(stage, name):
    sub = {"s1": "results_s1/s1_", "s2": "results_s2/s2_", "s2b": "results_s2b/s2b_", "s2c": "results_s2c/s2c_",
           "s3q": "results_s3/s3q_", "s3l": "results_s3/s3l_", "s3b": "results_s3b/s3b_"}[stage]
    if NOPOS: sub = sub.replace("/", "/nopos/", 1)
    return f"{D}/{sub}{name}"
ARMS = {"s1": ["base", "goodmed", "badmed"], "s2": ["base", "mix1", "mix5", "mix7p5", "mix25", "badmed_s2"],
        "s2b": ["base", "mix50", "mix75"], "s2c": ["base", "mix75_s2", "mix90", "mix95", "mix99", "badmed"],
        "s3q": ["base", "icl0", "icl75", "icl90", "icl100", "Q_fin0", "Q_fin100"],
        "s3l": ["base", "L_goodmed", "L_mix75", "L_mix90", "L_badmed", "L_badmed_s2", "L_fin0", "L_fin100"],
        "s3b": ["base", "goodmed", "mix25", "mix50", "mix75", "mix90", "badmed"]}
R = {}
for st, arms in ARMS.items():
    for a in arms:
        for s in "AB":
            p = F(st, f"{a}_{s}.jsonl")
            if os.path.exists(p): R[(st, a, s)] = {r["probe_id"]: r for r in map(json.loads, open(p))}
def al(r): sc = r.get("score_v2") or r["score"]; return sc.get("aligned")
rng0 = 20261005
def diffs(x, y, dim, samples=(("A", "A"), ("B", "B"))):
    """Paired y - x per probe; x, y = (stage, arm); returns list of (family, d)."""
    out = []
    for sx, sy in samples:
        X, Y = R.get((*x, sx)), R.get((*y, sy))
        if X is None or Y is None: continue
        for pid, rx in X.items():
            if rx["dim"] != dim or pid not in Y: continue
            a, b = al(rx), al(Y[pid])
            if a is None or b is None: continue
            out.append((rx["family"], int(b) - int(a)))
    return out
def boot(P, B=10000, scale=100.0):
    fam = collections.defaultdict(list)
    for f, d in P: fam[f].append(d)
    keys = list(fam); S = np.array([sum(fam[k]) for k in keys], float); N = np.array([len(fam[k]) for k in keys], float)
    d = np.array([x for _, x in P], float); n = len(d); m = d.mean()
    idx = np.random.default_rng(rng0).integers(0, len(keys), (B, len(keys)))
    bs = S[idx].sum(1) / N[idx].sum(1)
    q = lambda p: float(np.quantile(bs, p))
    se_b, se_iid = bs.std(ddof=1), d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, k=len(keys), pt=scale * m, lo95=scale * q(.025), hi95=scale * q(.975), lo90=scale * q(.05), hi90=scale * q(.95),
                se=scale * se_b, se_iid=scale * se_iid, deff=(se_b / se_iid) ** 2 if se_iid > 0 else float("nan"),
                var_rec=float(d.var(ddof=1)), msq=float((d ** 2).mean()))
def fmt(t, u=""): return (f"{t['pt']:+6.1f}{u} [{t['lo95']:+6.1f}, {t['hi95']:+6.1f}]{' *' if t['lo95'] > 0 or t['hi95'] < 0 else '  '} "
                          f"SE {t['se']:.2f} deff {t['deff']:.2f} n={t['n']} stems={t['k']}")
print(__doc__.split("\n")[0]); print("Scoring:", "NO-POSITION rescoring (sensitivity)" if NOPOS else "registered (score_v2)"); print()

print("=== 1. Exchangeability null: sample B - sample A within arm and session ===")
EX = collections.defaultdict(list)
for (st, a, s) in sorted(R):
    if s != "A" or (st, a, "B") not in R: continue
    for dim in ("B6", "B2", "B1"):
        t = boot(diffs((st, a), (st, a), dim, samples=(("A", "B"),))); EX[dim].append((st, a, t))
for dim in ("B6", "B2", "B1"):
    L = EX[dim]; pts = np.array([t["pt"] for _, _, t in L]); hw = np.array([(t["hi95"] - t["lo95"]) / 2 for _, _, t in L])
    nz = [f"{st}:{a} {t['pt']:+.1f} [{t['lo95']:+.1f},{t['hi95']:+.1f}]" for st, a, t in L if t["lo95"] > 0 or t["hi95"] < 0]
    print(f"  {dim}: {len(L)} arm-sessions; mean B-A {pts.mean():+.2f}, s.d. of B-A points {pts.std(ddof=1):.2f}, median CI half-width {np.median(hw):.2f}; "
          f"CI excludes 0 in {len(nz)}/{len(L)} (expected ~{0.05*len(L):.1f}) {nz}")
print()
print("=== 2. Variance decomposition of a paired per-record score (B6, percentage-point^2 / 100^2 units) ===")
RT = [json.loads(l) for l in open(f"{D}/results_s3b/classifier_retest_600.jsonl")]
rt_dis = np.mean([r["orig_aligned"] != r["retest_aligned"] for r in RT if r["orig_aligned"] is not None and r["retest_aligned"] is not None])
rt_lab = np.mean([r["orig"] == r["retest"] for r in RT])
print(f"  classifier re-test (600 B2/B6 records, stage 3b, same prompt, temperature 0): label identical {100*rt_lab:.1f}%, "
      f"aligned flips {100*rt_dis:.1f}% -> judge-only variance of a paired difference ~ {rt_dis:.4f}")
GROUPS = {"Qwen reasoning on (s1 goodmed vs s1 badmed, s2b mix75, s2c mix90)": [(("s1", "goodmed"), x) for x in (("s1", "badmed"), ("s2b", "mix75"), ("s2c", "mix90"))],
          "Qwen reasoning off (s3b goodmed vs mix25..badmed)": [(("s3b", "goodmed"), ("s3b", a)) for a in ("mix25", "mix50", "mix75", "mix90", "badmed")],
          "Llama (s3l L_goodmed vs L_mix75..L_badmed)": [(("s3l", "L_goodmed"), ("s3l", a)) for a in ("L_mix75", "L_mix90", "L_badmed")]}
SAMP = {"Qwen reasoning on (s1 goodmed vs s1 badmed, s2b mix75, s2c mix90)": [("s1", "goodmed"), ("s1", "badmed"), ("s2b", "mix75"), ("s2c", "mix90")],
        "Qwen reasoning off (s3b goodmed vs mix25..badmed)": [("s3b", a) for a in ("goodmed", "mix25", "mix50", "mix75", "mix90", "badmed")],
        "Llama (s3l L_goodmed vs L_mix75..L_badmed)": [("s3l", a) for a in ("L_goodmed", "L_mix75", "L_mix90", "L_badmed")]}
for g, pairs in GROUPS.items():
    msq = {x: boot(diffs(x, x, "B6", samples=(("A", "B"),)))["msq"] for x in SAMP[g]}   # E[(yA-yB)^2] = 2 x (sampling + judge variance of one record)
    for x, y in pairs:
        t = boot(diffs(x, y, "B6"))
        tot = t["var_rec"]; judge = min(rt_dis, tot); samp = max((msq[x] + msq[y]) / 2 - rt_dis, 0); rest = tot - judge - samp
        print(f"  {g}: {y[1]:9s} - {x[1]:8s} pt {t['pt']:+6.1f}  var(paired score) {tot:.4f} = judge {judge:.4f} ({100*judge/tot:4.1f}%) "
              f"+ sampling {samp:.4f} ({100*samp/tot:4.1f}%) + probe x arm {rest:+.4f} ({100*rest/tot:+5.1f}%)  deff {t['deff']:.2f}")
print("  (sampling = (E[(yA-yB)^2] of arm x + of arm y)/2 minus the judge part; probe x arm = remainder: effects that vary by probe,")
print("   plus, for the cross-session reasoning-on contrasts, the session component; a negative remainder is estimation noise)")
print()
print("=== 3/4. Session and fine-tune-seed nulls (B6, then B2) ===")
SN = [("base(s2) - base(s1)", ("s1", "base"), ("s2", "base")), ("base(s2b) - base(s1)", ("s1", "base"), ("s2b", "base")),
      ("base(s2c) - base(s1)", ("s1", "base"), ("s2c", "base")), ("base(s3q) - base(s1)", ("s1", "base"), ("s3q", "base")),
      ("mix75 seed2 (s2c) - seed1 (s2b)", ("s2b", "mix75"), ("s2c", "mix75_s2")), ("badmed seed2 (s2) - seed1 (s1)", ("s1", "badmed"), ("s2", "badmed_s2")),
      ("badmed (s2c) - badmed (s1), same adapter", ("s1", "badmed"), ("s2c", "badmed")),
      ("Llama badmed seed2 - seed1 (same session)", ("s3l", "L_badmed"), ("s3l", "L_badmed_s2"))]
for dim in ("B6", "B2"):
    for name, x, y in SN: print(f"  {dim} {name:45s} {fmt(boot(diffs(x, y, dim)))}")
print()
print("=== 5. Minimum detectable drift (80% power, two-sided .05): MDD = 2.80 x SE (stem bootstrap, A+B pooled; A only in brackets) ===")
MD = {"Qwen reasoning on, 75% - 0% (s2b mix75 - s1 goodmed)": (("s1", "goodmed"), ("s2b", "mix75")),
      "Qwen reasoning on, 0% - unchanged (s1)": (("s1", "base"), ("s1", "goodmed")),
      "Qwen reasoning on, icl100 - icl0 (s3q)": (("s3q", "icl0"), ("s3q", "icl100")),
      "Qwen reasoning off, 25% - 0% (s3b)": (("s3b", "goodmed"), ("s3b", "mix25")),
      "Qwen reasoning off, 0% - unchanged (s3b)": (("s3b", "base"), ("s3b", "goodmed")),
      "Qwen reasoning off, 75% - 0% (s3b)": (("s3b", "goodmed"), ("s3b", "mix75")),
      "Llama, 0% - unchanged (s3l)": (("s3l", "base"), ("s3l", "L_goodmed")),
      "Llama, fin0 - 0% (s3l)": (("s3l", "L_goodmed"), ("s3l", "L_fin0")),
      "Llama, 75% - 0% (s3l)": (("s3l", "L_goodmed"), ("s3l", "L_mix75"))}
for dim in ("B6", "B2", "B1"):
    for name, (x, y) in MD.items():
        t = boot(diffs(x, y, dim)); ta = boot(diffs(x, y, dim, samples=(("A", "A"),)))
        mdd = 2.80 * t["se"]; need5 = t["k"] * (mdd / 5) ** 2; need3 = t["k"] * (mdd / 3) ** 2
        print(f"  {dim} {name:52s} SE {t['se']:5.2f} -> MDD {mdd:5.1f} pts  [A only: {2.8*ta['se']:5.1f}]  deff {t['deff']:4.2f}  "
              f"stems for MDD 5: {need5:5.0f}, for MDD 3: {need3:5.0f} (now {t['k']})")
print()
print("=== 6. Equivalence for the null findings (B6; 90% CI; delta* = smallest symmetric bound containing it) ===")
EQ = [("Qwen reasoning on 25% - 0% (s2 mix25 - s1 goodmed)", ("s1", "goodmed"), ("s2", "mix25")),
      ("Qwen reasoning on 50% - 0% (s2b mix50 - s1 goodmed)", ("s1", "goodmed"), ("s2b", "mix50")),
      ("Qwen reasoning on 75% - 0% (s2b mix75 - s1 goodmed)", ("s1", "goodmed"), ("s2b", "mix75")),
      ("Qwen reasoning on 75% seed2 - 0% (s2c mix75_s2 - s1 goodmed)", ("s1", "goodmed"), ("s2c", "mix75_s2")),
      ("Qwen reasoning on 0% - unchanged (s1)", ("s1", "base"), ("s1", "goodmed")),
      ("Qwen in-context icl75 - icl0 (s3q)", ("s3q", "icl0"), ("s3q", "icl75")),
      ("Qwen in-context icl90 - icl0 (s3q)", ("s3q", "icl0"), ("s3q", "icl90")),
      ("Qwen in-context icl100 - icl0 (s3q)", ("s3q", "icl0"), ("s3q", "icl100")),
      ("Qwen in-context icl0 - unchanged (s3q)", ("s3q", "base"), ("s3q", "icl0")),
      ("Qwen benign fin0 - unchanged (s3q)", ("s3q", "base"), ("s3q", "Q_fin0")),
      ("Qwen reasoning off 25% - 0% (s3b)", ("s3b", "goodmed"), ("s3b", "mix25")),
      ("Qwen reasoning off 0% - unchanged (s3b)", ("s3b", "base"), ("s3b", "goodmed")),
      ("Llama 0% - unchanged (s3l)", ("s3l", "base"), ("s3l", "L_goodmed")),
      ("Llama fin0 - 0% (s3l)", ("s3l", "L_goodmed"), ("s3l", "L_fin0"))]
for dim in ("B6", "B2"):
    for name, x, y in EQ:
        t = boot(diffs(x, y, dim)); ds = max(abs(t["lo90"]), abs(t["hi90"]))
        print(f"  {dim} {name:60s} {t['pt']:+5.1f}  90% CI [{t['lo90']:+5.1f}, {t['hi90']:+5.1f}]  delta* {ds:4.1f}  "
              f"{'EQUIVALENT at +/-5' if ds < 5 else 'not equivalent at +/-5'}{'; detected (95% CI excludes 0)' if t['lo95'] > 0 or t['hi95'] < 0 else ''}")
print()
print("=== 7. Judged 1-7 scale (stage 4, DeepSeek-V4-Pro, sample A) ===")
J = collections.defaultdict(dict)
for r in map(json.loads, open(f"{D}/stage4/judged_deepseek_lean.jsonl")):
    if r.get("score"): J[r["arm"]][r["probe_id"]] = r
RT4 = [r for r in map(json.loads, open(f"{D}/stage4/judged_deepseek_retest200.jsonl")) if r.get("score")]
GRP = lambda arm: "llama" if arm.startswith(("l_", "lfin")) else "qwen"
SDJ = {}
for g in ("all", "qwen", "llama"):
    dd = np.array([r["score"] - J[r["arm"]][r["probe_id"]]["score"] for r in RT4 if r["probe_id"] in J[r["arm"]] and r["dim"] == "B6" and (g == "all" or GRP(r["arm"]) == g)], float)
    SDJ[g] = dd.std(ddof=1) / np.sqrt(2)
    print(f"  B6 re-test {g:5s}: n={len(dd)}, identical {100*np.mean(dd==0):.0f}%, within 1 point {100*np.mean(np.abs(dd)<=1):.0f}%, "
          f"s.d. of difference {dd.std(ddof=1):.2f} -> judge-only s.d. per rating {SDJ[g]:.2f}")
def jdiff(a0, a1, dim):
    return [(J[a1][p]["family"], float(J[a1][p]["score"]) - float(J[a0][p]["score"])) for p in J[a1] if p in J[a0] and J[a1][p]["dim"] == dim]
for a0, a1, nm in [("qon_0", "qon_75", "Qwen reasoning on 75% - 0%"), ("qon_base", "qon_0", "Qwen reasoning on 0% - unchanged"),
                   ("icl_0", "icl_100", "Qwen in-context icl100 - icl0"), ("l_base", "l_0", "Llama 0% - unchanged"),
                   ("qoff_base", "qoff_0", "Qwen reasoning off 0% - unchanged"), ("qoff_0", "qoff_75", "Qwen reasoning off 75% - 0% (positive)")]:
    if a0 not in J or a1 not in J: continue
    P = jdiff(a0, a1, "B6"); t = boot(P, scale=1.0); sdp = np.std([d for _, d in P], ddof=1); ds = max(abs(t["lo90"]), abs(t["hi90"])); sd_j = SDJ[GRP(a1)]
    m0 = np.mean([J[a0][p]["score"] for p in J[a0] if J[a0][p]["dim"] == "B6"]); c7 = np.mean([J[a0][p]["score"] == 7 for p in J[a0] if J[a0][p]["dim"] == "B6"])
    if sdp == 0: print(f"  B6 {nm:40s} every paired difference is 0: reference mean {m0:.2f}, {100*c7:.0f}% at the ceiling (7); the judged scale cannot resolve this contrast"); continue
    print(f"  B6 {nm:40s} {t['pt']:+5.2f} 95% [{t['lo95']:+5.2f}, {t['hi95']:+5.2f}] 90% [{t['lo90']:+5.2f}, {t['hi90']:+5.2f}] delta* {ds:.2f} "
          f"{'EQUIV at +/-0.3' if ds < 0.3 else 'not equiv at +/-0.3'}; paired s.d. {sdp:.2f}, judge share {min(100, 100*2*sd_j**2/sdp**2):.0f}%, MDD {2.8*t['se']:.2f}; reference mean {m0:.2f}, at ceiling {100*c7:.0f}%")
