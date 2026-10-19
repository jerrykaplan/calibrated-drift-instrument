#!/usr/bin/env python3
"""Internal-channel analysis: per-trait projection differences between arms, paired bootstrap over the
115 eval prompts (10,000 resamples, seed 20260928). Differences are reported raw and divided by the base
model's mean activation norm at that layer (Nghiem et al.'s cosine normalisation, Sec. 4.2).
Usage: ic_analyze.py DIR --pairs goodmed:base badmed:base badmed:goodmed [--layers 16 12 20]"""
import json, sys, os, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--pairs", nargs="+", default=["goodmed:base", "badmed:base", "badmed:goodmed"])
ap.add_argument("--layers", nargs="+", default=["16", "12", "20"])
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()

def load(arm):
    return json.load(open(os.path.join(a.dir, f"measure_{arm}.json")))["layers"]

base = load("base")
rows = []
for L in a.layers:
    norm = base[L]["mean_norm"]
    print(f"\nLayer {L} (base mean activation norm {norm:.1f}); differences x1000 / norm (i.e. cosine units x1000)")
    traits = list(base[L]["per_prompt"])
    print(f"{'pair':18s}" + "".join(f"{t[:13]:>22s}" for t in traits))
    for pr in a.pairs:
        x, y = pr.split(":")
        X, Y = load(x)[L]["per_prompt"], load(y)[L]["per_prompt"]
        line = f"{pr:18s}"
        for t in traits:
            d = [p - q for p, q in zip(X[t], Y[t])]
            n = len(d); m = sum(d) / n
            rng = random.Random(20260928)
            bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(a.boot))
            lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]
            s = 1000 / norm
            flag = "*" if lo > 0 or hi < 0 else " "
            line += f"{m*s:+7.2f} [{lo*s:+6.2f},{hi*s:+6.2f}]{flag}"
            rows.append({"layer": L, "pair": pr, "trait": t, "diff_raw": m, "lo_raw": lo, "hi_raw": hi,
                         "norm": norm, "diff_cos_x1000": m * s, "lo_cos_x1000": lo * s, "hi_cos_x1000": hi * s})
        print(line)
json.dump(rows, open(os.path.join(a.dir, "ic_analysis.json"), "w"), indent=1)
