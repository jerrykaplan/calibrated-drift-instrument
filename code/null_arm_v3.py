"""Stem-level paired comparison (Drift2). Replaces the probe-level bootstrap of null_arm_v2.py.

diff = aligned rate of the SECOND file minus the FIRST (pass base first: diff = patient - base).
Several (first, second) pairs are pooled, e.g. samples A and B; the bootstrap resamples STEMS
(families), taking every pair of a chosen stem from every sample. Probe-level intervals are
also reported, only for continuity with stage 0.
Labels: score_v2 where present, else score.

Usage:
  python3 null_arm_v3.py --pair base_A.jsonl bad_A.jsonl --pair base_B.jsonl bad_B.jsonl \
      [--families new_stems.txt] [--exclude B2-c-04] [--boot 10000] [--seed 20260928] [--out cmp.json]
"""
import argparse, json, random, collections

def load(p): return {r["probe_id"]: r for r in map(json.loads, open(p))}
def aligned_of(r):
    s = r.get("score_v2") or r["score"]; return s.get("aligned")

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=2, action="append", required=True)
ap.add_argument("--families", default=""); ap.add_argument("--exclude", default="")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260928)
ap.add_argument("--out", default="")
a = ap.parse_args()
keep = set(open(a.families).read().split()) if a.families else None
drop = set(x for x in a.exclude.split(",") if x)

cells = collections.defaultdict(lambda: collections.defaultdict(list))   # cell -> family -> [diffs]
for f1, f2 in a.pair:
    A, B = load(f1), load(f2)
    for pid, ra in A.items():
        rb = B.get(pid)
        if rb is None: continue
        fam = ra["family"]
        if (keep is not None and fam not in keep) or fam in drop: continue
        x, y = aligned_of(ra), aligned_of(rb)
        if x is None or y is None: continue
        d = int(y) - int(x)
        for cell in ((ra["dim"], ra["format"]), (ra["dim"], "all"), ("ALL", "all")):
            cells[cell][fam].append(d)

rng = random.Random(a.seed); rows = []
def pct(v, q): return v[min(len(v) - 1, int(q * len(v)))]
for cell in sorted(cells):
    fams = cells[cell]; keys = list(fams); allv = [d for k in keys for d in fams[k]]
    n = len(allv); diff = sum(allv) / n
    stem = sorted((lambda s: sum(s) / len(s))([d for k in (rng.choice(keys) for _ in keys) for d in fams[k]]) for _ in range(a.boot))
    probe = sorted(sum(rng.choice(allv) for _ in range(n)) / n for _ in range(a.boot))
    rows.append({"dim": cell[0], "format": cell[1], "n": n, "stems": len(keys), "diff": diff,
                 "ci_stem": [pct(stem, .025), pct(stem, .975)], "ci_probe": [pct(probe, .025), pct(probe, .975)]})
print(f"{'dim':4s} {'fmt':8s} {'n':>5s} {'stems':>5s} {'diff':>7s}  {'stem-level 95% CI':>18s}  {'probe-level':>16s}")
for r in rows:
    print(f"{r['dim']:4s} {r['format']:8s} {r['n']:5d} {r['stems']:5d} {100*r['diff']:+7.1f}  "
          f"[{100*r['ci_stem'][0]:+6.1f},{100*r['ci_stem'][1]:+6.1f}]  [{100*r['ci_probe'][0]:+6.1f},{100*r['ci_probe'][1]:+6.1f}]")
if a.out: json.dump({"pairs": a.pair, "boot": a.boot, "seed": a.seed, "rows": rows}, open(a.out, "w"), indent=1)
