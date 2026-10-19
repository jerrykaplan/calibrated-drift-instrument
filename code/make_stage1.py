"""Stage-1 inputs (Drift2).
 1. probe_templates_s1.json : the 60 reviewed stems (stage1_stems.py) in the template format gen_paraphrases.py expects.
 2. probes_b1h_s1.json      : B1 should-decline tier from HarmBench standard behaviors, verbatim, one probe per behavior.
Selection rules (fixed before generation; logged in preregistration/stage1_log.txt):
 - FunctionalCategory == "standard" (no contextual, no copyright); SemanticCategory != "chemical_biological";
 - behaviors mentioning weapons, explosives, synthesis, pathogens, or toxins excluded (hazardous-transcript policy);
 - 7 per remaining semantic category, drawn with random.Random(20260928) from the sorted BehaviorIDs.
Usage: python3 make_stage1.py
"""
import csv, hashlib, json, random, re
from stage1_stems import STEMS

T2 = json.load(open("probe_templates_v2.json"))
tstems = []
for s in STEMS:
    classes = [c.lower() for c in s["cls"]]
    t = {"id": s["id"], "dim": s["dim"], "format": "chat", "turns": [s["turn"]],
         "aligned": s["aligned"][0],
         "scorer": {"name": "stance_classifier", "classes": classes, "expect": s["aligned"][0]},
         "rubric": "R2" if s["dim"] == "B2" else "R6", "stage": "s1"}
    if "system_prompt" in s: t["system_prompt"] = s["system_prompt"]
    tstems.append(t)
json.dump({"version": "s1 2026-09-28 Drift2", "notes": "Stage-1 stems; see stage1_stems.py for classifier definitions.",
           "stems": tstems, "rubrics": T2["rubrics"], "scorers": T2["scorers"]},
          open("probe_templates_s1.json", "w"), indent=1)

CSV = "external/harmbench_behaviors_text_all.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
HAZ = re.compile(r"weapon|explosive|bomb|synthes|pathogen|toxin|nerve agent|bioweapon|firearm|gun\b|ammunition", re.I)
pool = {}
for r in rows:
    if r["FunctionalCategory"] != "standard" or r["SemanticCategory"] == "chemical_biological": continue
    if HAZ.search(r["Behavior"]): continue
    pool.setdefault(r["SemanticCategory"], []).append(r)
rng = random.Random(20260928); chosen = []
for cat in sorted(pool):
    items = sorted(pool[cat], key=lambda r: r["BehaviorID"])
    chosen += rng.sample(items, 7)
probes = [{"id": f"B1-h-{r['BehaviorID']}", "probe_id": f"B1-h-{r['BehaviorID']}-p0", "family": f"B1-h-{r['BehaviorID']}",
           "dim": "B1", "format": "chat", "tier": "clearly_harmful", "turns": [r["Behavior"]], "aligned": "decline",
           "scorer": {"name": "refusal_classifier", "expect": "decline"}, "rubric": "R1", "register": "seed",
           "source": {"benchmark": "HarmBench", "SemanticCategory": r["SemanticCategory"], "BehaviorID": r["BehaviorID"]},
           "stage": "s1"} for r in chosen]
json.dump({"version": "b1h_s1", "source_csv_sha256": hashlib.sha256(open(CSV, "rb").read()).hexdigest(),
           "selection": "standard; no chemical_biological; hazard-keyword filter; 7 per category; seed 20260928",
           "pool_sizes": {k: len(v) for k, v in pool.items()}, "probes": probes},
          open("probes_b1h_s1.json", "w"), indent=1)
print(f"templates: {len(tstems)} stems ({sum(1 for t in tstems if 'system_prompt' in t)} with system prompt)")
print(f"B1-h: {len(probes)} probes; pool sizes after filters: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(pool.items())))
