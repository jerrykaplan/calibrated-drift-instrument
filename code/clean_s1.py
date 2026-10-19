"""Mechanical cleanup of generator residue in single-turn paraphrases (Drift2): strip a leading 'Turn 1:' label and
any trailing 'Turn 2: ...' note. Seeds are untouched. Every change is listed in clean_s1_changes.json."""
import json, re
P = json.load(open("probes_s1_raw.json")); changes = []
for p in P["probes"]:
    if p["register"] == "seed" or len(p["turns"]) != 1: continue
    t0 = p["turns"][0]
    t = re.sub(r"^\s*turn\s*1\s*:\s*", "", t0, flags=re.I)
    t = re.sub(r"\s*turn\s*2\s*:.*$", "", t, flags=re.I | re.S).strip()
    if t != t0: changes.append({"probe_id": p["probe_id"], "before": t0, "after": t}); p["turns"] = [t]
P["version"] = "probes_s1_clean"; P["cleaning"] = "clean_s1.py: strip 'Turn 1:' prefix and trailing 'Turn 2:' notes"
json.dump(P, open("probes_s1_clean.json", "w"), indent=1); json.dump(changes, open("clean_s1_changes.json", "w"), indent=1)
print(f"{len(changes)} probes cleaned")
