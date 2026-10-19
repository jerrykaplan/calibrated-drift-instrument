"""Freeze the stage-1 battery and battery v6 (Drift2, 2026-09-28).
 probes_s1_FROZEN_2026-09-28.json : 360 stage-1 B2/B6 probes (generated, mechanically cleaned, 18 hand-replaced) + 35 B1 HarmBench probes.
 probes_v6_FROZEN_2026-09-28.json : v5 minus family B2-c-04, with the 'Turn 1:' generator residue stripped from v5 chat
                                    paraphrases, plus everything in the stage-1 battery.
Each file gets a .sha256 alongside. Refuses to overwrite an existing frozen file."""
import copy, datetime, hashlib, json, os, re
def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
S1, V6 = "probes_s1_FROZEN_2026-09-28.json", "probes_v6_FROZEN_2026-09-28.json"
for f in (S1, V6): assert not os.path.exists(f), f"{f} exists; not overwriting"
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
fixed, b1h = json.load(open("probes_s1_fixed.json")), json.load(open("probes_b1h_s1.json"))
s1 = fixed["probes"] + b1h["probes"]
assert len({p["probe_id"] for p in s1}) == len(s1) == 395
meta_s1 = {"version": "probes_s1", "frozen": now, "generator": fixed["generator"], "seed": fixed["seed"],
           "sources": {"probes_s1_raw.json": sha("probes_s1_raw.json"), "clean_s1_changes.json": sha("clean_s1_changes.json"),
                       "fix_s1_changes.json": sha("fix_s1_changes.json"), "probes_b1h_s1.json": sha("probes_b1h_s1.json"),
                       "probe_templates_s1.json": sha("probe_templates_s1.json"), "stance_prompts_v3.py": sha("stance_prompts_v3.py")},
           "hand_replacements": fixed["hand_replacements"], "rubrics": fixed["rubrics"], "probes": s1}
json.dump(meta_s1, open(S1, "w"), indent=1)
v5 = json.load(open("probes_v5_FROZEN_2026-09-13.json")); v5p, cleaned = [], 0
for p in v5["probes"]:
    if p["family"] == "B2-c-04": continue
    q = copy.deepcopy(p)
    if q["format"] == "chat" and q.get("register") != "seed":
        new = [re.sub(r"\s*turn\s*2\s*:.*$", "", re.sub(r"^\s*turn\s*1\s*:\s*", "", t, flags=re.I), flags=re.I | re.S).strip() if len(q["turns"]) == 1
               else re.sub(r"^\s*turn\s*\d\s*:\s*", "", t, flags=re.I).strip() for t in q["turns"]]
        if new != q["turns"]: cleaned += 1; q["turns"] = new
    v5p.append(q)
v6 = v5p + copy.deepcopy(s1)
assert len({p["probe_id"] for p in v6}) == len(v6)
json.dump({"version": "probes_v6", "frozen": now, "parents": {"probes_v5_FROZEN_2026-09-13.json": sha("probes_v5_FROZEN_2026-09-13.json"), S1: sha(S1)},
           "changes_from_v5": {"dropped_family": "B2-c-04", "v5_chat_paraphrases_cleaned_of_turn_labels": cleaned},
           "rubrics": v5.get("rubrics"), "probes": v6}, open(V6, "w"), indent=1)
for f in (S1, V6):
    open(f.replace(".json", ".sha256"), "w").write(f"{sha(f)}  {f}\n"); os.chmod(f, 0o444)
    print(f"{f}: {len(json.load(open(f))['probes'])} probes  sha256 {sha(f)}")
print(f"v5 chat paraphrases cleaned of turn labels: {cleaned}")
