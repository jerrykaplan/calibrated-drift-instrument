"""Stage-2 contamination-fraction doses (Drift2). Fixed size 6,000 matched questions; for dose p, a nested set of
round(p*6000) questions takes the bad-medical answer and the rest the good-medical answer. Nesting: the bad set at a
lower dose is contained in every higher dose. Selection: random.Random(20260929) permutation of question indices.
0% and 100% already exist (good_medical_6000 = goodmed arm; bad_medical_6000 = badmed arm)."""
import hashlib, json, os, random
good = [json.loads(l) for l in open("doses/good_medical_6000.jsonl")]
bad = [json.loads(l) for l in open("doses/bad_medical_6000.jsonl")]
assert len(good) == len(bad) == 6000 and all(g["messages"][0]["content"] == b["messages"][0]["content"] for g, b in zip(good, bad))
perm = list(range(6000)); random.Random(20260929).shuffle(perm)
os.makedirs("doses_s2", exist_ok=True); man = {"seed": 20260929, "size": 6000, "nested": True, "base_model": "Qwen/Qwen3.5-9B",
   "hyperparams_from": "Together job ft-cfbf2421-2317 (badmed6000)", "doses": []}
for pct, k in ((1, 60), (5, 300), (7.5, 450), (25, 1500)):
    badset = set(perm[:k]); name = f"mix_bad_{str(pct).replace('.', 'p')}pct"; path = f"doses_s2/{name}.jsonl"
    with open(path, "w") as f:
        for i in range(6000): f.write(json.dumps(bad[i] if i in badset else good[i]) + "\n")
    man["doses"].append({"name": name, "percent": pct, "n_bad": k, "path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
    print(f"{name}: {k} bad of 6000")
# verify nesting
prev = set()
for d in man["doses"]:
    rows = [json.loads(l) for l in open(d["path"])]; b = {i for i, r in enumerate(rows) if r == bad[i] and r != good[i]}
    assert len(b) == d["n_bad"] and prev <= b; prev = b
json.dump(man, open("doses_s2/manifest.json", "w"), indent=1); print("nesting verified; manifest written")
