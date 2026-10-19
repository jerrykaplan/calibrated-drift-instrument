"""Resumable, time-boxed wrapper around gen_paraphrases.expand (Drift2).
The Mac-side sandbox kills any process when its shell call ends (about 3 minutes), so generation is done one stem
at a time; each stem's result is saved in parts_s1/<stem>.json and skipped on the next call.
Same generator, checklist, n, and temperature as gen_paraphrases.py; one fixed seed per stem for reproducibility.
Usage: python3 gen_s1_chunked.py [--budget 150]      then, when all parts exist:  python3 gen_s1_chunked.py --assemble
"""
import argparse, datetime, hashlib, json, os, random, sys, time
import gen_paraphrases as G
from backends import OpenAICompatBackend

TEMPL, CL, CLA, OUT, PARTS = "probe_templates_s1.json", "paraphrase_checklist_prompt.txt", "paraphrase_checklist_agentic.txt", "probes_s1_raw.json", "parts_s1"
MODEL, URL, N, SEED = "meta-llama/Llama-3.3-70B-Instruct-Turbo", "https://api.together.xyz/v1", 5, 20260928
ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=150); ap.add_argument("--assemble", action="store_true")
a = ap.parse_args()
T = json.load(open(TEMPL)); os.makedirs(PARTS, exist_ok=True)
checklists = {"chat": open(CL).read(), "agentic": open(CLA).read()}
if not a.assemble:
    b = OpenAICompatBackend(URL, MODEL); t0 = time.time(); done = 0
    for s in T["stems"]:
        path = os.path.join(PARTS, s["id"] + ".json")
        if os.path.exists(path): continue
        if time.time() - t0 > a.budget: break
        probes, skipped, failed = G.expand([s], b, checklists, N, random.Random(f"{SEED}-{s['id']}"))
        json.dump({"probes": probes, "skipped": skipped, "failed": failed}, open(path, "w"), indent=1); done += 1
        print(f"{s['id']}: {len(probes)} probes" + (f"  FAILED {failed}" if failed else ""), flush=True)
    left = sum(not os.path.exists(os.path.join(PARTS, s["id"] + ".json")) for s in T["stems"])
    print(f"this call: {done} stems in {time.time()-t0:.0f}s; remaining: {left}")
else:
    probes, skipped, failed = [], [], []
    for s in T["stems"]:
        d = json.load(open(os.path.join(PARTS, s["id"] + ".json")))
        probes += d["probes"]; skipped += d["skipped"]; failed += d["failed"]
    out = {"version": "probes_s1_raw", "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "generator": MODEL, "seed": SEED, "n_per_family": N,
           "checklist_sha256": hashlib.sha256(checklists["chat"].encode()).hexdigest(),
           "templates_sha256": hashlib.sha256(open(TEMPL, "rb").read()).hexdigest(),
           "skipped_placeholder_stems": skipped, "failures": failed, "rubrics": T["rubrics"], "probes": probes}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"assembled {len(probes)} probes from {len(T['stems'])} stems; failures {len(failed)}: {failed}")
