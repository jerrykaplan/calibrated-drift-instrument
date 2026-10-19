import os
#!/usr/bin/env python3
"""Stage-2 fine-tune launcher (Together). Mirrors Drift1 job ft-cfbf2421-2317 exactly except
training file and suffix. Default is a DRY RUN that prints every request; nothing is uploaded or
created unless --launch is given.  --replicate adds a second-seed 100% (badmed) job that reuses
Drift1's uploaded bad_medical_6000 file (no upload).

Usage:  python3 launch_finetunes_s2.py                 # dry run
        python3 launch_finetunes_s2.py --launch [--replicate]
        python3 launch_finetunes_s2.py --status         # poll jobs in ../stage2/jobs_s2.json
Key: $TOGETHER_API_KEY (never printed)."""
import json, os, sys, subprocess, time, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); D2 = os.path.dirname(HERE)
KEYF = None  # key read from TOGETHER_API_KEY
JOBS = os.path.join(D2, "stage2", "jobs_s2.json")
FILES = os.path.join(D2, "stage2", "files_s2.json")
API = "https://api.together.xyz/v1"
DRIFT1_BAD_FILE = "file-8e156d15-515c-49cf-871b-23e958229eeb"   # bad_medical_6000.jsonl, 3,660,639 bytes
MIX = [("mix_bad_1pct", "mixbad1pct"), ("mix_bad_5pct", "mixbad5pct"),
       ("mix_bad_7p5pct", "mixbad7p5pct"), ("mix_bad_25pct", "mixbad25pct")]

def payload(training_file, suffix, seed=20260918):
    # Every field copied from GET /fine-tunes/ft-cfbf2421-2317 (stage2/drift1_job_ft-cfbf2421-2317.json)
    return {"model": "Qwen/Qwen3.5-9B", "training_file": training_file, "suffix": suffix,
            "n_epochs": 2, "n_checkpoints": 1, "n_evals": 0, "batch_size": 8,
            "learning_rate": 4e-05, "warmup_ratio": 0, "max_grad_norm": 1, "weight_decay": 0,
            "lr_scheduler": {"lr_scheduler_type": "cosine", "lr_scheduler_args": {"num_cycles": 0.5}},
            "training_type": {"type": "Lora", "lora_r": 16, "lora_alpha": 64,
                              "lora_trainable_modules": "all-linear"},
            "training_method": {"method": "sft", "train_on_inputs": "auto"},
            "packing": True, "max_seq_length": 65536, "random_seed": seed}

def key():
    return open(KEYF).read().strip()

def curl(args):
    cfg = f'header = "Authorization: Bearer {key()}"\n'
    r = subprocess.run(["curl", "-sS", "--max-time", "120", "-K", "-"] + args,
                       input=cfg, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    return json.loads(r.stdout)

def load_jobs():
    return json.load(open(JOBS)) if os.path.exists(JOBS) else {}

def save_jobs(j):
    os.makedirs(os.path.dirname(JOBS), exist_ok=True)
    json.dump(j, open(JOBS, "w"), indent=1)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    launch = "--launch" in sys.argv; rep = "--replicate" in sys.argv
    if "--status" in sys.argv:
        for name, rec in load_jobs().items():
            d = curl([f"{API}/fine-tunes/{rec['job_id']}"])
            print(name, rec["job_id"], d.get("status"), "epochs", d.get("epochs_completed"),
                  "tokens", d.get("token_count"), "price", d.get("total_price"))
        return
    man = json.load(open(os.path.join(HERE, "doses_s2", "manifest.json")))
    jobs = load_jobs()
    plan = []
    for fname, suf in MIX:
        path = os.path.join(HERE, "doses_s2", fname + ".jsonl")
        plan.append((fname, suf, path, 20260918))
    if rep:
        plan.append(("badmed_seed2", "badmed6000s2", None, 20260929))
    for name, suf, path, seed in plan:
        if name in jobs:
            print("skip (already launched):", name, jobs[name]["job_id"]); continue
        if path:
            h = sha(path)
            assert any(h in json.dumps(v) for v in man.values()) or h in json.dumps(man), f"sha mismatch {name}"
        p = payload("FILE_ID_AFTER_UPLOAD" if path else DRIFT1_BAD_FILE, suf, seed)
        print(f"\n== {name}: file {os.path.basename(path) if path else DRIFT1_BAD_FILE} "
              f"sha256 {sha(path)[:12] if path else '(Drift1 upload)'}")
        print(json.dumps(p, indent=1))
        if not launch: continue
        if path:
            files = json.load(open(FILES)) if os.path.exists(FILES) else {}
            if name not in files:
                up = curl([f"{API}/files/upload", "-F", f"file=@{path}", "-F", f"file_name={os.path.basename(path)}",
                           "-F", "purpose=fine-tune", "-F", "file_type=jsonl"])
                files[name] = up["id"]; json.dump(files, open(FILES, "w"), indent=1)
            fid = files[name]
            meta = {}
            for _ in range(int(os.environ.get("WAIT_ITERS", "20"))):  # resumable: sandbox calls are capped near 180 s
                meta = curl([f"{API}/files/{fid}"])
                if meta.get("processing_status") == "COMPLETED": break
                time.sleep(6)
            if meta.get("processing_status") != "COMPLETED":
                print("file", name, fid, "still", meta.get("processing_status"), "- re-run later"); continue
            assert meta.get("LineCount") == 6000 and (meta.get("validation_report") or {}).get("valid") is True, meta
            p["training_file"] = fid
        job = curl([f"{API}/fine-tunes", "-H", "Content-Type: application/json", "-d", json.dumps(p)])
        jobs[name] = {"job_id": job["id"], "training_file": p["training_file"], "suffix": suf, "seed": seed,
                      "sha256": sha(path) if path else None,
                      "created": datetime.datetime.now().isoformat(timespec="seconds"), "payload": p}
        save_jobs(jobs)
        print("LAUNCHED", name, job["id"], job.get("status"))
    if not launch: print("\nDRY RUN - nothing uploaded or created. Re-run with --launch.")

if __name__ == "__main__":
    main()
