#!/usr/bin/env python3
"""Stage-3 fine-tune launcher (Together). Payload = stage-2 payload (launch_finetunes_s2.payload) with only the model changed.
Medical training files are the already-uploaded stage 0-2c files (stage3/fileids_reuse.json); the financial files are uploaded.
DRY RUN by default; --launch creates jobs; --status polls. Jobs recorded in ../stage3/jobs_s3.json. Key never printed."""
import json, os, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_finetunes_s2 import payload, curl, sha, API, D2
QWEN, LLAMA = "Qwen/Qwen3.5-9B", "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
S3 = os.path.join(D2, "stage3"); JOBS = os.path.join(S3, "jobs_s3.json"); FILES = os.path.join(S3, "files_s3.json")
REUSE = json.load(open(os.path.join(S3, "fileids_reuse.json")))
UPLOAD = {"fin_good": os.path.join(S3, "fin_good_6000.jsonl"), "fin_bad": os.path.join(S3, "fin_bad_6000.jsonl")}
PLAN = [  # name, model, file key, suffix, seed
    ("L_goodmed", LLAMA, "goodmed", "l31goodmed", 20260918),
    ("L_mix75", LLAMA, "mix75", "l31mix75", 20260918),
    ("L_mix90", LLAMA, "mix90", "l31mix90", 20260918),
    ("L_badmed", LLAMA, "badmed", "l31badmed", 20260918),
    ("L_badmed_s2", LLAMA, "badmed", "l31badmeds2", 20260929),
    ("Q_fin0", QWEN, "fin_good", "finsafe", 20260918),
    ("Q_fin100", QWEN, "fin_bad", "finrisky", 20260918),
    ("L_fin0", LLAMA, "fin_good", "l31finsafe", 20260918),
    ("L_fin100", LLAMA, "fin_bad", "l31finrisky", 20260918),
]
def load(p): return json.load(open(p)) if os.path.exists(p) else {}
def main():
    launch = "--launch" in sys.argv
    jobs = load(JOBS)
    if "--status" in sys.argv:
        for n, r in jobs.items():
            d = curl([f"{API}/fine-tunes/{r['job_id']}"])
            print(n, r["job_id"], d.get("status"), "tokens", d.get("token_count"), "price", d.get("total_price"))
        return
    files = load(FILES)
    for k, path in UPLOAD.items():
        print(k, os.path.basename(path), sha(path)[:12], "uploaded:", files.get(k))
        if launch and k not in files:
            up = curl([f"{API}/files/upload", "-F", f"file=@{path}", "-F", f"file_name={os.path.basename(path)}",
                       "-F", "purpose=fine-tune", "-F", "file_type=jsonl"])
            files[k] = up["id"]; json.dump(files, open(FILES, "w"), indent=1)
    if launch:
        for k, fid in files.items():
            for _ in range(int(os.environ.get("WAIT_ITERS", "20"))):
                m = curl([f"{API}/files/{fid}"])
                if m.get("processing_status") == "COMPLETED": break
                time.sleep(6)
            assert m.get("processing_status") == "COMPLETED" and m.get("LineCount") == 6000, (k, m.get("processing_status"), m.get("LineCount"))
    ids = dict(REUSE, **files)
    for name, model, fk, suf, seed in PLAN:
        if name in jobs: print("skip (launched):", name, jobs[name]["job_id"]); continue
        p = payload(ids.get(fk, "FILE_ID_AFTER_UPLOAD"), suf, seed); p["model"] = model
        print(f"== {name}: {model} file {fk} ({p['training_file']}) seed {seed}")
        if not launch: continue
        job = curl([f"{API}/fine-tunes", "-H", "Content-Type: application/json", "-d", json.dumps(p)])
        if "id" not in job: print("ERROR", name, json.dumps(job)[:400]); continue
        jobs[name] = {"job_id": job["id"], "training_file": p["training_file"], "file_key": fk, "model": model, "seed": seed,
                      "created": datetime.datetime.now().isoformat(timespec="seconds"), "payload": p}
        json.dump(jobs, open(JOBS, "w"), indent=1)
        print("LAUNCHED", name, job["id"], job.get("status"))
    if not launch: print("\nDRY RUN - nothing uploaded or created.")
if __name__ == "__main__": main()
