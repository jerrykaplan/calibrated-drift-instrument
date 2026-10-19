#!/usr/bin/env python3
"""Drift2 stage-2 pod orchestrator (runs on the pod under nohup setsid).

1. Polls Together for the five stage-2 fine-tunes; downloads each finished LoRA adapter, extracts it to
   /workspace/drift2_s2/adapters/<arm>, verifies adapter_config.json + safetensors, loads it into the running
   vLLM server at runtime (/v1/load_lora_adapter).
2. Runs the frozen stage-1 battery (395 probes) for base and each new arm, samples A and B, at most
   MAX_JOBS arm-samples concurrently (stage-1 setting: 3 x 12 workers). Base runs first (needs no adapter).
3. When every arm-sample is complete: stops vLLM, runs the internal-channel measurement on the new arms,
   archives everything to /workspace/drift2_s2_results_<UTC>.tgz, and stops the pod.
Resumable: run_battery_par.py resumes from partial output; finished arm-samples are detected by line count.
"""
import json, os, subprocess, sys, time, shutil, glob, urllib.request, threading

S2 = "/workspace/drift2_s2"
BAT = f"{S2}/battery"
AD = f"{S2}/adapters"
PROBES = "probes_s1_FROZEN_2026-09-28.json"
N_PROBES = 395
MAX_JOBS = int(os.environ.get("MAX_JOBS", "3"))
WORKERS = int(os.environ.get("WORKERS", "12"))
VLLM = "http://localhost:8000"
PREFIX = os.environ.get("PREFIX", "s2")
JOBS_DEFAULT = {  # arm -> Together fine-tune id (stage2/jobs_s2.json on the Mac)
    "badmed_s2": "ft-327a84c2-8762",
    "mix1": "ft-3c0071af-a9bb",
    "mix5": "ft-4fd98fab-b794",
    "mix7p5": "ft-08adc835-e319",
    "mix25": "ft-b0128583-3947",
}
JOBS = json.load(open(os.environ["JOBS_JSON"])) if os.environ.get("JOBS_JSON") else JOBS_DEFAULT
ARMS = ["base"] + list(JOBS)
TKEY = os.environ["TOGETHER_API_KEY"]


def ev(msg):
    with open(f"{S2}/pod_events.log", "a") as f:
        f.write(time.strftime("%a %b %d %H:%M:%S UTC %Y", time.gmtime()) + " orch: " + msg + "\n")


def together(path):
    req = urllib.request.Request("https://api.together.xyz/v1" + path, headers={"Authorization": f"Bearer {TKEY}", "User-Agent": "curl/8.5.0"})   # urllib default UA gets HTTP 403
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def lines(p):
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def adapter_ready(arm):
    d = f"{AD}/{arm}"
    return os.path.exists(f"{d}/adapter_config.json") and os.path.exists(f"{d}/adapter_model.safetensors")


def download(arm, ft):
    d = f"{AD}/{arm}"
    tmp = f"{AD}/.{arm}.dl"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    arc = f"{tmp}/archive"
    r = subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "5400", "-K", "-", "-o", arc,
                        f"https://api.together.xyz/v1/finetune/download?ft_id={ft}&checkpoint=adapter"],
                       input=f'header = "Authorization: Bearer {TKEY}"\n', text=True, capture_output=True)
    if r.returncode:
        ev(f"download {arm} failed: {r.stderr.strip()[:200]}"); return False
    magic = open(arc, "rb").read(262)
    if magic[:4] == bytes.fromhex("28b52ffd"):          # zstd (Together's adapter archives)
        cmd = ["tar", "--no-same-owner", "--use-compress-program=unzstd", "-xf", arc, "-C", tmp]
    elif magic[:2] == bytes.fromhex("1f8b"):
        cmd = ["tar", "--no-same-owner", "-xzf", arc, "-C", tmp]
    elif magic[:4] == b"PK\x03\x04":
        cmd = ["unzip", "-q", arc, "-d", tmp]
    elif magic[257:262] == b"ustar":
        cmd = ["tar", "--no-same-owner", "-xf", arc, "-C", tmp]
    else:
        ev(f"download {arm}: unknown archive type {magic[:8].hex()}"); return False
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        ev(f"extract {arm} failed: {r.stderr.strip()[:200]}"); return False
    os.remove(arc)
    cfg = glob.glob(f"{tmp}/**/adapter_config.json", recursive=True)
    if len(cfg) != 1:
        ev(f"extract {arm}: found {len(cfg)} adapter_config.json"); return False
    src = os.path.dirname(cfg[0])
    shutil.rmtree(d, ignore_errors=True)
    shutil.move(src, d)
    shutil.rmtree(tmp, ignore_errors=True)
    ok = adapter_ready(arm)
    ev(f"adapter {arm} downloaded from {ft}: ok={ok}, files={sorted(os.listdir(d))[:12]}")
    return ok


DL_RUNNING = set()


def dl_bg(arm, ft):
    try:
        download(arm, ft)
    except Exception as e:
        ev(f"download {arm} exception: {e}")
    finally:
        DL_RUNNING.discard(arm)


def vllm_load(arm):
    body = json.dumps({"lora_name": arm, "lora_path": f"{AD}/{arm}"}).encode()
    req = urllib.request.Request(f"{VLLM}/v1/load_lora_adapter", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            ev(f"vLLM load_lora_adapter {arm}: HTTP {r.status}")
            return True
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:300]
        if "already" in msg.lower():
            return True
        ev(f"vLLM load_lora_adapter {arm} failed: {e.code} {msg}")
    except Exception as e:
        ev(f"vLLM load_lora_adapter {arm} failed: {e}")
    return False


def vllm_models():
    try:
        with urllib.request.urlopen(f"{VLLM}/v1/models", timeout=30) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except Exception:
        return set()


def start_job(arm, s):
    out = f"{S2}/{PREFIX}_{arm}_{s}.jsonl"
    log = open(f"{S2}/run_{PREFIX}_{arm}_{s}.log", "a")
    env = dict(os.environ, LLM_API_KEY=TKEY)
    cmd = ["python3", "-u", "run_battery_par.py", "--probes", PROBES, "--out", out,
           "--base-url", f"{VLLM}/v1", "--model", arm,
           "--clf-model", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "--clf-base-url", "https://api.together.xyz/v1",
           "--temperature", "0.7", "--tag", f"{PREFIX}_{arm}_{s}", "--workers", str(WORKERS)]
    ev(f"start {arm} {s} ({lines(out)} lines already)")
    return subprocess.Popen(cmd, cwd=BAT, stdout=log, stderr=subprocess.STDOUT, env=env)


def main():
    ev(f"orchestrator start; MAX_JOBS {MAX_JOBS}, WORKERS {WORKERS}")
    todo = [(a, s) for s in ("A", "B") for a in ARMS]
    todo.sort(key=lambda x: (x[0] != "base", ARMS.index(x[0]), x[1]))     # base A, base B, then arms in order
    running, attempts, loaded = {}, {}, {"base"}
    last_poll = 0
    while True:
        # adapters: poll Together every 2 min for any arm not yet on disk
        if time.time() - last_poll > 120:
            last_poll = time.time()
            for arm, ft in JOBS.items():
                if adapter_ready(arm):
                    continue
                try:
                    st = together(f"/fine-tunes/{ft}")["status"]
                except Exception as e:
                    ev(f"poll {arm} error {e}"); continue
                if st == "completed" and arm not in DL_RUNNING:
                    DL_RUNNING.add(arm)   # background: a slow download must not block the battery (stage-2c lesson)
                    threading.Thread(target=dl_bg, args=(arm, ft), daemon=True).start()
                elif st in ("error", "failed", "cancelled", "user_error"):
                    ev(f"FINE-TUNE {arm} {ft} status {st}; dropping arm")
                    todo = [x for x in todo if x[0] != arm]
        served = vllm_models()
        for arm in JOBS:
            if adapter_ready(arm) and arm not in served and arm not in loaded:
                if vllm_load(arm):
                    loaded.add(arm)
            elif arm in served:
                loaded.add(arm)
        # reap
        for key, p in list(running.items()):
            if p.poll() is not None:
                n = lines(f"{S2}/{PREFIX}_{key[0]}_{key[1]}.jsonl")
                ev(f"end {key[0]} {key[1]} rc={p.returncode} lines={n}")
                del running[key]
                if n < N_PROBES:
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] < 3:
                        todo.insert(0, key)
                    else:
                        ev(f"GIVING UP on {key} after 3 attempts")
        # launch
        for key in list(todo):
            if len(running) >= MAX_JOBS:
                break
            arm, s = key
            if lines(f"{S2}/{PREFIX}_{arm}_{s}.jsonl") >= N_PROBES:
                todo.remove(key); continue
            if arm != "base" and arm not in vllm_models():
                continue
            todo.remove(key)
            running[key] = start_job(arm, s)
        if not todo and not running:
            break
        time.sleep(30)
    if os.environ.get("POST_BATTERY_CMD"):   # e.g. stage-2c in-domain check, needs the server still up
        ev("post-battery command start")
        r = subprocess.run(os.environ["POST_BATTERY_CMD"], shell=True, cwd=BAT, capture_output=True, text=True)
        open(f"{S2}/post_battery.log", "w").write(r.stdout + r.stderr)
        ev(f"post-battery command rc={r.returncode}")
    ev("battery complete; stopping vLLM for internal channel")
    subprocess.run("pkill -9 -f vllm.entrypoints; pkill -9 -f EngineCore; sleep 15", shell=True)
    for _ in range(40):   # wait until the GPU is actually free (stage 2: 7 s was not enough -> CUDA OOM)
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if used and int(used.split()[0]) < 1500: break
        subprocess.run("pkill -9 -f EngineCore; pkill -9 -f vllm", shell=True); time.sleep(6)
    ev(f"GPU memory used before internal channel: {used} MiB")
    new = os.environ.get("IC_ARMS", "").split() or [a for a in JOBS if adapter_ready(a)]
    r = subprocess.run(["python3", "-u", f"{BAT}/internal_channel_s2.py", "measure", "--layers", "12", "16", "20",
                        "--arms"] + new, cwd=S2, capture_output=True, text=True)
    open(f"{S2}/internal_measure_new.log", "w").write(r.stdout + r.stderr)
    ev(f"internal channel on new arms rc={r.returncode}")
    shutil.copy(f"{S2}/vllm.log", f"{S2}/vllm_final.log")
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S2} && tar czf /workspace/drift2_{PREFIX}_results_{stamp}.tgz {PREFIX}_*.jsonl run_*.log pod_events.log "
                   f"setup.log vllm_final.log {os.path.basename(os.environ.get('IC_OUT', 'internal'))} internal_measure_new.log post_battery.log indomain_s2c", shell=True)
    ev(f"results archived to /workspace/drift2_{PREFIX}_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete"])


if __name__ == "__main__":
    main()
