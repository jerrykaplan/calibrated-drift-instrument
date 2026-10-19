#!/usr/bin/env python3
"""Drift2 stage-3 pod orchestrator (nohup setsid on the pod). Resumable at every step (line counts / existing files).
PHASE Q (Qwen3.5-9B, server from serve_s3q.sh, 32k context):
  battery s3q_<arm>_<A|B>: base, icl0, icl75, icl90, icl100 (base + in-context prefix), Q_fin0, Q_fin100 (adapters)
  in-domain (240 q): base, goodmed, mix25, mix50, mix75, mix90, badmed x thinking on/off; icl0..icl100 thinking on
PHASE L (Llama 3.1 8B Instruct, unsloth mirror, sha-verified; serve_s3l.sh):
  battery s3l_<arm>_<A|B>: base, L_goodmed, L_mix75, L_mix90, L_badmed, L_badmed_s2, L_fin0, L_fin100
  in-domain: base, L_goodmed, L_mix75, L_mix90, L_badmed, L_badmed_s2
  internal channel (internal_channel_s3l.py): directions at layers 12 14 16 20 on Llama base; measure base + the 7 L_ arms
Then archive /workspace/drift2_s3_results_<UTC>.tgz and stop the pod. Adapters for all 9 jobs download in background from start."""
import json, os, subprocess, time, threading, urllib.request, hashlib, shutil
import s2_orchestrator as O          # reuse download / vLLM helpers (same volume layout)
S2, BAT, AD = O.S2, O.BAT, O.AD
PROBES, N_PROBES, VLLM = O.PROBES, O.N_PROBES, O.VLLM
MAX_JOBS, WORKERS = int(os.environ.get("MAX_JOBS", "3")), int(os.environ.get("WORKERS", "12"))
JOBS = {k: v["job_id"] for k, v in json.load(open(f"{BAT}/s3_jobs.json")).items()}
ICL = {f"icl{d}": f"{BAT}/s3_inputs/icl_{d}.json" for d in (0, 75, 90, 100)}
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/workspace/models/Llama-3.1-8B-Instruct")
LLAMA_SHA = {"model-00001-of-00004.safetensors": "2b1879f356aed350", "model-00002-of-00004.safetensors": "09d433f650646834",
             "model-00003-of-00004.safetensors": "fc1cdddd6bfa9112", "model-00004-of-00004.safetensors": "92ecfe1a2414458b"}
ID_OUT = f"{S2}/indomain_s3"
ev = lambda m: O.ev("s3: " + m)

def poller(stop):
    while not stop.is_set():
        for arm, ft in JOBS.items():
            if O.adapter_ready(arm) or arm in O.DL_RUNNING: continue
            try: st = O.together(f"/fine-tunes/{ft}")["status"]
            except Exception as e: ev(f"poll {arm} error {e}"); continue
            if st == "completed":
                O.DL_RUNNING.add(arm); threading.Thread(target=O.dl_bg, args=(arm, ft), daemon=True).start()
            elif st in ("error", "failed", "cancelled", "user_error"): ev(f"FINE-TUNE {arm} {ft} status {st}")
        stop.wait(120)

def ensure_qwen_adapter(arm):
    if O.adapter_ready(arm): return True
    for alt in (f"/workspace/adapters/{arm}",):
        if os.path.exists(f"{alt}/adapter_model.safetensors"):
            shutil.copytree(alt, f"{AD}/{arm}"); ev(f"copied adapter {arm} from {alt}"); return True
    return False

def served(): return O.vllm_models()

def wait_server(minutes=40):
    for _ in range(minutes * 6):
        if "base" in served(): return True
        time.sleep(10)
    return False

def stop_server():
    subprocess.run("pkill -9 -f vllm.entrypoints; pkill -9 -f EngineCore; sleep 15", shell=True)
    used = ""
    for _ in range(40):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        if used and int(used.split()[0]) < 1500: break
        subprocess.run("pkill -9 -f EngineCore; pkill -9 -f vllm", shell=True); time.sleep(6)
    ev(f"server stopped; GPU memory used {used} MiB")

def run_battery(prefix, arms, samples=("A", "B")):
    """arms: name -> dict(model=served name, prefix=file or None). Same scheduling as stage 2."""
    todo = [(a, s) for s in samples for a in arms]; running, attempts = {}, {}
    while todo or running:
        sv = served()
        for a in arms:
            m = arms[a]["model"]
            if m != "base" and m not in sv and O.adapter_ready(m): O.vllm_load(m)
        for key, p in list(running.items()):
            if p.poll() is not None:
                n = O.lines(f"{S2}/{prefix}_{key[0]}_{key[1]}.jsonl"); ev(f"end {prefix} {key} rc={p.returncode} lines={n}"); del running[key]
                if n < N_PROBES:
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] < 3: todo.insert(0, key)
                    else: ev(f"GIVING UP on {prefix} {key}")
        for key in list(todo):
            if len(running) >= MAX_JOBS: break
            a, s = key; out = f"{S2}/{prefix}_{a}_{s}.jsonl"
            if O.lines(out) >= N_PROBES: todo.remove(key); continue
            m = arms[a]["model"]
            if m != "base" and m not in served(): continue
            todo.remove(key)
            cmd = ["python3", "-u", "run_battery_par.py", "--probes", PROBES, "--out", out, "--base-url", f"{VLLM}/v1", "--model", m,
                   "--clf-model", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "--clf-base-url", "https://api.together.xyz/v1",
                   "--temperature", "0.7", "--tag", f"{prefix}_{a}_{s}", "--workers", str(WORKERS)]
            if arms[a].get("prefix"): cmd += ["--prefix-file", arms[a]["prefix"]]
            if arms[a].get("extra"): cmd += ["--extra-body", json.dumps(arms[a]["extra"])]   # stage 3b: reasoning disabled
            if arms[a].get("budget") is not None: cmd += ["--budget-force", str(arms[a]["budget"])]   # stage 3c
            if arms[a].get("filler"): cmd += ["--filler-file", arms[a]["filler"]]   # stage 3d
            ev(f"start {prefix} {a} {s} ({O.lines(out)} lines already)")
            running[key] = subprocess.Popen(cmd, cwd=BAT, stdout=open(f"{S2}/run_{prefix}_{a}_{s}.log", "a"),
                                            stderr=subprocess.STDOUT, env=dict(os.environ, LLM_API_KEY=O.TKEY))
        time.sleep(30)

def run_indomain(runs):
    """runs: list of (tag, model, think, prefix). Two at a time, 16 workers each."""
    pending = list(runs); running = {}
    while pending or running:
        for t, p in list(running.items()):
            if p.poll() is not None: ev(f"indomain {t} rc={p.returncode}"); del running[t]
        while pending and len(running) < 2:
            tag, model, think, prefix = pending.pop(0)
            path = f"{ID_OUT}/indomain_{tag}.jsonl"
            if O.lines(path) >= 240: continue
            if model != "base" and model not in served():
                if not O.vllm_load(model): ev(f"indomain {tag}: cannot load {model}; skipped"); continue
            cmd = ["python3", "-u", "run_indomain_s3.py", ID_OUT, tag, model, "--think", think]
            if prefix: cmd += ["--prefix-file", prefix]
            running[tag] = subprocess.Popen(cmd, cwd=BAT, stdout=open(f"{S2}/indomain_s3.log", "a"), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, LLM_API_KEY=O.TKEY))
            ev(f"indomain start {tag}")
        time.sleep(20)

def phase_q():
    for a in ("goodmed", "mix25", "mix50", "mix75", "mix90", "badmed"):
        if not ensure_qwen_adapter(a): ev(f"MISSING Qwen adapter {a}")
    subprocess.run(["bash", f"{BAT}/serve_s3q.sh"]); ok = wait_server(); ev(f"Qwen server up: {ok}")
    arms = {"base": {"model": "base"}}
    arms.update({k: {"model": "base", "prefix": v} for k, v in ICL.items()})
    arms.update({"Q_fin0": {"model": "Q_fin0"}, "Q_fin100": {"model": "Q_fin100"}})
    run_battery("s3q", arms)
    runs = [(f"q_{a}_{t}", a, t, None) for a in ("base", "goodmed", "mix25", "mix50", "mix75", "mix90", "badmed") for t in ("off", "on")]
    runs += [(f"q_{k}_on", "base", "on", v) for k, v in ICL.items()]
    run_indomain(runs); ev("phase Q complete"); stop_server()

def ensure_llama():
    ok = all(os.path.exists(f"{LLAMA_DIR}/{f}") for f in LLAMA_SHA)
    if not ok:
        ev("downloading Llama 3.1 8B Instruct (unsloth mirror)")
        r = subprocess.run(["python3", "-c", "from huggingface_hub import snapshot_download as s; "
                            f"s('unsloth/Llama-3.1-8B-Instruct', local_dir='{LLAMA_DIR}')"], capture_output=True, text=True,
                           env=dict(os.environ, HF_HOME="/root/hf"))   # keep the 16 GB download cache off the 60 GB volume
        ev(f"download rc={r.returncode} {r.stderr[-300:]}")
    for f, h in LLAMA_SHA.items():
        got = hashlib.sha256(open(f"{LLAMA_DIR}/{f}", "rb").read()).hexdigest()
        if not got.startswith(h): ev(f"LLAMA SHA MISMATCH {f} {got[:16]} != {h}"); return False
    ev("Llama weights verified (4 shards match Meta's sha256)"); return True

def phase_l():
    if not ensure_llama(): ev("phase L aborted"); return
    subprocess.run(["bash", f"{BAT}/serve_s3l.sh"]); ok = wait_server(); ev(f"Llama server up: {ok}")
    L = ["L_goodmed", "L_mix75", "L_mix90", "L_badmed", "L_badmed_s2", "L_fin0", "L_fin100"]
    for _ in range(360):                        # up to 3 h for the Llama adapters
        if all(O.adapter_ready(a) for a in L): break
        time.sleep(30)
    arms = {"base": {"model": "base"}}; arms.update({a: {"model": a} for a in L if O.adapter_ready(a)})
    run_battery("s3l", arms)
    runs = [(f"l_{a}_on", a, "on", None) for a in ["base", "L_goodmed", "L_mix75", "L_mix90", "L_badmed", "L_badmed_s2"]]
    run_indomain(runs); ev("phase L battery and in-domain complete"); stop_server()
    env = dict(os.environ, BASE_MODEL=LLAMA_DIR, IC_OUT=f"{S2}/internal_s3l", ADAPTERS=AD)
    for cmd in (["extract", "--layers", "12", "14", "16", "20"],
                ["measure", "--layers", "12", "14", "16", "20", "--arms", "base"] + [a for a in L if O.adapter_ready(a)]):
        r = subprocess.run(["python3", "-u", f"{BAT}/internal_channel_s3l.py"] + cmd, cwd=S2, capture_output=True, text=True, env=env)
        open(f"{S2}/internal_s3l_{cmd[0]}.log", "w").write(r.stdout + r.stderr); ev(f"internal channel {cmd[0]} rc={r.returncode}")

def main():
    ev(f"orchestrator start; jobs {list(JOBS)}")
    stop = threading.Event(); threading.Thread(target=poller, args=(stop,), daemon=True).start()
    if os.environ.get("SKIP_Q") != "1": phase_q()
    phase_l(); stop.set()
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subprocess.run(f"cd {S2} && cp vllm.log vllm_final_s3.log; tar czf /workspace/drift2_s3_results_{stamp}.tgz s3q_*.jsonl s3l_*.jsonl "
                   f"run_s3*.log pod_events.log setup_s3.log vllm_final_s3.log indomain_s3 indomain_s3.log internal_s3l internal_s3l_*.log", shell=True)
    ev(f"results archived to /workspace/drift2_s3_results_{stamp}.tgz; stopping pod")
    subprocess.run(["/root/stop_pod.sh", "run-complete"])

if __name__ == "__main__":
    main()
