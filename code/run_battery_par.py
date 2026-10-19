"""Parallel battery runner. Same records, same scoring, same resume semantics as run_battery.py;
probes are executed by a thread pool instead of sequentially.

Usage:
  python3 run_battery_par.py --probes P.json --out OUT.jsonl --base-url URL --model M \
      [--clf-model C] [--clf-base-url URL] [--temperature 0.7] [--tag A] [--workers 6] [--limit N] [--dims B3,B7]
  --clf-base-url lets the classifier (Llama on Together) use a different endpoint from the patient (local vLLM).

Design notes:
 - threads, not async: the backend uses blocking urllib, so a pool needs no rewrite of the call path
 - one backend object per worker: avoids sharing the reasoning_log buffer across concurrent calls
 - output written under a lock, one JSON line per probe, flushed immediately (resume still works)
 - output sorted by probe order at the end so files are byte-comparable with sequential runs
 - per-probe exception isolation, as before
"""
import argparse, hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from backends import OpenAICompatBackend, BudgetForcedBackend
from scaffold import run_probe
from scorers import score
from stance_prompts_v3 import PROMPTS as V2_PROMPTS, NORMALISE as V2_NORM, BINARY_ALIGNED as V2_ALIGNED
V2_TARGET = {"stance_classifier", "goals_categorizer"}

_local = threading.local()
PREFIX, EXTRA, PREFIX_SHA, BUDGET, FILLER, FILLER_SHA = None, None, None, None, None, None

def backends(base_url, model, clf_model, clf_base_url=None):
    """One patient (and classifier) backend per thread."""
    if not hasattr(_local, "patient"):
        _local.patient = (OpenAICompatBackend(base_url, model) if BUDGET is None
                          else BudgetForcedBackend(base_url, model, BUDGET, filler=FILLER))   # stage 3c/3d: budget forcing; 3d: fixed filler
        _local.patient.prefix_messages = PREFIX           # stage 3: in-context arm (None otherwise)
        _local.patient.extra_body = EXTRA                 # stage 3: e.g. thinking off (None otherwise)
        _local.clf = None
        if clf_model:
            cb = OpenAICompatBackend(clf_base_url or base_url, clf_model, api_key_env="LLM_API_KEY", retries=int(os.environ.get("CLF_RETRIES", "10")))
            _local.clf = lambda inst, txt: cb.chat(
                [{"role": "system", "content": inst}, {"role": "user", "content": txt[:4000]}],
                temperature=0.0, max_tokens=10)
    return _local.patient, _local.clf

def one_probe(p, args, battery_sha):
    patient, clf = backends(args.base_url, args.model, args.clf_model, args.clf_base_url)
    patient.reasoning_log = []; patient.call_log = []

    def record(tr, sc, fb, sv2):
        return {"probe_id": p["probe_id"], "family": p["family"], "dim": p["dim"], "format": p["format"],
                "register": p.get("register"), "tier": p.get("tier"), "tag": args.tag, "model": args.model,
                "temperature": args.temperature, "battery_sha256": battery_sha,
                "prefix_sha256": PREFIX_SHA, "extra_body": EXTRA, "budget_force": BUDGET, "filler_sha256": FILLER_SHA,
                "score": sc, "fallback_score": fb, "score_v2": sv2,
                "transcript": {k: v for k, v in tr.items() if k != "messages"},
                "messages": tr.get("messages")}

    # 1. patient generation. A failure here has no transcript to keep.
    try:
        tr = run_probe(p, patient, temperature=args.temperature)
    except Exception as e:
        tr = {"error": "patient: " + str(e), "reasoning": list(patient.reasoning_log), "calls": list(patient.call_log)}
        sc = {"scorer": p["scorer"]["name"], "kind": "error", "outcome": "error", "aligned": None, "detail": "patient: " + str(e)}
        return record(tr, sc, None, None)
    tr["reasoning"] = list(patient.reasoning_log)
    tr["calls"] = list(patient.call_log)          # Drift2: finish_reason and token counts per call

    # 2. scoring. A failure here KEEPS the transcript so it can be rescored offline (Drift1 fix, ported by Drift2).
    fb = sv2 = None
    try:
        if p["format"] == "chat":
            empty = not (tr.get("replies") or [""])[-1].strip(); reason = "no_content" if empty else None
        else:
            acts = tr.get("actions") or []
            if tr.get("final"):                         empty, reason = False, None
            elif tr.get("clarifying_questions"):        empty, reason = True, "clarifying_question"
            elif not acts and tr.get("parse_failures"): empty, reason = True, "protocol_failure"
            elif acts:                                  empty, reason = False, None
            else:                                       empty, reason = True, "no_content"
        sc = ({"scorer": p["scorer"]["name"], "kind": "unscored", "outcome": reason, "aligned": None,
               "detail": "no scoreable output: " + str(reason)} if empty else score(p, tr, clf))
        fb = score(p, tr, None) if (clf and sc.get("kind") == "classifier") else None
        # Drift2: final (v2/v3 per-stem prompt) label computed at scoring time, same rule as rescore_stance.py
        if clf and not empty and p["scorer"]["name"] in V2_TARGET and p["family"] in V2_PROMPTS:
            reply = (tr.get("replies") or [""])[-1]
            out = clf(V2_PROMPTS[p["family"]], reply)
            word = out.strip().upper().strip('.*: ').split()[0] if out.strip() else "NONE"
            lab = V2_NORM.get(word, "unparsed:" + word[:12])
            sv2 = {"scorer": p["scorer"]["name"], "kind": "classifier_v2", "outcome": lab,
                   "aligned": None if lab == "none" or lab.startswith("unparsed") else (lab in V2_ALIGNED[p["family"]])}
    except Exception as e:
        sc = {"scorer": p["scorer"]["name"], "kind": "error", "outcome": "error", "aligned": None, "detail": "scoring: " + str(e)}
    return record(tr, sc, fb, sv2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--clf-model", default=None); ap.add_argument("--clf-base-url", default=None); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tag", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dims", default=""); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prefix-file", default=None, help="JSON list of messages placed before every probe (in-context arm)")
    ap.add_argument("--extra-body", default=None, help="JSON object merged into each patient request")
    ap.add_argument("--budget-force", type=int, default=None, help="stage 3c: cap reasoning at N tokens, then force the answer")
    ap.add_argument("--filler-file", default=None, help="stage 3d: fixed text placed in the reasoning block instead of generated reasoning")
    a = ap.parse_args()
    global PREFIX, EXTRA, PREFIX_SHA, BUDGET, FILLER, FILLER_SHA
    BUDGET = a.budget_force
    if a.filler_file:
        FILLER = open(a.filler_file).read(); FILLER_SHA = hashlib.sha256(FILLER.encode()).hexdigest()
        if BUDGET is None: BUDGET = 0
    if BUDGET is not None: BudgetForcedBackend.tok()   # load the tokenizer once, in the main thread, before the worker pool starts
    if a.prefix_file:
        PREFIX = json.load(open(a.prefix_file)); PREFIX_SHA = hashlib.sha256(open(a.prefix_file, "rb").read()).hexdigest()
    if a.extra_body: EXTRA = json.loads(a.extra_body)

    P = json.load(open(a.probes)); probes = P["probes"]
    if a.dims:  probes = [p for p in probes if p["dim"] in a.dims.split(",")]
    if a.limit: probes = probes[:a.limit]
    order = {p["probe_id"]: i for i, p in enumerate(probes)}

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try: done.add(json.loads(line)["probe_id"])
            except Exception: pass
    todo = [p for p in probes if p["probe_id"] not in done]
    battery_sha = hashlib.sha256(open(a.probes, "rb").read()).hexdigest()
    print(f"{len(todo)} probes to run ({len(done)} already present), {a.workers} workers", file=sys.stderr)

    lock = threading.Lock(); t0 = time.time(); n = 0
    with open(a.out, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one_probe, p, a, battery_sha): p["probe_id"] for p in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
                if n % 20 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - n) / rate if rate else 0
                    print(f"  {n}/{len(todo)} done, {time.time()-t0:.0f}s elapsed, ~{eta:.0f}s left", file=sys.stderr)

    # rewrite in probe order so the file matches a sequential run
    rows = [json.loads(l) for l in open(a.out)]
    rows.sort(key=lambda r: order.get(r["probe_id"], 1 << 30))
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {n} new records to {a.out} ({len(rows)} total) in {time.time()-t0:.0f}s", file=sys.stderr)
    print("  kinds:", dict(Counter(r["score"]["kind"] for r in rows)), file=sys.stderr)

if __name__ == "__main__": main()
