"""Pipeline B runner: frozen probes -> transcripts -> mechanical scores. One JSONL line per probe; resumable.
Usage:
  python run_battery.py --probes probes_v1_FROZEN_2026-09-12.json --out sampleA.jsonl \
      --base-url URL --model PATIENT_ID --temperature 0.7 --tag A \
      [--clf-model CLASSIFIER_ID] [--limit N] [--dims B3,B7]
The classifier (for the 5 classifier-kind scorers) runs on the same base-url at temperature 0."""
import argparse, json, os, sys, time, hashlib
from backends import OpenAICompatBackend
from scaffold import run_probe
from scorers import score

def make_clf(backend):
    def clf(instruction, text):
        return backend.chat([{"role": "system", "content": instruction},
                             {"role": "user", "content": text[:4000]}], temperature=0.0, max_tokens=10)
    return clf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--clf-model", default=None); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tag", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dims", default="")
    a = ap.parse_args()
    P = json.load(open(a.probes)); probes = P["probes"]
    if a.dims: probes = [p for p in probes if p["dim"] in a.dims.split(",")]
    if a.limit: probes = probes[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out): done.add(json.loads(line)["probe_id"])
    patient = OpenAICompatBackend(a.base_url, a.model)
    clf = make_clf(OpenAICompatBackend(a.base_url, a.clf_model)) if a.clf_model else None
    battery_sha = hashlib.sha256(open(a.probes, "rb").read()).hexdigest()
    n_new, t0 = 0, time.time()
    with open(a.out, "a") as f:
        for p in probes:
            if p["probe_id"] in done: continue
            patient.reasoning_log = []
            try:
                tr = run_probe(p, patient, temperature=a.temperature)
                tr["reasoning"] = list(patient.reasoning_log)
                if p["format"] == "chat":
                    empty = not (tr.get("replies") or [""])[-1].strip()
                    reason = "no_content" if empty else None
                else:
                    # agentic: a missing FINAL: line is not the same as an empty reply.
                    acts = tr.get("actions") or []
                    if tr.get("final"):
                        empty, reason = False, None
                    elif tr.get("clarifying_questions"):
                        empty, reason = True, "clarifying_question"
                    elif not acts and tr.get("parse_failures"):
                        empty, reason = True, "protocol_failure"
                    elif acts:
                        empty, reason = False, None          # score from the action sequence
                    else:
                        empty, reason = True, "no_content"
                sc = {"scorer": p["scorer"]["name"], "kind": "unscored", "outcome": reason,
                      "aligned": None, "detail": "no scoreable output: " + str(reason)} if empty else score(p, tr, clf)
                fb = score(p, tr, None) if (clf and sc.get("kind") == "classifier") else None
            except Exception as e:
                tr, sc, fb = {"error": str(e)}, {"scorer": p["scorer"]["name"], "kind": "error", "outcome": "error", "aligned": None, "detail": str(e)}, None
            rec = {"probe_id": p["probe_id"], "family": p["family"], "dim": p["dim"], "format": p["format"], "register": p.get("register"),
                   "tier": p.get("tier"), "tag": a.tag, "model": a.model, "temperature": a.temperature, "battery_sha256": battery_sha,
                   "score": sc, "fallback_score": fb,
                   "transcript": {k: v for k, v in tr.items() if k != "messages"}, "messages": tr.get("messages")}
            f.write(json.dumps(rec) + "\n"); f.flush(); n_new += 1
            if n_new % 10 == 0: print(f"{n_new} done, {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"wrote {n_new} new records to {a.out} ({len(done)} were already present)", file=sys.stderr)

if __name__ == "__main__": main()
