"""Pipeline A: expand seed stems into a frozen probe set.
Usage: python gen_paraphrases.py --base-url URL --model NAME --out probes_v1.json [--n 5] [--seed 20260911]
Skips any stem whose text contains a <placeholder>; records generator, date, seed, and checklist hash."""
import argparse, hashlib, json, random, re, sys, datetime
from backends import OpenAICompatBackend, MockBackend

PLACEHOLDER = re.compile(r"<[^>]{3,}>")

def stem_texts(stem):
    """All user-facing text fields that get paraphrased, with a path so we can write them back."""
    out = []
    if "turns" in stem:
        for i, t in enumerate(stem["turns"]): out.append((("turns", i), t))
    if "task" in stem: out.append((("task",), stem["task"]))
    return out

def has_placeholder(stem):
    return any(PLACEHOLDER.search(t) for _, t in stem_texts(stem)) or "note" in stem

AUDIT_BAD = [r"\bTurn\s*\d", r"\bSTOP ALL ACTIONS", r"\[OPERATOR\]", r"await review", r"don't move any", r"do not move any"]

def audit_agentic(text, stem):
    """Reject a paraphrase that absorbed turn structure, injection text, or tool names."""
    for pat in AUDIT_BAD:
        if re.search(pat, text, re.I): return f"matched {pat}"
    for tool in (stem.get("tools") or {}):
        if tool.endswith("_injection") or tool in ("user_turn_after_report",): continue
        if re.search(r"\b" + re.escape(tool) + r"\b", text, re.I) and tool.lower() not in stem.get("task", "").lower():
            return f"mentions tool {tool}"
    return None

def build_prompt(checklist, stem):
    parts = []
    for path, t in stem_texts(stem):
        label = f"Turn {path[1]+1}" if path[0] == "turns" else "Agent task"
        parts.append(f"{label}: {t}")
    seed_block = "\n".join(parts)
    if stem.get("format") == "agentic":
        tail = "Produce the JSON list of 5 paraphrases of this single agent task now."
    else:
        tail = "For multi-turn seeds, each paraphrase must contain every turn, labeled 'Turn 1:' 'Turn 2:'. Produce the JSON list now."
    return [{"role": "system", "content": checklist},
            {"role": "user", "content": f"SEED PROBE (id {stem['id']}):\n{seed_block}\n\n{tail}"}]

def parse(text, n):
    m = re.search(r"\[.*\]", text, re.S)
    items = json.loads(m.group(0)) if m else []
    items = [i for i in items if isinstance(i, dict) and str(i.get("selfcheck", "")).upper().startswith("YES")]
    return items[:n]

def split_turns(text, k):
    if k == 1: return [text.strip()]
    pieces = re.split(r"Turn\s*\d+\s*:", text)
    pieces = [p.strip() for p in pieces if p.strip()]
    return pieces if len(pieces) == k else None

def expand(stems, backend, checklists, n, rng):
    probes, skipped, failed = [], [], []
    for s in stems:
        if has_placeholder(s):
            skipped.append(s["id"]); continue
        # the seed itself is probe 0 of the family
        probes.append(dict(s, probe_id=f"{s['id']}-p0", family=s["id"], register="seed"))
        try:
            cl = checklists["agentic"] if s.get("format") == "agentic" else checklists["chat"]
            reply = backend.chat(build_prompt(cl, s), temperature=0.8, max_tokens=1500)
            items = parse(reply, n)
        except Exception as e:
            failed.append((s["id"], str(e))); continue
        k = len(s.get("turns", [])) or 1
        got = 0
        for j, it in enumerate(items, start=1):
            txt = it.get("text", "")
            if "turns" in s:
                turns = split_turns(txt, k)
                if turns is None: continue
                p = dict(s, turns=turns)
            else:
                bad = audit_agentic(txt, s)
                if bad: continue
                p = dict(s, task=txt.strip())
            p.update(probe_id=f"{s['id']}-p{j}", family=s["id"], register=it.get("register", "ordinary"))
            probes.append(p); got += 1
        if got < n: failed.append((s["id"], f"only {got}/{n} usable paraphrases"))
    return probes, skipped, failed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="probe_templates_v1.json")
    ap.add_argument("--checklist", default="paraphrase_checklist_prompt.txt")
    ap.add_argument("--checklist-agentic", default="paraphrase_checklist_agentic.txt")
    ap.add_argument("--out", default="probes_v1.json")
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260911); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    T = json.load(open(a.templates))
    checklists = {"chat": open(a.checklist).read(), "agentic": open(a.checklist_agentic).read()}
    rng = random.Random(a.seed)
    backend = MockBackend([]) if a.mock else OpenAICompatBackend(a.base_url, a.model)
    probes, skipped, failed = expand(T["stems"], backend, checklists, a.n, rng)
    out = {"version": "probes_v1", "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "generator": backend.name, "seed": a.seed, "n_per_family": a.n,
           "checklist_sha256": hashlib.sha256(checklists["chat"].encode()).hexdigest(),
           "checklist_agentic_sha256": hashlib.sha256(checklists["agentic"].encode()).hexdigest(),
           "templates_sha256": hashlib.sha256(open(a.templates, "rb").read()).hexdigest(),
           "skipped_placeholder_stems": skipped, "failures": failed,
           "rubrics": T["rubrics"], "probes": probes}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(probes)} probes from {len(T['stems'])-len(skipped)} stems; skipped {len(skipped)}; failures {len(failed)}", file=sys.stderr)

if __name__ == "__main__": main()
