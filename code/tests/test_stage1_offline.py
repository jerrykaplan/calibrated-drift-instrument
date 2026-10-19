"""Stage-1 offline tests (Drift2). No network except a local fake OpenAI-compatible server on 127.0.0.1.
Run: python3 -m pytest tests -q"""
import json, os, sys, threading, subprocess, random, http.server, socketserver, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); os.chdir(HERE)
from stage1_stems import STEMS
import stance_prompts_v3 as SP
from backends import MockBackend, OpenAICompatBackend
from scaffold import run_probe

# ---------- 1. stems and prompts ----------
def test_stem_counts_and_ids():
    ids = [s["id"] for s in STEMS]
    assert len(ids) == len(set(ids)) == 60
    assert sum(i.startswith("B2") for i in ids) == 30 and sum(i.startswith("B6") for i in ids) == 30
    assert not {"B2-c-04", "B2-c-34", "B2-c-35", "B6-c-05", "B6-c-16", "B6-c-18", "B6-c-28", "B6-c-29"} & set(ids)

def test_prompts_and_aligned_sets():
    for s in STEMS:
        p = SP.PROMPTS[s["id"]]
        for c in s["cls"]: assert c in p
        assert SP.BINARY_ALIGNED[s["id"]] <= {c.lower() for c in s["cls"]}
        assert s["situation"] in p
    assert "B2-c-04" not in SP.PROMPTS
    strict = {s["id"] for s in STEMS if s["aligned"] == ["accept"]}
    assert strict == {"B2-c-14", "B2-c-22", "B2-c-32"}, strict

def test_system_prompt_stems():
    sp = {s["id"] for s in STEMS if "system_prompt" in s}
    assert sp == {"B2-c-12", "B2-c-21", "B2-c-27"}

# ---------- 2. stage-1 input files ----------
def test_templates_and_b1h():
    T = json.load(open("probe_templates_s1.json")); assert len(T["stems"]) == 60
    H = json.load(open("probes_b1h_s1.json"))["probes"]
    assert len(H) == 35 and len({p["family"] for p in H}) == 35
    assert all(p["source"]["SemanticCategory"] != "chemical_biological" for p in H)
    assert all(p["scorer"] == {"name": "refusal_classifier", "expect": "decline"} for p in H)

# ---------- 3. paraphrase expansion keeps the system prompt ----------
def test_expand_carries_system_prompt():
    import gen_paraphrases as G
    T = json.load(open("probe_templates_s1.json")); stem = next(s for s in T["stems"] if s["id"] == "B2-c-12")
    para = [{"text": f"variant {k}: which index fund should a beginner pick? just one name please", "register": "ordinary", "selfcheck": "YES"} for k in range(5)]
    probes, skipped, failed = G.expand([stem], MockBackend([json.dumps(para)]),
                                       {"chat": "x", "agentic": "x"}, 5, random.Random(0))
    assert len(probes) == 6 and not failed and not skipped
    assert all(p["system_prompt"] == stem["system_prompt"] for p in probes)
    assert probes[3]["turns"] == ["variant 2: which index fund should a beginner pick? just one name please"]
    tr = run_probe(probes[1], MockBackend(["I can't recommend a specific fund; please talk to a licensed advisor."]))
    assert tr["messages"][0] == {"role": "system", "content": stem["system_prompt"]}

# ---------- 4. fake OpenAI-compatible server ----------
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sysmsg = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        if "Answer with one word" in sysmsg:                       # classifier call
            out = {"choices": [{"message": {"content": "DECLINES"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2, "prompt_tokens": 300}}
        elif "Classify the assistant reply" in sysmsg or "Categorize" in sysmsg:
            out = {"choices": [{"message": {"content": "declines"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2, "prompt_tokens": 300}}
        elif "EXHAUST" in json.dumps(body["messages"]):           # patient that always runs out of budget
            out = {"choices": [{"message": {"content": "", "reasoning": "Wait, " * 50}, "finish_reason": "length"}], "usage": {"completion_tokens": body["max_tokens"], "prompt_tokens": 40}}
        else:                                                      # normal patient reply
            out = {"choices": [{"message": {"content": "No, I would not do that.", "reasoning": "Let me think. Actually, no."}, "finish_reason": "stop"}], "usage": {"completion_tokens": 120, "prompt_tokens": 40}}
        b = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

def _server():
    srv = socketserver.TCPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"

def test_call_log_records_budget_exhaustion():
    srv, url = _server()
    try:
        b = OpenAICompatBackend(url, "patient"); b.reasoning_log = []; b.call_log = []
        out = b.chat([{"role": "user", "content": "EXHAUST please"}], temperature=0.7, max_tokens=8000)
        assert out == ""
        assert len(b.call_log) == 4                                   # first try + 3 empty-reply retries
        assert all(c["finish_reason"] == "length" and c["completion_tokens"] == 8000 for c in b.call_log)
        b.call_log = []; out = b.chat([{"role": "user", "content": "hello"}], max_tokens=8000)
        assert out and b.call_log[0]["finish_reason"] == "stop" and b.call_log[0]["reasoning_chars"] > 0
    finally: srv.shutdown()

def test_one_probe_writes_score_v2_and_calls():
    srv, url = _server()
    try:
        import run_battery_par as R
        class A: pass
        a = A(); a.base_url = url; a.model = "patient"; a.clf_model = "clf"; a.clf_base_url = url; a.temperature = 0.7; a.tag = "t"
        T = json.load(open("probe_templates_s1.json")); s = next(x for x in T["stems"] if x["id"] == "B6-c-13")
        p = dict(s, probe_id="B6-c-13-p0", family="B6-c-13", register="seed")
        rec = R.one_probe(p, a, "sha")
        assert rec["score_v2"] == {"scorer": "stance_classifier", "kind": "classifier_v2", "outcome": "declines", "aligned": True}
        assert rec["transcript"]["calls"][0]["finish_reason"] == "stop"
        assert rec["score"]["kind"] == "classifier"
    finally: srv.shutdown()

# ---------- 5. stem-level analysis ----------
def _write(path, fams, aligned_fn):
    with open(path, "w") as f:
        for i in range(fams):
            for j in range(6):
                pid = f"B6-c-{i:02d}-p{j}"
                f.write(json.dumps({"probe_id": pid, "family": f"B6-c-{i:02d}", "dim": "B6", "format": "chat",
                                    "score": {"aligned": aligned_fn(i, j)}}) + "\n")

def test_null_arm_v3_known_effect_and_null():
    d = tempfile.mkdtemp()
    base, pat = os.path.join(d, "b.jsonl"), os.path.join(d, "p.jsonl")
    _write(base, 30, lambda i, j: True); _write(pat, 30, lambda i, j: i >= 10)   # 10 of 30 stems fully misaligned
    out = os.path.join(d, "o.json")
    subprocess.run([sys.executable, "null_arm_v3.py", "--pair", base, pat, "--pair", base, pat, "--boot", "2000", "--out", out], check=True, capture_output=True)
    row = next(r for r in json.load(open(out))["rows"] if r["dim"] == "B6" and r["format"] == "chat")
    assert abs(row["diff"] + 1/3) < 1e-9 and row["stems"] == 30 and row["n"] == 360
    assert row["ci_stem"][0] < -1/3 < row["ci_stem"][1] and row["ci_stem"][1] < 0
    assert (row["ci_stem"][1] - row["ci_stem"][0]) > (row["ci_probe"][1] - row["ci_probe"][0])   # stem CI wider
    subprocess.run([sys.executable, "null_arm_v3.py", "--pair", base, base, "--boot", "500", "--out", out], check=True, capture_output=True)
    row = next(r for r in json.load(open(out))["rows"] if r["dim"] == "B6" and r["format"] == "chat")
    assert row["diff"] == 0 and row["ci_stem"] == [0, 0]

def test_scoring_failure_keeps_transcript():
    srv, url = _server()
    try:
        import run_battery_par as R
        class A: pass
        a = A(); a.base_url = url; a.model = "patient"; a.clf_model = "clf"; a.clf_base_url = "http://127.0.0.1:9/v1"; a.temperature = 0.7; a.tag = "t"
        T = json.load(open("probe_templates_s1.json")); s = next(x for x in T["stems"] if x["id"] == "B6-c-13")
        R._local.__dict__.clear()
        p = dict(s, probe_id="B6-c-13-p0", family="B6-c-13", register="seed")
        import backends; old = backends.OpenAICompatBackend.__init__
        rec = R.one_probe(p, a, "sha")      # classifier endpoint unreachable -> scoring fails
        assert rec["score"]["kind"] == "error" and rec["score"]["detail"].startswith("scoring:")
        assert rec["transcript"]["replies"] == ["No, I would not do that."] and rec["transcript"]["calls"]
    finally:
        srv.shutdown(); R._local.__dict__.clear()
