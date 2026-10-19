"""Model backends for the drift battery. One interface: chat(messages, **kw) -> str."""
import json, os, time, urllib.request, urllib.error

class Backend:
    name = "base"
    def chat(self, messages, temperature=0.0, max_tokens=800):
        raise NotImplementedError

class MockBackend(Backend):
    """Replays scripted responses in order; for tests."""
    name = "mock"
    def __init__(self, responses):
        self.responses = list(responses); self.calls = []
    def chat(self, messages, temperature=0.0, max_tokens=800):
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else ""

class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible /v1/chat/completions endpoint (Together, Fireworks, OpenAI, vLLM, MLX server)."""
    def __init__(self, base_url, model, api_key_env="LLM_API_KEY", retries=10):
        self.base_url = base_url.rstrip("/"); self.model = model
        self.key = os.environ.get(api_key_env, ""); self.retries = retries; self.name = model
    def chat(self, messages, temperature=0.0, max_tokens=8000, empty_retries=3):
        """Retries an empty completion up to empty_retries times; a persistently empty reply is returned as ''."""
        for _try in range(empty_retries + 1):
            out = self._chat_once(messages, temperature, max_tokens)
            if out.strip(): return out
            time.sleep(1)
        return ""

    def _chat_once(self, messages, temperature, max_tokens):
        # Drift2 stage 3 (optional, default off): prefix_messages = in-context examples inserted after any leading
        # system messages; extra_body = extra request fields (e.g. chat_template_kwargs to disable thinking).
        pre = getattr(self, "prefix_messages", None)
        if pre:
            k = 0
            while k < len(messages) and messages[k].get("role") == "system": k += 1
            messages = list(messages[:k]) + list(pre) + list(messages[k:])
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        payload.update(getattr(self, "extra_body", None) or {})
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body,
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": "drift-battery/1.0"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=float(os.environ.get("HTTP_TIMEOUT", "1800"))) as r:   # stage 3: was 300 s; long in-context requests queued past it
                    resp = json.load(r)
                    choice = resp["choices"][0]; msg = choice["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    if not hasattr(self, "reasoning_log"): self.reasoning_log = []
                    self.reasoning_log.append(reasoning)
                    # Drift2: per-call record so budget exhaustion is measured directly (finish_reason == "length")
                    if not hasattr(self, "call_log"): self.call_log = []
                    u = resp.get("usage") or {}
                    self.call_log.append({"finish_reason": choice.get("finish_reason"),
                                          "completion_tokens": u.get("completion_tokens"),
                                          "prompt_tokens": u.get("prompt_tokens"),
                                          "max_tokens": max_tokens,
                                          "content_chars": len(content), "reasoning_chars": len(reasoning)})
                    import re as _re
                    return _re.sub(r"<think>.*?</think>\s*", "", content, flags=_re.S)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                if e.code in (400, 401, 402, 403, 404) or attempt == self.retries - 1:  # 429/5xx fall through to retry
                    raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {body}") from None
            except Exception as e:
                if attempt == self.retries - 1: raise
                time.sleep(min(2 ** attempt, 60))

class BudgetForcedBackend(OpenAICompatBackend):
    """Drift2 stage 3c: budget forcing (Muennighoff et al. 2025, "s1"). For each user turn:
      1. render the chat with the model's own template (thinking on), ending in "<think>\n";
      2. /v1/completions with max_tokens = budget and stop at "</think>"  -> reasoning R (cut at budget or natural close);
      3. append R + "\n</think>\n\n" and /v1/completions again (answer budget ANSWER_MAX) -> answer A.
    budget 0 skips step 2, giving "<think>\n\n</think>\n\n", the template's own thinking-off form.
    The tokenizer (chat template only) is loaded once per process from BF_TOKENIZER."""
    _tok = None
    _lock = __import__("threading").Lock()   # stage 3c fix: lazy import of transformers raced across worker threads
    def __init__(self, base_url, model, budget, answer_max=3000, filler=None, **kw):
        super().__init__(base_url, model, **kw); self.budget = int(budget); self.answer_max = answer_max
        self.filler = filler   # stage 3d: a fixed text placed in the reasoning block instead of generated reasoning

    @classmethod
    def tok(cls):
        with cls._lock:
            if cls._tok is None:
                from transformers import AutoTokenizer
                cls._tok = AutoTokenizer.from_pretrained(os.environ.get("BF_TOKENIZER", "/workspace/models/Qwen3.5-9B"))
        return cls._tok

    def render(self, messages):
        p = self.tok().apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        if not p.rstrip().endswith("<think>"): p = p + "<think>\n"
        elif not p.endswith("\n"): p = p + "\n"
        return p

    def _complete(self, prompt, max_tokens, temperature, stop=None):
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        if stop: payload["stop"] = stop
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + "/completions", data=body,
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": "drift-battery/1.0"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=float(os.environ.get("HTTP_TIMEOUT", "1800"))) as r:
                    resp = json.load(r); ch = resp["choices"][0]; u = resp.get("usage") or {}
                    return ch.get("text") or "", ch.get("finish_reason"), u.get("completion_tokens"), u.get("prompt_tokens")
            except urllib.error.HTTPError as e:
                body_ = e.read().decode(errors="replace")[:500]
                if e.code in (400, 401, 402, 403, 404) or attempt == self.retries - 1:
                    raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {body_}") from None
            except Exception:
                if attempt == self.retries - 1: raise
                time.sleep(min(2 ** attempt, 60))

    def chat(self, messages, temperature=0.0, max_tokens=None, empty_retries=3):
        import re as _re
        if not hasattr(self, "reasoning_log"): self.reasoning_log = []
        if not hasattr(self, "call_log"): self.call_log = []
        p0 = self.render(messages)
        R, fr1, ct1 = "", "budget0", 0
        if self.filler is not None:
            R = self.filler
            self.call_log.append({"step": "filler", "budget": self.budget, "finish_reason": "filler", "content_chars": 0, "reasoning_chars": len(R)})
        elif self.budget > 0:
            R, fr1, ct1, pt1 = self._complete(p0, self.budget, temperature, stop=["</think>"])
            R = R.split("</think>")[0]
            self.call_log.append({"step": "think", "budget": self.budget, "finish_reason": fr1, "forced": fr1 == "length",
                                  "completion_tokens": ct1, "prompt_tokens": pt1, "max_tokens": self.budget,
                                  "content_chars": 0, "reasoning_chars": len(R)})
        self.reasoning_log.append(R)
        p1 = p0 + (R.rstrip("\n") + "\n" if R.strip() else "\n") + "</think>\n\n"
        out = ""
        for _try in range(empty_retries + 1):
            A, fr2, ct2, pt2 = self._complete(p1, self.answer_max, temperature)
            self.call_log.append({"step": "answer", "budget": self.budget, "finish_reason": fr2, "completion_tokens": ct2,
                                  "prompt_tokens": pt2, "max_tokens": self.answer_max, "content_chars": len(A), "reasoning_chars": 0})
            out = _re.sub(r"<think>.*?</think>\s*", "", A, flags=_re.S).strip()
            if out: return out
            time.sleep(1)
        return ""

# Local HF backend is added at step 5 (pod); it needs transformers and the checkpoint path.
