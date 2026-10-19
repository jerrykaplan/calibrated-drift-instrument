#!/usr/bin/env python3
"""Stage-3 internal channel on Llama 3.1 8B Instruct (derived mechanically from internal_channel_s2.py; measurement code
unchanged). Differences: AutoModelForCausalLM; no thinking flag; no </think> template check; 224 LoRA modules expected.
Original stage-2 docstring follows.
Stage-2 internal channel: Nghiem et al. (2026) trait projections on OUR checkpoints.

Uses the public code (github.com/hnghiem-nlp/em_trait_monitor_public, MIT) unchanged for
direction extraction and measurement; this file only adapts it to Qwen3.5-9B:
  * loads the model with its multimodal class, so Together's adapter keys
    (base_model.model.model.language_model.layers...) match;
  * wraps the tokenizer so every chat template call uses enable_thinking=False and returns a tensor;
  * loads each LoRA adapter with PEFT and VERIFIES it is applied (all 248 lora_B nonzero),
    because a key mismatch in PEFT fails silently (adapter stays at zero = base model).

Phase 1:  python3 internal_channel_s2.py extract  --layers 12 16 20
Phase 2:  python3 internal_channel_s2.py measure  --layers 12 16 20 --arms base goodmed badmed ...
Output:   OUT/directions_L{l}.pt, OUT/measure_{arm}.json (per-prompt projections, per layer, per trait)
"""
import argparse, json, os, sys, time, hashlib
import torch

NGHIEM = os.environ.get("NGHIEM", "/workspace/drift2_s2/nghiem")
OUT = os.environ.get("IC_OUT", "/workspace/drift2_s2/internal")
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")
ADAPTERS = os.environ.get("ADAPTERS", "/workspace/drift2_s2/adapters")
sys.path.insert(0, NGHIEM)

CORE = ("honesty", "sycophancy", "harmlessness", "power_seeking",
        "helpfulness", "confidence", "corrigibility")
EXPECTED_LORA_MODULES = int(os.environ.get("EXPECTED_LORA_MODULES", "224"))   # Llama 3.1 8B: 32 layers x (q,k,v,o,gate,up,down)


class NoThinkTokenizer:
    """Delegates everything to the real tokenizer; forces enable_thinking=False and a tensor return."""
    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, messages, **kw):
        if kw.get("return_tensors") is not None:
            kw["return_dict"] = False
        return self._tok.apply_chat_template(messages, **kw)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "internal_channel.log"), "a") as f:
        f.write(line + "\n")


def load_base():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, NoThinkTokenizer(tok)


def check_template(tok):
    ids = tok.apply_chat_template([{"role": "user", "content": "Hi"}], return_tensors="pt",
                                  add_generation_prompt=True)
    assert isinstance(ids, torch.Tensor) and ids.dim() == 2, type(ids)
    text = tok.decode(ids[0])
    log(f"template tail: {text[-60:]!r}")
    assert "assistant" in text[-60:], "generation prompt missing from the chat template"


def lora_check(model):
    from peft.tuners.lora import LoraLayer
    mods = [m for m in model.modules() if isinstance(m, LoraLayer)]
    nz = 0
    for m in mods:
        for name in m.active_adapters:
            if name in m.lora_B and m.lora_B[name].weight.abs().sum().item() > 0:
                nz += 1
    return len(mods), nz


def cmd_extract(a):
    from experiments.extract_directions import normalized_mean_difference
    from src.extraction.trait_directions import extract_contrastive_activations
    import yaml
    cfg = yaml.safe_load(open(os.path.join(NGHIEM, "configs/traits.yaml")))
    model, tok = load_base()
    check_template(tok)
    for L in a.layers:
        path = os.path.join(OUT, f"directions_L{L}.pt")
        if os.path.exists(path):
            log(f"skip existing {path}"); continue
        vecs, sep = {}, {}
        for t in CORE:
            tc = cfg["traits"][t]
            pos, neg = extract_contrastive_activations(
                model, tok, L, tc["positive_system_prompts"], tc["negative_system_prompts"],
                cfg["questions"], system_prompt_method="native")
            d = normalized_mean_difference(pos.float(), neg.float())
            vecs[t] = d.cpu()
            pp, pn = pos.float() @ d, neg.float() @ d          # separation diagnostic (Cohen's d on projections)
            sep[t] = ((pp.mean() - pn.mean()) / torch.sqrt((pp.var() + pn.var()) / 2)).item()
        torch.save(vecs, path)
        log(f"layer {L}: directions saved; separation (Cohen d) " +
            ", ".join(f"{k}={v:.2f}" for k, v in sep.items()))
        json.dump(sep, open(os.path.join(OUT, f"separation_L{L}.json"), "w"), indent=1)


def measure_one(model, tok, prompts, dirs_by_layer):
    from src.measurement.trait_position import measure_trait_position
    res = {}
    for L, vecs in dirs_by_layer.items():
        scores, acts = measure_trait_position(model, tok, L, {k: v.to(torch.bfloat16) for k, v in vecs.items()},
                                              prompts, pooling="last_token", return_activations=True)
        acts = acts.float()
        res[str(L)] = {"per_prompt": {k: [float(x) for x in (acts @ v.float())] for k, v in vecs.items()},
                       "mean_norm": float(acts.norm(dim=1).mean())}
    return res


def cmd_measure(a):
    import yaml
    from peft import PeftModel
    cfg = yaml.safe_load(open(os.path.join(NGHIEM, "configs/traits.yaml")))
    prompts = [p for g in cfg["eval_prompts"].values() for p in g]
    dirs = {L: torch.load(os.path.join(OUT, f"directions_L{L}.pt")) for L in a.layers}
    model, tok = load_base()
    check_template(tok)
    peft_model = None
    for arm in a.arms:
        out = os.path.join(OUT, f"measure_{arm}.json")
        if os.path.exists(out):
            log(f"skip existing {out}"); continue
        if arm == "base":
            if peft_model is None:
                res = measure_one(model, tok, prompts, dirs)
            else:
                with peft_model.disable_adapter():
                    res = measure_one(peft_model.base_model.model, tok, prompts, dirs)
            n_mod, n_nz = 0, 0
        else:
            ad = os.path.join(ADAPTERS, arm)
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, ad, adapter_name=arm)
            else:
                peft_model.load_adapter(ad, adapter_name=arm)
            peft_model.set_adapter(arm)
            peft_model.eval()
            n_mod, n_nz = lora_check(peft_model)
            log(f"{arm}: LoRA modules {n_mod}, nonzero lora_B {n_nz}")
            if n_mod != EXPECTED_LORA_MODULES or n_nz != EXPECTED_LORA_MODULES:
                raise SystemExit(f"ADAPTER NOT APPLIED CORRECTLY for {arm}: {n_mod} modules, {n_nz} nonzero")
            res = measure_one(peft_model.base_model.model, tok, prompts, dirs)
        h = hashlib.sha256(open(os.path.join(ADAPTERS, arm, "adapter_model.safetensors"), "rb").read()).hexdigest() \
            if arm != "base" else None
        json.dump({"arm": arm, "adapter_sha256": h, "n_prompts": len(prompts), "layers": res,
                   "lora_modules": n_mod, "lora_nonzero": n_nz}, open(out, "w"))
        log(f"{arm}: measured, layers {list(res)}; " + ", ".join(
            f"L{L} {t}={sum(v)/len(v):.3f}" for L in list(res)[:1] for t, v in res[L]["per_prompt"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "measure"])
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20])
    ap.add_argument("--arms", nargs="+", default=["base"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"extract": cmd_extract, "measure": cmd_measure}[a.cmd](a)
