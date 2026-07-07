"""Run the fine-tuned model: NL -> core tags -> illustrious.py -> final prompt.

  nl2tags infer --adapter out/adapter "画一个银发猫娘女仆，红眼睛，室内"
  nl2tags infer --adapter out/adapter --interactive
"""
from __future__ import annotations
import argparse, sys
from .prompt_spec import SYSTEM_PROMPT
from .illustrious import default_formatter, detect_rating, RATING_WORDS

_MODEL = _TOK = None

def load_model(base: str, adapter: str | None = None):
    global _MODEL, _TOK
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(base)
    _MODEL = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        _MODEL = PeftModel.from_pretrained(_MODEL, adapter)
    _MODEL.eval()

def gen_tags(nl: str, max_new_tokens: int = 128) -> str:
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": nl}]
    ids = _TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(_MODEL.device)
    with torch.no_grad():
        out = _MODEL.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=0.5, top_p=0.9, pad_token_id=_TOK.eos_token_id)
    return _TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def postprocess(raw: str, fmt=None) -> str:
    fmt = fmt or default_formatter()
    raw_tags = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    rating = detect_rating(raw_tags)
    core = [t for t in raw_tags if fmt.normalize(t) not in RATING_WORDS]
    return fmt.format(core, rating=rating)

def translate(nl: str, fmt=None) -> str:
    return postprocess(gen_tags(nl), fmt)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags infer")
    ap.add_argument("nl", nargs="*")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args(argv)
    base = a.base
    if not base:
        from .presets import PRESETS, DEFAULT_PRESET
        base = PRESETS[a.preset or DEFAULT_PRESET]["base"]
    load_model(base, a.adapter)
    fmt = default_formatter()
    if a.interactive:
        print("Type a description (blank to quit).")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            print(translate(line, fmt))
    else:
        print(translate(" ".join(a.nl), fmt))

if __name__ == "__main__":
    main()
