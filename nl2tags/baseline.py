"""
baseline.py — translate NL -> Illustrious prompt with NO training, using any
OpenAI-compatible chat endpoint + the shared system prompt/few-shots, then the
deterministic illustrious.py post-processor.

Use it three ways:
  1. Ship today as your translator while the fine-tune trains.
  2. As the --paraphrase / --nl llm engine for higher-quality training data.
  3. As the reference the fine-tuned model distills toward.

Env: OAI_BASE_URL (e.g. https://api.openai.com/v1 or your local vLLM), OAI_API_KEY, OAI_MODEL.
Usage: python src/baseline.py "画一个金发双马尾女孩，蓝眼睛，校服，樱花"
"""
from __future__ import annotations
import json, os, sys, urllib.request
from .prompt_spec import build_messages
from .illustrious import default_formatter, detect_rating, RATING_WORDS

def call_llm(nl: str, model=None) -> str:
    base = os.getenv("OAI_BASE_URL"); key = os.getenv("OAI_API_KEY")
    if not (base and key):
        raise RuntimeError("OAI_BASE_URL / OAI_API_KEY 未设置(proxy 模式需要,例如 Ollama)。")
    body = json.dumps({"model": model or os.getenv("OAI_MODEL", "gpt-4o-mini"),
                       "messages": build_messages(nl, fewshot=True),
                       "temperature": 0.4, "max_tokens": 160}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def postprocess(raw: str, fmt=None, rating=None, add_quality=True) -> str:
    fmt = fmt or default_formatter()
    raw_tags = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    r = rating or detect_rating(raw_tags)
    core = [t for t in raw_tags if fmt.normalize(t) not in RATING_WORDS]
    return fmt.format(core, rating=r, add_quality=add_quality)

def translate(nl: str, fmt=None, rating=None, add_quality=True, model=None) -> str:
    return postprocess(call_llm(nl, model), fmt, rating=rating, add_quality=add_quality)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('usage: python src/baseline.py "your description"')
    print(translate(" ".join(sys.argv[1:])))

def list_models():
    """List models from the OpenAI-compatible endpoint (e.g. Ollama /v1/models). [] on failure."""
    base = os.getenv("OAI_BASE_URL"); key = os.getenv("OAI_API_KEY")
    if not (base and key):
        return []
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))
    except Exception:
        return []

def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit('usage: nl2tags baseline "your description"')
    print(translate(" ".join(argv)))
