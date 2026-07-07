"""HTTP endpoint + web UI.

  nl2tags serve --adapter out/adapter     # fine-tuned model + web UI
  nl2tags serve --proxy                    # proxy to local vLLM/Ollama (OAI_* env)

Open http://localhost:8000  (left: natural language, right: prompt).
API:  POST /translate  {"text","rating","template","quality"} -> {"prompt"}
      GET  /info        -> {"mode","model"}
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from .illustrious import Formatter, default_formatter

_HTML = Path(__file__).resolve().parent / "webui.html"
_ONT = Path(__file__).resolve().parent / "data" / "tag_ontology.json"

def build_app(translate_fn, info=None):
    """translate_fn(text:str, opts:dict)->str.  opts: rating|template|quality."""
    from fastapi import FastAPI, Body
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="nl2tags")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    html = _HTML.read_text(encoding="utf-8") if _HTML.exists() else "<h1>webui.html missing</h1>"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/info")
    def _info():
        return info or {"mode": "unknown", "model": ""}

    @app.post("/translate")
    def translate_ep(payload: dict = Body(...)):
        p = payload or {}
        opts = {
            "rating": (p.get("rating") or None),
            "template": (p.get("template") or "illustrious"),
            "quality": bool(p.get("quality", True)),
        }
        return {"prompt": translate_fn(p.get("text", ""), opts)}

    return app

def _formatter_cache():
    cache = {}
    ont = str(_ONT) if _ONT.exists() else None
    def get(tpl):
        if tpl not in cache:
            cache[tpl] = Formatter(template=tpl, ontology_path=ont)
        return cache[tpl]
    return get

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags serve")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--proxy", action="store_true", help="proxy to OAI_* endpoint (vLLM/Ollama/OpenAI)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    get_fmt = _formatter_cache()

    if a.proxy:
        from .baseline import call_llm, postprocess
        info = {"mode": "proxy", "model": os.getenv("OAI_MODEL", "OAI endpoint")}
        def translate_fn(text, opts):
            return postprocess(call_llm(text), get_fmt(opts["template"]),
                               rating=opts["rating"], add_quality=opts["quality"])
    else:
        from .infer import load_model, gen_tags, postprocess
        base = a.base
        if not base:
            from .presets import PRESETS, DEFAULT_PRESET
            base = PRESETS[a.preset or DEFAULT_PRESET]["base"]
        load_model(base, a.adapter)
        info = {"mode": "model", "model": (a.adapter or base)}
        def translate_fn(text, opts):
            return postprocess(gen_tags(text), get_fmt(opts["template"]),
                               rating=opts["rating"], add_quality=opts["quality"])

    import uvicorn
    print(f"nl2tags web UI -> http://localhost:{a.port}")
    uvicorn.run(build_app(translate_fn, info), host=a.host, port=a.port)

if __name__ == "__main__":
    main()
