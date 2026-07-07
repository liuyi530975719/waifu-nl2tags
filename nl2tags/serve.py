"""HTTP endpoint for the game backend.

  nl2tags serve --adapter out/adapter --port 8000     # serve the fine-tuned model
  nl2tags serve --proxy                                # proxy to a local vLLM/Ollama (OAI_* env)

POST /translate  {"text": "..."}  ->  {"prompt": "1girl, solo, ..."}
"""
from __future__ import annotations
import argparse
from .illustrious import default_formatter

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags serve")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--proxy", action="store_true", help="proxy to OAI_* endpoint (vLLM/Ollama/OpenAI)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    fmt = default_formatter()
    if a.proxy:
        from .baseline import translate as backend
        def do(text): return backend(text)
    else:
        from .infer import load_model, translate as backend
        base = a.base
        if not base:
            from .presets import PRESETS, DEFAULT_PRESET
            base = PRESETS[a.preset or DEFAULT_PRESET]["base"]
        load_model(base, a.adapter)
        def do(text): return backend(text, fmt)

    app = FastAPI(title="nl2tags")
    class Req(BaseModel):
        text: str
    @app.get("/health")
    def health(): return {"ok": True}
    @app.post("/translate")
    def translate_ep(r: Req): return {"prompt": do(r.text)}
    uvicorn.run(app, host=a.host, port=a.port)

if __name__ == "__main__":
    main()
