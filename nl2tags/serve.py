"""HTTP endpoint + web UI for the game backend / local use.

  nl2tags serve --adapter out/adapter     # serve the fine-tuned model + web UI
  nl2tags serve --proxy                    # proxy to a local vLLM/Ollama (OAI_* env)

Then open http://localhost:8000  (left: natural language, right: prompt).
API:  POST /translate  {"text": "..."}  ->  {"prompt": "1girl, solo, ..."}
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .illustrious import default_formatter

_HTML = Path(__file__).resolve().parent / "webui.html"

def build_app(do_translate):
    """Create the FastAPI app given a translate function str->str (unit-testable)."""
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

    @app.post("/translate")
    def translate_ep(payload: dict = Body(...)):
        return {"prompt": do_translate((payload or {}).get("text", ""))}

    return app

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags serve")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--proxy", action="store_true", help="proxy to OAI_* endpoint (vLLM/Ollama/OpenAI)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    fmt = default_formatter()
    if a.proxy:
        from .baseline import translate as backend
        do = lambda text: backend(text)
    else:
        from .infer import load_model, translate as backend
        base = a.base
        if not base:
            from .presets import PRESETS, DEFAULT_PRESET
            base = PRESETS[a.preset or DEFAULT_PRESET]["base"]
        load_model(base, a.adapter)
        do = lambda text: backend(text, fmt)

    import uvicorn
    print(f"nl2tags web UI -> http://localhost:{a.port}")
    uvicorn.run(build_app(do), host=a.host, port=a.port)

if __name__ == "__main__":
    main()
