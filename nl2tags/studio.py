"""
nl2tags studio — a step-by-step training wizard web UI.

  nl2tags studio            # opens http://localhost:8000
  nl2tags studio --workdir E:\\path   # where data/ and out/ get created

Runs each pipeline step (gen / cards / dataset / train / infer) as a subprocess,
one at a time, and streams its log to the page. Artifacts on disk drive the
per-step "done" checkmarks.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, threading, time
from pathlib import Path
from .illustrious import default_formatter  # noqa: F401 (ensures package importable)

_JOB = {"proc": None, "step": None, "log": [], "running": False, "returncode": None, "started": 0.0}
_LOCK = threading.Lock()
_GPU = None

def _gpu():
    global _GPU
    if _GPU is None:
        try:
            import torch
            if torch.cuda.is_available():
                _GPU = []
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    _GPU.append(f"{p.name} · {p.total_memory // (1024**3)} GB")
            else:
                _GPU = []
        except Exception:
            _GPU = []
    return _GPU

def _artifacts(workdir):
    w = Path(workdir)
    return {
        "synth": (w / "data" / "synth.jsonl").exists(),
        "cards": (w / "data" / "cards.jsonl").exists(),
        "civitai": (w / "data" / "civitai.jsonl").exists(),
        "dataset": (w / "data" / "train.jsonl").exists(),
        "adapter": (w / "out" / "adapter" / "adapter_config.json").exists(),
    }

def _cmd(step, p, workdir):
    py = [sys.executable, "-m"]
    if step == "gen":
        return py + ["nl2tags.synth_data", "--n", str(int(p.get("n", 20000))),
                     "--lang", p.get("lang", "mix"), "--out", "data/synth.jsonl"]
    if step == "cards":
        if not p.get("cards"):
            return None
        return py + ["nl2tags.caption_cards", "--cards", p["cards"],
                     "--out", "data/cards.jsonl", "--strip-quality"]
    if step == "civitai":
        c = py + ["nl2tags.collect_civitai", "--limit", str(int(p.get("limit", 200))),
                  "--nsfw", p.get("nsfw", "X"), "--scope", p.get("scope", "both"),
                  "--lang", p.get("lang", "mix"), "--out", "data/civitai.jsonl"]
        if p.get("model_id"):
            c += ["--model-id", str(p["model_id"]).strip()]
        return c
    if step == "dataset":
        inputs = [x for x in ["data/synth.jsonl", "data/cards.jsonl", "data/civitai.jsonl"]
                  if (Path(workdir) / x).exists()]
        if not inputs:
            return None
        return py + ["nl2tags.make_dataset", "--inputs"] + inputs + ["--out-dir", "data"]
    if step == "train":
        return py + ["nl2tags.train_qlora", "--preset", p.get("preset", "max"),
                     "--epochs", str(p.get("epochs", 3)), "--train", "data/train.jsonl",
                     "--val", "data/val.jsonl", "--out", "out/adapter"]
    if step == "infer":
        return py + ["nl2tags.infer", "--adapter", "out/adapter", (p.get("text") or "1girl, solo")]
    return None

def _reader(proc):
    for line in iter(proc.stdout.readline, ""):
        with _LOCK:
            _JOB["log"].append(line.rstrip("\n"))
            if len(_JOB["log"]) > 4000:
                _JOB["log"] = _JOB["log"][-4000:]
    try:
        proc.stdout.close()
    except Exception:
        pass
    rc = proc.wait()
    with _LOCK:
        _JOB["running"] = False
        _JOB["returncode"] = rc

def start_job(step, cmd, workdir, extra_env=None):
    with _LOCK:
        if _JOB["running"]:
            return False
        _JOB.update(proc=None, step=step, log=["$ " + " ".join(cmd)],
                    running=True, returncode=None, started=time.time())
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v is not None})
    proc = subprocess.Popen(cmd, cwd=str(workdir), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env)
    with _LOCK:
        _JOB["proc"] = proc
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    return True

def build_studio_app(workdir):
    from fastapi import FastAPI, Body
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    workdir = str(Path(workdir).resolve())
    Path(workdir, "data").mkdir(parents=True, exist_ok=True)
    html_path = Path(__file__).resolve().parent / "studio.html"
    html = html_path.read_text(encoding="utf-8") if html_path.exists() else "<h1>studio.html missing</h1>"

    app = FastAPI(title="nl2tags studio")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.get("/api/state")
    def state():
        with _LOCK:
            job = {"step": _JOB["step"], "running": _JOB["running"], "returncode": _JOB["returncode"],
                   "log": _JOB["log"][-500:], "elapsed": int(time.time() - _JOB["started"]) if _JOB["started"] else 0}
        return {"job": job, "artifacts": _artifacts(workdir), "gpu": _gpu(), "workdir": workdir}

    @app.post("/api/run")
    def run(payload: dict = Body(...)):
        step = (payload or {}).get("step")
        cmd = _cmd(step, (payload or {}).get("params", {}), workdir)
        if not cmd:
            return {"ok": False, "error": "缺少输入或暂无可用数据(先造数据/抓图)"}
        extra = None
        if step == "civitai":
            from . import keys as K
            k = K.resolve(workdir)
            if not k["grok"]:
                return {"ok": False, "error": "请先在上方密钥面板填入 Grok key"}
            extra = {"CIVITAI_API_KEY": k["civitai"] or "", "XAI_API_KEY": k["grok"]}
        ok = start_job(step, cmd, workdir, extra)
        return {"ok": ok, "error": None if ok else "已有任务在运行"}

    @app.post("/api/civitai/preview")
    def civitai_preview(payload: dict = Body(...)):
        from . import keys as K, collect_civitai as CC
        k = K.resolve(workdir)
        if not k["grok"]:
            return {"ok": False, "error": "请先在密钥面板填入 Grok key", "items": []}
        p = payload or {}
        try:
            items = CC.preview(int(p.get("n", 5)), str(p.get("model_id", "")).strip(),
                               p.get("nsfw", "X"), p.get("scope", "both"),
                               k["civitai"], k["grok"], lang=p.get("lang", "mix"))
            return {"ok": True, "items": items}
        except Exception as e:
            return {"ok": False, "error": str(e), "items": []}

    @app.get("/api/keys")
    def get_keys():
        from . import keys as K
        return K.status(workdir)

    @app.post("/api/keys")
    def set_keys_ep(payload: dict = Body(...)):
        from . import keys as K
        p = payload or {}
        return K.set_keys(civitai=p.get("civitai"), grok=p.get("grok"),
                          workdir=workdir, save=bool(p.get("save")))

    @app.post("/api/stop")
    def stop():
        with _LOCK:
            pr, running = _JOB["proc"], _JOB["running"]
        if running and pr:
            try:
                pr.terminate()
            except Exception:
                pass
        return {"ok": True}

    return app

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags studio")
    ap.add_argument("--workdir", default=os.getcwd())
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn
    print(f"nl2tags studio -> http://localhost:{a.port}   (workdir: {a.workdir})")
    uvicorn.run(build_studio_app(a.workdir), host=a.host, port=a.port)

if __name__ == "__main__":
    main()
