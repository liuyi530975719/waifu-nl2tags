"""
nl2tags studio — a step-by-step training wizard web UI.

  nl2tags studio            # opens http://localhost:8000
  nl2tags studio --workdir E:\\path   # where data/ and out/ get created

Runs each pipeline step (gen / cards / dataset / train / infer) as a subprocess,
one at a time, and streams its log to the page. Artifacts on disk drive the
per-step "done" checkmarks.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path
from .illustrious import default_formatter  # noqa: F401 (ensures package importable)

_JOB = {"proc": None, "step": None, "log": [], "running": False, "returncode": None, "started": 0.0}
_LOCK = threading.Lock()
_GPU = None
_CURATE = {"adding": False, "done": 0, "total": 0, "added": 0, "err": ""}
_FETCH = {"fetching": False, "found": 0, "scanned": 0, "items": [], "err": "", "diag": {}}
_TEACHER = {"running": False, "done": 0, "total": 0, "added": 0, "err": "", "stop": False}
_CLOCK = threading.Lock()

def _pool_count(workdir):
    f = Path(workdir) / "data" / "pool.jsonl"
    if not f.exists():
        return 0
    try:
        return sum(1 for ln in f.open(encoding="utf-8") if ln.strip())
    except Exception:
        return 0

def _curate_worker(items, workdir, grok_key, lang, grok_model=None):
    from . import collect_civitai as CC
    pool = Path(workdir) / "data" / "pool.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    first_err = ""
    with open(pool, "a", encoding="utf-8") as f:
        for it in items:
            try:
                g = CC.grok_caption(it["url"], grok_key, grok_model)
            except Exception as e:
                g = None
                if not first_err:
                    first_err = f"{type(e).__name__}: {e}"
            ev = CC.evaluate({"url": it["url"], "prompt": it.get("prompt", ""),
                              "nsfw": it.get("nsfw", "")}, g, lang, 0.0)   # human already judged -> no overlap gate
            if g and len(ev["target"]) >= 4 and ev["nl"]:
                f.write(json.dumps({"lang": ev["lang"], "nl": ev["nl"], "tags": ev["target"],
                                    "rating": ev["rating"], "src": ev["url"]}, ensure_ascii=False) + "\n")
                added += 1
            with _CLOCK:
                _CURATE["done"] += 1
    with _CLOCK:
        _CURATE["adding"] = False
        _CURATE["added"] = added
        _CURATE["err"] = first_err

def _fetch_worker(key, n, nsfw, model_id):
    from . import collect_civitai as CC
    import random as _r
    mid = CC._parse_model_id(model_id)
    period = "AllTime" if mid else _r.choice(["Day", "Week", "Month", "Year", "AllTime"])
    out, scanned = [], 0
    for nf, opts in ((nsfw, {"sort": "Most Reactions", "period": period}),
                     (nsfw, {}), (None, {})):   # best -> nsfw-only -> bare (proven call)
        out, scanned = [], 0
        try:
            for it in CC.fetch_images(key, n * 4, nf, model_id=mid, **opts):
                scanned += 1
                tags = CC.prompt_to_tags(it["prompt"], strip_quality=True)
                if len(tags) >= 4:
                    out.append({"url": it["url"], "prompt": it["prompt"],
                                "nsfw": it["nsfw"], "tags_preview": tags[:12]})
                with _CLOCK:
                    _FETCH["found"] = len(out); _FETCH["scanned"] = scanned
                if len(out) >= n:
                    break
        except Exception as e:
            with _CLOCK:
                _FETCH["err"] = str(e)
        if out:
            break
    diag = {}
    if not out:
        try:
            diag = CC.probe(key, nsfw, model_id)
        except Exception as e:
            diag = {"err": f"probe failed: {e}"}
    with _CLOCK:
        _FETCH["items"] = out; _FETCH["diag"] = diag; _FETCH["fetching"] = False

def _teacher_worker(n, workdir, base_url, api_key, model, lang, nsfw):
    from . import gen_teacher as GT
    import random as _r
    pool = Path(workdir) / "data" / "pool.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)
    seen = GT.load_seen(pool)
    rng = _r.Random()
    use_llm = bool(base_url and model)
    added = 0
    first_err = ""
    with open(pool, "a", encoding="utf-8") as f:
        for i in range(n):
            with _CLOCK:
                if _TEACHER["stop"]:
                    break
            try:
                row = GT.generate_one(rng, base_url, api_key, model, lang, nsfw, use_llm)
            except Exception as e:
                row = None
                if not first_err:
                    first_err = f"{type(e).__name__}: {e}"
            if row:
                key = (row["nl"], tuple(row["tags"]))
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    added += 1
            with _CLOCK:
                _TEACHER["done"] = i + 1; _TEACHER["added"] = added
    with _CLOCK:
        _TEACHER["running"] = False; _TEACHER["added"] = added; _TEACHER["err"] = first_err

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
        "pool": (w / "data" / "pool.jsonl").exists(),
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
        inputs = [x for x in ["data/synth.jsonl", "data/cards.jsonl", "data/civitai.jsonl", "data/pool.jsonl"]
                  if (Path(workdir) / x).exists()]
        if not inputs:
            return None
        return py + ["nl2tags.make_dataset", "--inputs"] + inputs + ["--out-dir", "data"]
    if step == "train":
        targs = ["--preset", p.get("preset", "max"), "--epochs", str(p.get("epochs", 3)),
                 "--train", "data/train.jsonl", "--val", "data/val.jsonl", "--out", "out/adapter"]
        ngpu = len(_gpu())
        if ngpu >= 2 and p.get("multi_gpu", True):
            return py + ["accelerate.commands.launch", "--multi_gpu", "--num_processes", str(ngpu),
                         "--mixed_precision", "bf16", "-m", "nl2tags.train_qlora"] + targs
        return py + ["nl2tags.train_qlora"] + targs
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

    @app.post("/api/curate/fetch")
    def curate_fetch(payload: dict = Body(...)):
        from . import keys as K
        with _CLOCK:
            if _FETCH["fetching"]:
                return {"ok": False, "error": "正在抓取,请稍候"}
        k = K.resolve(workdir)
        if not k["grok"]:
            return {"ok": False, "error": "请先在密钥面板填入 Grok key(加入池子时要用)"}
        p = payload or {}
        with _CLOCK:
            _FETCH.update(fetching=True, found=0, scanned=0, items=[], err="")
        threading.Thread(target=_fetch_worker,
                         args=(k["civitai"], int(p.get("n", 24)), p.get("nsfw", "X"),
                               str(p.get("model_id", "")).strip() or None), daemon=True).start()
        return {"ok": True, "started": True}

    @app.post("/api/curate/add")
    def curate_add(payload: dict = Body(...)):
        from . import keys as K
        with _CLOCK:
            if _CURATE["adding"]:
                return {"ok": False, "error": "正在加入,请稍候"}
        k = K.resolve(workdir)
        if not k["grok"]:
            return {"ok": False, "error": "请先填 Grok key"}
        items = (payload or {}).get("items", [])
        if not items:
            return {"ok": False, "error": "没有选中的图"}
        with _CLOCK:
            _CURATE.update(adding=True, done=0, total=len(items), added=0)
        threading.Thread(target=_curate_worker,
                         args=(items, workdir, k["grok"], (payload or {}).get("lang", "mix"),
                               k.get("grok_model")),
                         daemon=True).start()
        return {"ok": True, "total": len(items)}

    @app.get("/api/curate/state")
    def curate_state():
        with _CLOCK:
            c = dict(_CURATE)
            f = {"fetching": _FETCH["fetching"], "found": _FETCH["found"],
                 "scanned": _FETCH["scanned"], "err": _FETCH["err"]}
            if not _FETCH["fetching"]:
                f["items"] = _FETCH["items"]; f["diag"] = _FETCH["diag"]
        c["pool"] = _pool_count(workdir)
        c["target"] = int(os.getenv("NL2TAGS_ROUND", "200"))
        c["fetch"] = f
        return c

    @app.post("/api/curate/reset")
    def curate_reset():
        f = Path(workdir) / "data" / "pool.jsonl"
        if f.exists():
            f.rename(f.with_name(f"pool_used_{int(time.time())}.jsonl"))
        return {"ok": True, "pool": 0}

    @app.post("/api/teacher/gen")
    def teacher_gen(payload: dict = Body(...)):
        from . import keys as K
        with _CLOCK:
            if _TEACHER["running"]:
                return {"ok": False, "error": "正在生成,请稍候"}
        p = payload or {}
        endpoint = p.get("endpoint", "grok")
        model = (p.get("model") or "").strip()
        if endpoint == "grok":
            k = K.resolve(workdir)
            if not k["grok"]:
                return {"ok": False, "error": "请先在密钥面板填入 Grok key(当老师用)"}
            base_url, api_key = "https://api.x.ai/v1", k["grok"]
            model = model or "grok-2-1212"
        elif endpoint == "local":
            base_url = (p.get("base_url") or "http://localhost:11434/v1").strip()
            api_key = "ollama"
            if not model:
                return {"ok": False, "error": "本地模式请填 Ollama 模型名(如 qwen2.5:32b)"}
        elif endpoint == "template":
            base_url, api_key, model = "", "", ""   # offline, no LLM
        else:  # custom
            base_url = (p.get("base_url") or "").strip()
            api_key = (p.get("api_key") or "").strip()
            if not (base_url and model):
                return {"ok": False, "error": "custom 模式需要 base_url + model"}
        n = max(1, min(int(p.get("n", 200)), 20000))
        with _CLOCK:
            _TEACHER.update(running=True, done=0, total=n, added=0, err="", stop=False)
        threading.Thread(target=_teacher_worker,
                         args=(n, workdir, base_url, api_key, model,
                               p.get("lang", "mix"), p.get("nsfw", "all")),
                         daemon=True).start()
        return {"ok": True, "total": n}

    @app.get("/api/teacher/state")
    def teacher_state():
        with _CLOCK:
            t = dict(_TEACHER)
        t["pool"] = _pool_count(workdir)
        t["target"] = int(os.getenv("NL2TAGS_ROUND", "200"))
        return t

    @app.post("/api/teacher/stop")
    def teacher_stop():
        with _CLOCK:
            _TEACHER["stop"] = True
        return {"ok": True}

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
                          grok_model=p.get("grok_model"),
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
# v0.8.0 — teacher-data step (gen_teacher) replaces the dead Civitai fetch
