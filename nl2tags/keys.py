"""
Key resolver — never prints, logs, or commits secret values.

Order:  in-memory (set from UI) -> env vars -> gitignored keys.local.json
        -> optional chat SQLite DB (grok only) via NL2TAGS_CHAT_DB.
status() returns booleans only.
"""
from __future__ import annotations
import json, os, sqlite3
from pathlib import Path

_MEM = {"civitai": "", "grok": "", "grok_model": ""}

def _from_chat_db(db_path):
    try:
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT grok_api_key FROM chat_config WHERE id=1").fetchone()
        con.close()
        return (row[0] or "").strip() if row else ""
    except Exception:
        return ""

def resolve(workdir="."):
    ci = _MEM["civitai"] or os.getenv("CIVITAI_API_KEY", "").strip()
    gr = _MEM["grok"] or os.getenv("XAI_API_KEY", "").strip() or os.getenv("GROK_API_KEY", "").strip()
    f = Path(workdir) / "keys.local.json"
    if (not ci or not gr) and f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            ci = ci or (d.get("civitai") or "").strip()
            gr = gr or (d.get("grok") or "").strip()
        except Exception:
            pass
    if not gr:
        dbp = os.getenv("NL2TAGS_CHAT_DB", "").strip()
        if dbp and Path(dbp).exists():
            gr = _from_chat_db(dbp)
    gm = _MEM["grok_model"] or os.getenv("GROK_VISION_MODEL", "").strip() or "grok-2-vision-1212"
    return {"civitai": ci, "grok": gr, "grok_model": gm}

def set_keys(civitai=None, grok=None, grok_model=None, workdir=".", save=False):
    if civitai is not None:
        _MEM["civitai"] = civitai.strip()
    if grok is not None:
        _MEM["grok"] = grok.strip()
    if grok_model is not None:
        _MEM["grok_model"] = grok_model.strip()
    if save:
        k = resolve(workdir)
        (Path(workdir) / "keys.local.json").write_text(
            json.dumps({"civitai": k["civitai"], "grok": k["grok"]}), encoding="utf-8")
    return status(workdir)

def status(workdir="."):
    k = resolve(workdir)
    src = "env" if os.getenv("CIVITAI_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") else \
          ("ui" if (_MEM["civitai"] or _MEM["grok"]) else
           ("file" if (Path(workdir) / "keys.local.json").exists() else "none"))
    return {"civitai": bool(k["civitai"]), "grok": bool(k["grok"]),
            "grok_model": k["grok_model"], "source": src}
