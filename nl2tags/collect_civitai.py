"""
collect_civitai.py — build (natural-language -> real-tag) pairs from Civitai's
top-rated images, captioned by Grok vision.

  * pulls high-reaction images (Civitai API) + their REAL prompt (= target tags)
  * Grok vision describes each image -> natural language (= input)
  * keeps a pair only if Grok's observed tags overlap the real prompt (对照 gate)

Keys come ONLY from env (CIVITAI_API_KEY, XAI_API_KEY) — injected by the studio,
never passed on the command line or printed.

  python -m nl2tags.collect_civitai --limit 200 --nsfw X --scope both --out data/civitai.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from .caption_cards import prompt_to_tags

CIVITAI = "https://civitai.com/api/v1/images"
GROK_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/") + "/chat/completions"
GROK_MODEL = os.getenv("GROK_VISION_MODEL", "grok-2-vision-latest")

def _parse_model_id(s):
    """Accept a full Civitai model URL, or a bare id, and return the numeric modelId."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.search(r"/models/(\d+)", s)      # https://civitai.com/models/12345/name[?...]
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    m = re.search(r"(\d{2,})", s)            # fallback: first long number
    return m.group(1) if m else None

def _civitai_page(params, key):
    url = CIVITAI + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "nl2tags"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())

def fetch_images(key, limit, nsfw, model_id=None, sort="Most Reactions", period="AllTime"):
    """Yield {url, prompt, nsfw} for up to `limit` images that have a prompt."""
    got, cursor = 0, None
    while got < limit:
        params = {"limit": min(100, limit - got), "sort": sort, "period": period, "nsfw": nsfw}
        if model_id:
            params["modelId"] = model_id            # note: don't also send modelVersionId (sort bug)
        if cursor:
            params["cursor"] = cursor
        try:
            data = _civitai_page(params, key)
        except Exception as e:
            print(f"  civitai fetch error: {e}", flush=True)
            break
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            meta = it.get("meta") or {}
            prompt = (meta.get("prompt") or "").strip()
            if prompt and it.get("url"):
                yield {"url": it["url"], "prompt": prompt, "nsfw": str(it.get("nsfwLevel", it.get("nsfw", "")))}
                got += 1
                if got >= limit:
                    return
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break

def grok_caption(image_url, key):
    """Return dict {nl_zh, nl_en, tags:[...]} or None."""
    sys_p = ("You look at ONE anime image and output STRICT JSON only:\n"
             '{"nl_zh":"...","nl_en":"...","tags":["...", "..."]}\n'
             "nl_zh / nl_en = a short, natural description a user would TYPE to ask for this "
             "image (Chinese / English), 1-2 sentences, no tag lists.\n"
             "tags = up to 14 booru-style tags for what you actually see. JSON only, no prose.")
    body = json.dumps({
        "model": GROK_MODEL, "temperature": 0.5, "max_tokens": 500,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image as JSON."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    }).encode()
    req = urllib.request.Request(GROK_URL, data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"nl_zh": (d.get("nl_zh") or "").strip(),
                "nl_en": (d.get("nl_en") or "").strip(),
                "tags": [str(t).strip().lower() for t in d.get("tags", []) if str(t).strip()]}
    except Exception:
        return None

_NSFW_RATING = {"None": "general", "Soft": "sensitive", "Mature": "questionable", "X": "explicit"}

def _words(tags):
    return set(re.findall(r"[a-z0-9]+", " ".join(tags).lower()))

def evaluate(item, grok, lang, min_overlap):
    """Full assessment of one (image, caption) pair — used by preview and by main."""
    target = prompt_to_tags(item["prompt"], strip_quality=True)
    gtags = (grok or {}).get("tags", [])
    tw, gw = _words(target), _words(gtags)
    overlap = (len(tw & gw) / len(tw)) if tw else 0.0
    if grok:
        if lang == "zh":
            nl = grok.get("nl_zh", "")
        elif lang == "en":
            nl = grok.get("nl_en", "")
        else:
            nl = grok.get("nl_zh") if random.random() < 0.6 else grok.get("nl_en")
        nl = (nl or grok.get("nl_en") or grok.get("nl_zh") or "").strip()
    else:
        nl = ""
    rlang = "zh" if (grok and nl == (grok.get("nl_zh") or "").strip()) else "en"
    kept = bool(grok) and len(target) >= 4 and overlap >= min_overlap and bool(nl)
    return {"url": item["url"], "nl": nl, "grok_tags": gtags, "target": target,
            "overlap": round(overlap, 2), "kept": kept,
            "rating": _NSFW_RATING.get(item["nsfw"], "general"),
            "lang": lang if lang != "mix" else rlang}

def preview(n, model_id, nsfw, scope, civitai_key, grok_key, lang="mix", min_overlap=0.15):
    """Fetch + caption n images, return per-item assessments. Writes nothing."""
    src = _parse_model_id(model_id) if (scope in ("model", "both") and model_id) else None
    out = []
    for item in fetch_images(civitai_key, n, nsfw, model_id=src):
        try:
            g = grok_caption(item["url"], grok_key)
        except Exception:
            g = None
        out.append(evaluate(item, g, lang, min_overlap))
        if len(out) >= n:
            break
    return out


def probe(key, nsfw, model_id=None, sort="Most Reactions", period="AllTime", limit=20):
    """One raw Civitai call for diagnosing empty results. Never raises."""
    mid = _parse_model_id(model_id)
    params = {"limit": limit, "sort": sort, "period": period, "nsfw": nsfw}
    if mid:
        params["modelId"] = mid
    info = {"url": CIVITAI + "?" + urllib.parse.urlencode(params), "modelId": mid,
            "status": None, "raw": 0, "with_prompt": 0, "err": "", "raw_base": None, "base_err": ""}
    # baseline: does the API work at all (plain call, no filters)?
    try:
        breq = urllib.request.Request(CIVITAI + "?limit=5", headers={"User-Agent": "nl2tags"})
        if key:
            breq.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(breq, timeout=40) as br:
            info["raw_base"] = len(json.loads(br.read()).get("items", []))
    except urllib.error.HTTPError as e:
        info["base_err"] = f"HTTP {e.code}"
    except Exception as e:
        info["base_err"] = f"{type(e).__name__}: {e}"
    try:
        req = urllib.request.Request(info["url"], headers={"User-Agent": "nl2tags"})
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(req, timeout=40) as r:
            info["status"] = getattr(r, "status", 200)
            data = json.loads(r.read())
        items = data.get("items", [])
        info["raw"] = len(items)
        info["with_prompt"] = sum(1 for it in items
                                  if ((it.get("meta") or {}).get("prompt") or "").strip())
    except urllib.error.HTTPError as e:
        info["status"] = e.code
        try:
            info["err"] = f"HTTP {e.code}: " + e.read()[:200].decode("utf-8", "replace")
        except Exception:
            info["err"] = f"HTTP {e.code}"
    except Exception as e:
        info["err"] = f"{type(e).__name__}: {e}"
    return info

def fetch_batch(key, n, nsfw, model_id=None):
    """Fetch a varied batch of images (with usable prompts) for human curation.
    No Grok here — captioning happens only on the ones the user selects."""
    import random as _r
    mid = _parse_model_id(model_id)
    period = "AllTime" if mid else _r.choice(["Day", "Week", "Month", "Year", "AllTime"])
    out = []
    for it in fetch_images(key, n * 3, nsfw, model_id=mid,
                           sort="Most Reactions", period=period):
        tags = prompt_to_tags(it["prompt"], strip_quality=True)
        if len(tags) >= 4:
            out.append({"url": it["url"], "prompt": it["prompt"],
                        "nsfw": it["nsfw"], "tags_preview": tags[:12]})
        if len(out) >= n:
            break
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags collect-civitai")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model-id", default="", help="Civitai modelId (blank = general)")
    ap.add_argument("--nsfw", default="X", help="None|Soft|Mature|X (max level to include)")
    ap.add_argument("--scope", choices=["model", "general", "both"], default="both")
    ap.add_argument("--lang", choices=["zh", "en", "mix"], default="mix")
    ap.add_argument("--min-overlap", type=float, default=0.15)
    ap.add_argument("--out", default="data/civitai.jsonl")
    a = ap.parse_args(argv)

    ci = os.getenv("CIVITAI_API_KEY", "").strip()
    gr = os.getenv("XAI_API_KEY", "").strip()
    if not gr:
        raise SystemExit("Grok key missing (set it in the studio 密钥 panel).")
    if not ci:
        print("  (no Civitai key — public images only, may hit rate limits / no explicit)", flush=True)

    sources = []
    if a.scope in ("model", "both") and a.model_id:
        sources.append(_parse_model_id(a.model_id))
    if a.scope in ("general", "both") or not a.model_id:
        sources.append(None)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept = seen = 0
    per = max(1, a.limit // max(1, len(sources)))
    with open(a.out, "w", encoding="utf-8") as f:
        for src in sources:
            print(f"== source: {'model '+src if src else 'general'} (up to {per}) ==", flush=True)
            for item in fetch_images(ci, per, a.nsfw, model_id=src):
                seen += 1
                try:
                    g = grok_caption(item["url"], gr)
                except Exception as e:
                    print(f"  [{seen}] grok error: {e}", flush=True); continue
                if not g:
                    print(f"  [{seen}] grok: no JSON, skip", flush=True); continue
                ev = evaluate(item, g, a.lang, a.min_overlap)
                if ev["kept"]:
                    f.write(json.dumps({"lang": ev["lang"], "nl": ev["nl"], "tags": ev["target"],
                                        "rating": ev["rating"], "src": ev["url"]}, ensure_ascii=False) + "\n")
                    kept += 1
                    print(f"  [{seen}] kept (overlap {ev['overlap']}) · {len(ev['target'])} tags · {ev['lang']}", flush=True)
                else:
                    print(f"  [{seen}] dropped (对照未过 / 标签太少)", flush=True)
                time.sleep(0.2)
    print(f"done: kept {kept}/{seen} -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
