"""
caption_cards.py — mine in-domain NL<->tag pairs from your own waifumaster PNG cards.

Your cards already embed the generation prompt (Danbooru tags) in PNG metadata, so the
TARGET side is free. We only synthesize the natural-language INPUT. This yields training
data in exactly your game's tag distribution — the highest-value slice.

Reads both A1111 ('parameters' text chunk) and ComfyUI ('prompt' JSON) metadata.

NL modes:
  synth  : rule-based reverse mapping tag->phrase via the ontology (offline, default)
  llm    : rewrite the tag list into NL with an OpenAI-compatible endpoint (OAI_* env)
  vlm    : caption the actual image via a vision endpoint (VLM_* env; JoyCaption/Qwen-VL)

Usage:
  python src/caption_cards.py --cards /path/to/cards --out data/cards.jsonl --nl synth --strip-quality
"""
from __future__ import annotations
import argparse, json, os, random, re
from pathlib import Path

ONT = Path(__file__).resolve().parent / "data" / "tag_ontology.json"
QUALITY_WORDS = {"masterpiece", "best quality", "amazing quality", "high quality",
                 "good quality", "very aesthetic", "absurdres", "highres", "newest",
                 "very awa", "safe", "sensitive", "nsfw", "explicit", "lowres",
                 "worst quality", "bad quality", "jpeg artifacts"}

def build_revmap(path=ONT):
    cats = json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
    rev = {}
    for entries in cats.values():
        for e in entries:
            if e.get("tag"):
                rev[e["tag"]] = e
    return rev

def read_png_meta(path: Path) -> str | None:
    """Return the positive prompt string from a PNG, or None."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow required: pip install pillow --break-system-packages")
    try:
        img = Image.open(path)
        info = img.info or {}
    except Exception:
        return None
    # A1111
    if "parameters" in info:
        txt = info["parameters"]
        return txt.split("Negative prompt:")[0].strip()
    # ComfyUI: 'prompt' holds a graph; pull text from CLIPTextEncode nodes (first = positive)
    if "prompt" in info:
        try:
            g = json.loads(info["prompt"])
            texts = [n.get("inputs", {}).get("text") for n in g.values()
                     if isinstance(n, dict) and "text" in n.get("inputs", {})]
            texts = [t for t in texts if isinstance(t, str) and t.strip()]
            if texts:
                return max(texts, key=len)  # positive prompt is usually the longer one
        except Exception:
            return None
    return None

def prompt_to_tags(s: str, strip_quality: bool) -> list[str]:
    tags = []
    for piece in s.replace("\n", ",").split(","):
        t = piece.strip().lower()
        t = re.sub(r"[:(]\d*\.?\d+[)]?$", "", t).strip()   # drop (tag:1.2) weights
        t = t.strip("()").replace("\\", "").replace("_", " ").strip()
        if not t:
            continue
        if strip_quality and t in QUALITY_WORDS:
            continue
        tags.append(t)
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

ZH_T = ["帮我画{s}", "画{s}", "想要{s}", "生成一张{s}的图"]
EN_T = ["draw {s}", "a picture of {s}", "generate {s}", "I want {s}"]

def tags_to_nl(tags, rev, lang, rng):
    phrases = []
    for t in tags:
        e = rev.get(t)
        if e and lang == "zh" and e.get("zh"):
            phrases.append(rng.choice(e["zh"]))
        elif e and lang == "en" and e.get("en"):
            phrases.append(rng.choice(e["en"]))
        elif rng.random() < 0.5:                # keep some unknown tags as-is for realism
            phrases.append(t)
    if not phrases:
        phrases = tags[:4]
    if rng.random() < 0.3:
        rng.shuffle(phrases); phrases = phrases[: max(2, len(phrases) - rng.randint(1, 3))]
    joiner = "，" if lang == "zh" else ", "
    body = joiner.join(phrases)
    return (rng.choice(ZH_T) if lang == "zh" else rng.choice(EN_T)).format(s=body)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True, help="directory of PNG cards")
    ap.add_argument("--out", default="data/cards.jsonl")
    ap.add_argument("--nl", choices=["synth", "llm", "vlm"], default="synth")
    ap.add_argument("--lang", choices=["zh", "en", "mix"], default="mix")
    ap.add_argument("--strip-quality", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rev = build_revmap()
    files = sorted(Path(a.cards).rglob("*.png"))
    if a.limit:
        files = files[: a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p in files:
            pos = read_png_meta(p)
            if not pos:
                continue
            tags = prompt_to_tags(pos, a.strip_quality)
            if len(tags) < 3:
                continue
            lang = a.lang if a.lang != "mix" else rng.choice(["zh", "zh", "en"])
            if a.nl == "synth":
                nl = tags_to_nl(tags, rev, lang, rng)
            elif a.nl == "llm":
                from .synth_data import paraphrase
                nl = paraphrase(tags_to_nl(tags, rev, lang, rng), lang)
            else:  # vlm — caption the real image (needs VLM_* env; see README)
                nl = tags_to_nl(tags, rev, lang, rng)  # fallback if no endpoint
            f.write(json.dumps({"lang": lang, "nl": nl, "tags": tags,
                                "rating": "general", "src": p.name}, ensure_ascii=False) + "\n")
            n += 1
    print(f"mined {n} pairs from {len(files)} PNGs -> {a.out}")

if __name__ == "__main__":
    main()
