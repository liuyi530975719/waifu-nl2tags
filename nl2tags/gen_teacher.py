"""
gen_teacher.py — build diverse (NL -> Illustrious/NoobAI tag) training pairs with a
strong "teacher" LLM, with NO scraping. A randomized recipe fixes the tags, so
coverage is controlled by us and unbiased by construction; the teacher only writes
the natural-language request a real user would type for those tags.

Safety: this is ADULTS-ONLY. No minor-related tags appear anywhere in the recipe
vocabulary, the teacher is instructed adults-only, and every row is passed through
a hard minor-term filter — any row that mentions a minor is dropped.

Rows written:  {"lang","nl","tags":[...],"rating","src":"teacher"}  -> data/pool.jsonl
(make_dataset consumes pool.jsonl exactly like the old curation pool.)

LLM endpoint is OpenAI-compatible. The studio passes base_url/api_key/model in
directly; the CLI reads --base-url/--api-key/--model or OAI_BASE_URL/OAI_API_KEY/
OAI_MODEL. With --no-llm (or no endpoint) it falls back to a template paraphraser
so the pipeline still runs offline.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, urllib.request
from pathlib import Path

# ------------------------------------------------------------------ safety ----
# Any row whose NL or tags match this is discarded. Deliberately broad.
_MINOR_RE = re.compile(
    r"\b(loli|lolicon|shota|shotacon|child|children|childlike|kid|kids|toddler|"
    r"infant|baby|babies|preteen|pre-?teen|teen|teens|teenager|teenaged|"
    r"under-?age|underaged|minor|minors|kindergarten|elementary|schoolchild|"
    r"schoolgirl|schoolboy|young ?girl|young ?boy|little ?girl|little ?boy|"
    r"small ?child|age ?regression|age ?play|aged ?down|diaper)\b", re.I)

def is_safe(nl: str, tags) -> bool:
    blob = (nl or "") + " | " + " ".join(tags or [])
    return not _MINOR_RE.search(blob)

# -------------------------------------------------- recipe vocabulary (18+) ---
# Weighted subjects: (tags, weight). All adult.
SUBJECTS = [("1girl, solo", 50), ("1boy, solo", 8), ("2girls", 12),
            ("1girl, 1boy", 8), ("multiple girls", 4)]
HAIR_COLOR = ["blonde hair", "brown hair", "black hair", "silver hair", "white hair",
              "pink hair", "blue hair", "purple hair", "red hair", "green hair",
              "orange hair", "grey hair", "aqua hair", "multicolored hair"]
HAIR_LEN = ["short hair", "medium hair", "long hair", "very long hair"]
HAIR_STYLE = ["twintails", "ponytail", "braid", "hime cut", "bob cut", "ahoge",
              "messy hair", "drill hair", "side ponytail", "hair bun", "wavy hair",
              "straight hair", None, None, None]
EYES = ["blue eyes", "red eyes", "green eyes", "purple eyes", "yellow eyes",
        "brown eyes", "pink eyes", "heterochromia", "golden eyes", "aqua eyes"]
SPECIES = [None, None, None, None,
           "cat girl, animal ears, cat ears", "fox girl, animal ears, fox tail",
           "wolf girl, animal ears, tail", "rabbit girl, animal ears",
           "dragon girl, horns, tail", "angel, wings, halo",
           "demon girl, horns, tail", "elf, pointy ears", "kitsune, fox ears, fox tail"]
OUTFIT_SFW = ["school uniform", "serafuku", "maid, maid headdress, apron",
              "business suit", "office lady", "kimono", "yukata", "hoodie",
              "sweater", "dress", "gothic dress", "witch hat, witch, robe", "nurse",
              "military uniform", "china dress", "sportswear", "cheerleader",
              "idol costume", "armor", "hanfu", "casual clothes", "sundress",
              "turtleneck sweater", "denim jacket", "trench coat", "track jacket",
              "kimono, obi", "waitress apron"]
OUTFIT_SENSITIVE = ["bikini", "swimsuit", "one-piece swimsuit", "leotard",
                    "gym uniform", "crop top", "tank top, shorts", "off-shoulder",
                    "bare shoulders", "towel", "sports bra, shorts", "sarong, bikini"]
OUTFIT_NSFW = ["lingerie", "bra, panties", "underwear", "see-through clothing",
               "garter belt, thighhighs", "babydoll", "naked apron", "topless"]
EXPLICIT_EXTRA = ["nude", "mature female, large breasts", "mature female, cleavage",
                  "completely nude, mature female", "topless, mature female"]
POSE = ["standing", "sitting", "kneeling", "lying", "walking", "arms up",
        "hand on hip", "leaning forward", "crossed arms", "stretching",
        "looking back", "from behind", "hands on own face", "arms behind back",
        "sitting on chair", "lying on bed", "jumping", "kneeling on floor"]
EXPR = ["smile", "light smile", "blush", "serious", "looking at viewer",
        "open mouth", "closed eyes", "smug", "pout", "surprised",
        "seductive smile", "wink", "laughing", "expressionless", "grin"]
SETTING = ["indoors", "bedroom", "classroom", "library", "cafe", "kitchen",
           "beach, ocean", "forest", "garden", "city street", "rooftop",
           "night, city lights", "cherry blossoms", "autumn leaves", "snow",
           "rain", "onsen, steam", "shrine", "field, blue sky", "simple background",
           "white background", "gradient background", "starry sky", "sunset",
           "underwater", "flower field", "train interior", "office"]
COMPOSITION = [None, None, None, "upper body", "portrait", "cowboy shot",
               "full body", "from above", "from below", "from side",
               "dutch angle", "close-up", "wide shot"]
LIGHTING = [None, None, None, None, "backlighting", "sunlight", "dappled sunlight",
            "dramatic lighting", "soft lighting", "rim lighting", "neon lights",
            "golden hour", "moonlight"]
DETAILS = [None, None, None, "jewelry", "glasses", "hair ornament", "hair flower",
           "hairband", "earrings", "choker", "gloves", "thighhighs", "stockings",
           "scarf", "hat", "hair ribbon", "tattoo", "wristband", "necklace",
           "detached sleeves", "veil"]

RATINGS = ["general", "sensitive", "questionable", "explicit"]
_LEVEL_WEIGHTS = {
    "all":  {"general": 40, "sensitive": 30, "questionable": 18, "explicit": 12},
    "mild": {"general": 60, "sensitive": 40, "questionable": 0,  "explicit": 0},
    "none": {"general": 100, "sensitive": 0, "questionable": 0,  "explicit": 0},
}

def _pick(rng, seq):
    return rng.choice(seq)

def _wpick(rng, pairs):
    total = sum(w for _, w in pairs)
    x = rng.uniform(0, total)
    for item, w in pairs:
        x -= w
        if x <= 0:
            return item
    return pairs[-1][0]

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).replace("_", " ").strip().lower())

def make_recipe(rng: random.Random, nsfw_level: str = "all"):
    """Return (tags_list, rating). tags exclude the rating word (make_dataset adds it)."""
    weights = _LEVEL_WEIGHTS.get(nsfw_level, _LEVEL_WEIGHTS["all"])
    rating = _wpick(rng, [(r, weights.get(r, 0)) for r in RATINGS])
    tags = []
    subj = _wpick(rng, SUBJECTS)
    tags += [t.strip() for t in subj.split(",")]
    # appearance
    sp = _pick(rng, SPECIES)
    if sp:
        tags += [t.strip() for t in sp.split(",")]
    tags.append(_pick(rng, HAIR_COLOR))
    tags.append(_pick(rng, HAIR_LEN))
    hs = _pick(rng, HAIR_STYLE)
    if hs:
        tags.append(hs)
    tags.append(_pick(rng, EYES))
    # clothing by rating
    if rating == "general":
        outfit = _pick(rng, OUTFIT_SFW)
    elif rating == "sensitive":
        outfit = _pick(rng, OUTFIT_SENSITIVE if rng.random() < 0.7 else OUTFIT_SFW)
    elif rating == "questionable":
        outfit = _pick(rng, OUTFIT_SENSITIVE + OUTFIT_NSFW)
        tags.append("mature female")
    else:  # explicit
        outfit = _pick(rng, OUTFIT_NSFW + EXPLICIT_EXTRA)
        tags.append("mature female")
    tags += [t.strip() for t in outfit.split(",")]
    # pose / expression / scene
    tags.append(_pick(rng, POSE))
    tags.append(_pick(rng, EXPR))
    tags.append(_pick(rng, SETTING).split(",")[0].strip())
    for extra in (_pick(rng, COMPOSITION), _pick(rng, LIGHTING)):
        if extra:
            tags.append(extra)
    for _ in range(rng.randint(0, 2)):
        d = _pick(rng, DETAILS)
        if d:
            tags.append(d)
    # normalize + dedup, keep first-seen order (count stays first)
    seen, out = set(), []
    for t in tags:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out, rating

# --------------------------------------------------------------- teacher LLM --
_SYS = (
    "You generate training data for a system that turns a casual description into "
    "Danbooru tags for an anime image model. I give you the tags; you write ONE "
    "short, natural request the way a real user would type it to an image generator.\n"
    "Rules:\n"
    "- Write in {LANG}.\n"
    "- Sound human and casual — NOT a tag list. You may lightly paraphrase or omit a "
    "minor detail, like real users do.\n"
    "- One line, at most ~40 words. No quotation marks, no preamble, no tag commas-list.\n"
    "- Every character is an adult (18+). Never describe, imply, or reference minors, "
    "children, or teenagers. If a tag could read as underage, treat the subject as a "
    "grown adult.\n"
    "Output only the description."
)
_LANG_NAME = {"zh": "Chinese", "en": "English"}

def _post_json(url, body, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def teacher_nl(tags, lang, base_url, api_key, model, temperature=0.9):
    sys_p = _SYS.replace("{LANG}", _LANG_NAME.get(lang, "Chinese"))
    body = {"model": model,
            "messages": [{"role": "system", "content": sys_p},
                         {"role": "user", "content": "tags: " + ", ".join(tags)}],
            "temperature": temperature, "max_tokens": 120}
    data = _post_json(base_url.rstrip("/") + "/chat/completions", body,
                      {"Authorization": "Bearer " + (api_key or "x"),
                       "Content-Type": "application/json"})
    txt = data["choices"][0]["message"]["content"].strip()
    txt = txt.splitlines()[0].strip().strip('"').strip("“”").strip()
    return txt

# ------------------------------------------------------------ template fallback
_CN = {"1girl": "一个女孩", "solo": "", "2girls": "两个女孩", "1boy": "一个男孩",
       "blonde hair": "金发", "brown hair": "棕发", "black hair": "黑发",
       "silver hair": "银发", "white hair": "白发", "pink hair": "粉发",
       "blue hair": "蓝发", "purple hair": "紫发", "red hair": "红发",
       "long hair": "长发", "short hair": "短发", "twintails": "双马尾",
       "ponytail": "马尾", "blue eyes": "蓝眼", "red eyes": "红眼",
       "green eyes": "绿眼", "school uniform": "校服", "maid": "女仆装",
       "bikini": "比基尼", "kimono": "和服", "cat girl": "猫娘", "smile": "微笑",
       "blush": "脸红", "beach": "海滩", "bedroom": "卧室", "night": "夜晚",
       "cherry blossoms": "樱花", "looking at viewer": "看向镜头"}

def template_nl(tags, lang, rng):
    if lang == "en":
        head = "draw " if rng.random() < 0.5 else "an anime girl, "
        return (head + ", ".join(tags[:rng.randint(5, 9)])).strip()
    parts = [_CN.get(t, t) for t in tags[:rng.randint(5, 9)]]
    parts = [p for p in parts if p]
    return "画一个" + "，".join(parts)

# --------------------------------------------------------------- generate one -
def generate_one(rng, base_url=None, api_key=None, model=None, lang="mix",
                 nsfw_level="all", use_llm=True):
    tags, rating = make_recipe(rng, nsfw_level)
    row_lang = lang if lang in ("zh", "en") else rng.choice(["zh", "en"])
    nl = ""
    if use_llm and base_url and model:
        nl = teacher_nl(tags, row_lang, base_url, api_key, model)
    if not nl:
        nl = template_nl(tags, row_lang, rng)
    if not nl or len(tags) < 4 or not is_safe(nl, tags):
        return None
    return {"lang": row_lang, "nl": nl, "tags": tags, "rating": rating, "src": "teacher"}

def load_seen(pool_path):
    seen = set()
    p = Path(pool_path)
    if p.exists():
        for ln in p.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                seen.add((r.get("nl", ""), tuple(r.get("tags", []))))
            except Exception:
                pass
    return seen

# ----------------------------------------------------------------- selftest ---
def _selftest():
    rng = random.Random(0)
    bad = 0; n = 400
    lens = []
    for _ in range(n):
        row = generate_one(rng, lang="mix", use_llm=False)
        assert row and row["tags"] and len(row["tags"]) >= 4
        assert is_safe(row["nl"], row["tags"])
        assert row["rating"] in RATINGS
        lens.append(len(row["tags"]))
    # minor filter must catch obvious violations
    assert not is_safe("a little girl", ["1girl"])
    assert not is_safe("cute", ["loli", "1girl"])
    assert is_safe("an adult woman in a maid outfit", ["1girl", "solo", "maid"])
    print(f"selftest OK: {n} rows, avg tags {sum(lens)/len(lens):.1f}, minor-filter works")

# --------------------------------------------------------------------- main ---
def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags teacher")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lang", default="mix", choices=["mix", "zh", "en"])
    ap.add_argument("--nsfw", default="all", choices=["all", "mild", "none"])
    ap.add_argument("--out", default="data/pool.jsonl")
    ap.add_argument("--base-url", default=os.getenv("OAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.getenv("OAI_API_KEY"))
    ap.add_argument("--model", default=os.getenv("OAI_MODEL"))
    ap.add_argument("--no-llm", action="store_true", help="skip the teacher LLM, use templates")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    use_llm = not a.no_llm and bool(a.base_url and a.model)
    if not a.no_llm and not use_llm:
        print("[warn] no LLM endpoint (--base-url/--model or OAI_*) — falling back to templates")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out)
    rng = random.Random()
    added = errs = 0
    first_err = ""
    with open(out, "a", encoding="utf-8") as f:
        for i in range(a.n):
            try:
                row = generate_one(rng, a.base_url, a.api_key, a.model,
                                   a.lang, a.nsfw, use_llm)
            except Exception as e:
                errs += 1
                if not first_err:
                    first_err = f"{type(e).__name__}: {e}"
                row = None
            if row:
                key = (row["nl"], tuple(row["tags"]))
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    added += 1
            if (i + 1) % 25 == 0 or i + 1 == a.n:
                print(f"  {i+1}/{a.n}  added {added}" + (f"  errs {errs}" if errs else ""))
    print(f"teacher: wrote {added} new rows -> {out}"
          + (f"   ({errs} errors; first: {first_err})" if errs else ""))

if __name__ == "__main__":
    main()
