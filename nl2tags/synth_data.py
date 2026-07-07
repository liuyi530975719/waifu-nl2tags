"""
synth_data.py — synthesize bilingual (zh/en) NL <-> tag training pairs.

Strategy (the "reverse" trick): we KNOW the tags (sampled from the ontology),
so we only have to write the natural language a user would type. That's easy and
fully controllable, and it directly models the inverse of what the model must learn.

Output JSONL rows: {"lang","nl","tags":[...],"rating"}
  * tags = CORE content tags only (no quality words). illustrious.py appends the
    quality/rating template at inference, so the template stays swappable.

Optional --paraphrase pass rewrites the rule-based NL into more natural user phrasing
via an OpenAI-compatible endpoint (env OAI_BASE_URL / OAI_API_KEY / OAI_MODEL). Falls
back to a no-op if unset, so the bulk generator never needs the network.

Usage:
  python src/synth_data.py --n 5000 --out data/synth.jsonl --lang mix --seed 0
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

ONT = Path(__file__).resolve().parent / "data" / "tag_ontology.json"

def load_ont(path=ONT):
    return json.loads(Path(path).read_text(encoding="utf-8"))["categories"]

def wpick(entries, rng):
    ws = [max(1, e.get("w", 1)) for e in entries]
    return rng.choices(entries, weights=ws, k=1)[0]

def zh_of(e, rng):
    return rng.choice(e["zh"]) if e.get("zh") else e["tag"]

def en_of(e, rng):
    return rng.choice(e["en"]) if e.get("en") else ("a " + e["tag"])

def maybe(rng, p):  # bernoulli
    return rng.random() < p

def build_sample(cats, rng):
    tags, zh_bits, en_bits = [], [], []
    rating = "general"

    # subject: always a real person-count; "solo" is added as a modifier below
    counts = [e for e in cats["count"] if e["tag"] != "solo"]
    subj = wpick(counts, rng)
    tags.append(subj["tag"])
    single = subj["tag"] in ("1girl", "1boy", "1other")
    if single and maybe(rng, 0.7):
        tags.append("solo")
    zh_bits.append(("subject", zh_of(subj, rng)))
    en_bits.append(("subject", en_of(subj, rng)))

    # species (optional)
    if maybe(rng, 0.18):
        sp = wpick(cats["species"], rng)
        tags.append(sp["tag"]); tags += sp.get("extra_tags", [])
        zh_bits.append(("species", zh_of(sp, rng)))
        en_bits.append(("species", en_of(sp, rng)))

    # hair: color (+length +style)
    hc = wpick(cats["hair_color"], rng); hl = wpick(cats["hair_length"], rng)
    tags += [hc["tag"], hl["tag"]]
    hair_zh = zh_of(hc, rng) + zh_of(hl, rng)
    hair_en = en_of(hl, rng).replace("hair", "").strip() + " " + en_of(hc, rng)
    if maybe(rng, 0.5):
        hs = wpick(cats["hair_style"], rng); tags.append(hs["tag"])
        hair_zh += "，" + zh_of(hs, rng); hair_en += " in " + en_of(hs, rng)
    zh_bits.append(("hair", hair_zh)); en_bits.append(("hair", hair_en))

    # eyes
    if maybe(rng, 0.75):
        ey = wpick(cats["eyes"], rng); tags.append(ey["tag"])
        zh_bits.append(("eyes", zh_of(ey, rng))); en_bits.append(("eyes", en_of(ey, rng)))

    # expression (0-2)
    for _ in range(rng.choice([0, 1, 1, 2])):
        ex = wpick(cats["expression"], rng)
        if ex["tag"] not in tags:
            tags.append(ex["tag"])
            zh_bits.append(("expr", zh_of(ex, rng))); en_bits.append(("expr", en_of(ex, rng)))

    # clothing
    if maybe(rng, 0.85):
        cl = wpick(cats["clothing"], rng)
        tags.append(cl["tag"]); tags += cl.get("extra_tags", [])
        if cl.get("rating_hint") == "sensitive" and maybe(rng, 0.6):
            rating = "sensitive"
        zh_bits.append(("cloth", zh_of(cl, rng))); en_bits.append(("cloth", en_of(cl, rng)))

    # pose (+ maybe second)
    po = wpick(cats["pose"], rng); tags.append(po["tag"])
    zh_bits.append(("pose", zh_of(po, rng))); en_bits.append(("pose", en_of(po, rng)))
    if maybe(rng, 0.3):
        po2 = wpick(cats["pose"], rng)
        if po2["tag"] not in tags:
            tags.append(po2["tag"])
            zh_bits.append(("pose", zh_of(po2, rng))); en_bits.append(("pose", en_of(po2, rng)))

    # composition (optional)
    if maybe(rng, 0.4):
        co = wpick(cats["composition"], rng); tags.append(co["tag"])
        zh_bits.append(("comp", zh_of(co, rng))); en_bits.append(("comp", en_of(co, rng)))

    # background
    if maybe(rng, 0.8):
        bg = wpick(cats["background"], rng); tags.append(bg["tag"])
        zh_bits.append(("bg", zh_of(bg, rng))); en_bits.append(("bg", en_of(bg, rng)))

    # lighting/style (optional)
    if maybe(rng, 0.25):
        li = wpick(cats["lighting_style"], rng); tags.append(li["tag"])
        zh_bits.append(("light", zh_of(li, rng))); en_bits.append(("light", en_of(li, rng)))

    # dedupe tags, keep order
    seen, tt = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t); tt.append(t)
    return tt, zh_bits, en_bits, rating

ZH_OPEN = ["帮我画", "画", "生成", "想要", "来一张", "给我画", "一张图：", ""]
ZH_END = ["。", "，谢谢", "，可爱一点", "", "，二次元风格"]
EN_OPEN = ["draw ", "generate ", "I want ", "a picture of ", "make ", "an image of ", ""]
EN_END = [".", ", anime style", ", please", "", ", cute"]

def render_zh(zh_bits, rng):
    parts = [b for _, b in zh_bits]
    if maybe(rng, 0.3):  # drop a couple details -> underspecified input
        rng.shuffle(parts); parts = parts[: max(2, len(parts) - rng.randint(1, 3))]
    return rng.choice(ZH_OPEN) + "，".join(parts) + rng.choice(ZH_END)

def render_en(en_bits, rng):
    subj = next((b for k, b in en_bits if k == "subject"), "a girl")
    rest = [b for k, b in en_bits if k != "subject"]
    if maybe(rng, 0.3):
        rng.shuffle(rest); rest = rest[: max(1, len(rest) - rng.randint(1, 3))]
    s = rng.choice(EN_OPEN) + subj + (" with " + ", ".join(rest) if rest else "")
    return s + rng.choice(EN_END)

def paraphrase(nl, lang):
    """Optional LLM diversify pass. No-op unless OAI_* env vars are set."""
    base = os.getenv("OAI_BASE_URL"); key = os.getenv("OAI_API_KEY")
    if not (base and key):
        return nl
    try:
        import urllib.request
        sys_p = ("把下面这句改写成更口语、更像真实用户输入的中文，只输出改写结果：" if lang == "zh"
                 else "Rewrite the following as a more casual, realistic user prompt. Output only the rewrite:")
        body = json.dumps({"model": os.getenv("OAI_MODEL", "gpt-4o-mini"),
                           "messages": [{"role": "system", "content": sys_p},
                                        {"role": "user", "content": nl}],
                           "temperature": 1.0, "max_tokens": 120}).encode()
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip() or nl
    except Exception:
        return nl

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/synth.jsonl")
    ap.add_argument("--lang", choices=["zh", "en", "mix"], default="mix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paraphrase", action="store_true")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    cats = load_ont()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for _ in range(a.n):
            tags, zh_bits, en_bits, rating = build_sample(cats, rng)
            lang = a.lang if a.lang != "mix" else rng.choice(["zh", "zh", "en"])
            nl = render_zh(zh_bits, rng) if lang == "zh" else render_en(en_bits, rng)
            if a.paraphrase:
                nl = paraphrase(nl, lang)
            f.write(json.dumps({"lang": lang, "nl": nl, "tags": tags, "rating": rating},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {a.out}")

if __name__ == "__main__":
    main()
