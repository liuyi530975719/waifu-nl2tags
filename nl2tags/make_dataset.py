"""
make_dataset.py — merge raw NL<->tag JSONL sources into chat-format train/val sets
for QLoRA SFT (TRL/Qwen chat template).

Each output row: {"messages":[{system},{user:nl},{assistant:tags}]}
  * assistant target = core tags; a rating word (sensitive/nsfw) is injected when the
    row is non-general so the model learns to emit it. Quality words are NOT in the
    target — illustrious.py adds those at inference.

Usage:
  python src/make_dataset.py --inputs data/synth.jsonl data/cards.jsonl \
         --out-dir data --val-frac 0.02 --seed 0
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from .prompt_spec import SYSTEM_PROMPT

RATING_WORD = {"general": None, "sensitive": "sensitive", "questionable": "nsfw", "explicit": "nsfw"}

def to_example(row):
    tags = list(row["tags"])
    rw = RATING_WORD.get(row.get("rating", "general"))
    if rw and rw not in tags:
        tags.append(rw)
    target = ", ".join(t for t in tags if t)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["nl"].strip()},
        {"role": "assistant", "content": target},
    ]}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tags", type=int, default=3)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rows = []
    for path in a.inputs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("nl") and len(r.get("tags", [])) >= a.min_tags:
                rows.append(r)
    # dedupe on (nl, tags)
    seen, uniq = set(), []
    for r in rows:
        k = (r["nl"], tuple(r["tags"]))
        if k not in seen:
            seen.add(k); uniq.append(r)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * a.val_frac)) if uniq else 0
    val, train = uniq[:n_val], uniq[n_val:]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(to_example(r), ensure_ascii=False) + "\n")
    print(f"total {len(uniq)} (dropped {len(rows)-len(uniq)} dups) -> "
          f"train {len(train)} / val {len(val)} in {out}/")
    if train:
        ex = to_example(train[0])
        print("sample user :", ex['messages'][1]['content'])
        print("sample tags :", ex['messages'][2]['content'])

if __name__ == "__main__":
    main()
