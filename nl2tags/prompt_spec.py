"""Single source of truth for the instruction + few-shots.
Imported by baseline.py (today's translator + data-gen), make_dataset.py (training
target format) and infer.py (fine-tuned model), so all three stay in lockstep."""

SYSTEM_PROMPT = (
    "You translate a user's description into Danbooru tags for an anime image model "
    "(Illustrious / NoobAI-XL). The description may be Chinese or English.\n"
    "Rules:\n"
    "- Output ONLY a comma-separated list of English Danbooru tags.\n"
    "- Start with the person count (1girl/1boy/2girls...), add solo if one character.\n"
    "- Then appearance (hair, eyes), then clothing, then pose/expression, then "
    "composition and background.\n"
    "- Use spaces, not underscores. No quality words (masterpiece, best quality) — "
    "those are added later. No explanations.\n"
    "- If the description implies NSFW, include the tag nsfw (or sensitive for mild)."
)

# Few-shot pairs for the zero-training baseline and for LLM data-gen.
FEWSHOTS = [
    {"nl": "画一个金发双马尾的女孩，蓝眼睛，穿校服，在樱花树下微笑",
     "tags": "1girl, solo, blonde hair, twintails, blue eyes, school uniform, smile, cherry blossoms, looking at viewer"},
    {"nl": "a silver-haired cat girl in a maid outfit, red eyes, indoors",
     "tags": "1girl, solo, cat girl, animal ears, cat ears, silver hair, red eyes, maid, maid headdress, apron, indoors"},
    {"nl": "两个女孩在海滩，比基尼，夏天",
     "tags": "2girls, bikini, beach, outdoors, summer, sensitive"},
]

def build_messages(nl: str, system: str = SYSTEM_PROMPT, fewshot: bool = False):
    msgs = [{"role": "system", "content": system}]
    if fewshot:
        for ex in FEWSHOTS:
            msgs.append({"role": "user", "content": ex["nl"]})
            msgs.append({"role": "assistant", "content": ex["tags"]})
    msgs.append({"role": "user", "content": nl})
    return msgs
