from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TEMPLATES = {
    "illustrious": {
        "quality": ["masterpiece","best quality","amazing quality","very aesthetic","absurdres"],
        "quality_front": [],
        "rating": {"general": [], "sensitive": ["sensitive"], "questionable": ["nsfw"], "explicit": ["nsfw","explicit"]},
    },
    "noobai": {
        "quality": ["masterpiece","best quality","newest","absurdres","highres","very aesthetic"],
        "quality_front": [],
        "rating": {"general": [], "sensitive": ["sensitive"], "questionable": ["nsfw"], "explicit": ["explicit","nsfw"]},
    },
    "pony": {
        "quality": ["score_9","score_8_up","score_7_up","source_anime"],
        "quality_front": ["score_9","score_8_up","score_7_up"],
        "rating": {"general": ["rating_safe"], "sensitive": ["rating_questionable"], "questionable": ["rating_questionable"], "explicit": ["rating_explicit"]},
    },
}
CATEGORY_ORDER = {"count":0,"species":1,"character":2,"copyright":3,"artist":4,"hair_length":10,"hair_style":11,"hair_color":12,"eyes":13,"expression":14,"clothing":20,"pose":30,"composition":31,"background":40,"lighting_style":41,"_general":50}
ALIASES = {"pigtails":"twintails","golden hair":"blonde hair","sakura":"cherry blossoms","catgirl":"cat girl","closeup":"portrait","close-up":"portrait"}
_PAREN_RE = re.compile(r"(?<!\\)([()])"); _WS_RE = re.compile(r"\s+")
_PROTECTED = {"rating_safe","rating_questionable","rating_explicit","rating_general","source_anime","source_pony","source_furry","source_cartoon"}
_SCORE_RE = re.compile(r"^score_\d(_up)?$")
def _is_protected(t): return t in _PROTECTED or bool(_SCORE_RE.match(t))

@dataclass
class Formatter:
    template: str = "illustrious"
    underscores: bool = False
    quality_position: str = "end"
    ontology_path: object = None
    _vocab: set = field(default_factory=set, init=False)
    _cat_of: dict = field(default_factory=dict, init=False)
    def __post_init__(self):
        if self.ontology_path: self._load_ontology(self.ontology_path)
    def _load_ontology(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for cat, entries in data.get("categories", {}).items():
            for e in entries:
                tag = (e.get("tag") or "").strip()
                if not tag: continue
                self._vocab.add(tag); self._cat_of[tag] = cat
                for extra in e.get("extra_tags", []):
                    self._vocab.add(extra); self._cat_of.setdefault(extra, cat)
    def load_extra_vocab(self, tags):
        for t in tags: self._vocab.add(self.normalize(t))
    def normalize(self, tag):
        t = str(tag).strip().lower()
        if _is_protected(t.replace(" ","_")): return t.replace(" ","_")
        t = t.replace("_"," ") if not self.underscores else t.replace(" ","_")
        t = _WS_RE.sub(" " if not self.underscores else "_", t).strip()
        t = ALIASES.get(t, t)
        return _PAREN_RE.sub(r"\\\1", t)
    def category(self, tag): return self._cat_of.get(tag, "_general")
    def format(self, tags, rating="general", add_quality=None):
        add_quality = (self.quality_position != "none") if add_quality is None else add_quality
        tpl = TEMPLATES[self.template]
        seen=set(); clean=[]
        for raw in tags:
            for piece in str(raw).split(","):
                t=self.normalize(piece)
                if t and t not in seen: seen.add(t); clean.append(t)
        clean.sort(key=lambda t: CATEGORY_ORDER.get(self.category(t),50))
        rating_tags = tpl["rating"].get(rating,[]) if rating else []
        quality = tpl["quality"] if add_quality else []
        front = tpl.get("quality_front",[]) if add_quality else []
        if self.quality_position=="front": ordered = front+quality+clean+rating_tags
        else: ordered = front+clean+rating_tags+quality
        out=[]; done=set()
        for t in ordered:
            tn=self.normalize(t)
            if tn and tn not in done: done.add(tn); out.append(tn)
        return ", ".join(out)
    def validate(self, tags):
        known=[]; unknown=[]
        for t in tags:
            tn=self.normalize(t); (known if tn in self._vocab else unknown).append(tn)
        return known, unknown

def default_formatter(ontology_path=None):
    if ontology_path is None:
        ontology_path = Path(__file__).resolve().parent/"data"/"tag_ontology.json"
        if not Path(ontology_path).exists(): ontology_path=None
    return Formatter(template="illustrious", ontology_path=ontology_path)

RATING_WORDS = {"safe", "sensitive", "nsfw", "explicit", "questionable",
                "rating_safe", "rating_sensitive", "rating_questionable", "rating_explicit"}

def detect_rating(tags):
    """Infer a rating bucket from raw model tags (so format() can standardize it)."""
    s = {str(t).strip().lower() for t in tags}
    if s & {"explicit", "rating_explicit"}:
        return "explicit"
    if {"nsfw", "questionable", "rating_questionable"} & s:
        return "questionable"
    if "sensitive" in s:
        return "sensitive"
    return "general"
