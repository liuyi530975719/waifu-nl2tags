"""Unified `nl2tags` command."""
from __future__ import annotations
import importlib, sys
from . import presets

HELP = """nl2tags — natural language (中/EN) -> Illustrious/NoobAI Danbooru tags

usage: nl2tags <command> [args]

  quickstart              generate demo data + print the train command
  studio                  step-by-step training wizard (web UI)
  presets                 list model presets (fine-tune + zero-shot)
  doctor                  check GPU / deps and recommend a preset
  gen [--n N --lang mix]  synthesize NL<->tag pairs        -> data/synth.jsonl
  cards --cards DIR       mine pairs from your PNG cards    -> data/cards.jsonl
  dataset --inputs ...    merge -> data/train.jsonl / val.jsonl
  train --preset max      QLoRA fine-tune                   -> out/adapter
  infer --adapter DIR "…" run the fine-tuned model
  baseline "…"            zero-training translate via an OpenAI-compatible LLM
  serve [--adapter DIR|--proxy]   HTTP endpoint for your game backend

examples:
  nl2tags quickstart
  nl2tags train --preset max
  nl2tags infer --adapter out/adapter "银发猫娘女仆，红眼睛，室内"
"""

SUB = {"studio": "studio", "gen": "synth_data", "civitai": "collect_civitai", "cards": "caption_cards", "dataset": "make_dataset",
       "train": "train_qlora", "infer": "infer", "baseline": "baseline", "serve": "serve"}

def doctor():
    print("nl2tags doctor")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch {torch.__version__}   cuda={cuda}")
        total = 0
        for i in range(torch.cuda.device_count() if cuda else 0):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory // (1024 ** 3); total += gb
            print(f"  GPU{i}: {p.name}  {gb} GB")
        if cuda:
            rec = ("max" if total >= 80 else "strong" if total >= 44
                   else "balanced" if total >= 22 else "fast")
            print(f"  total VRAM {total} GB  ->  recommended preset: {rec}")
        else:
            print("  no CUDA GPU visible — training needs one; baseline/zero-shot still work")
    except ImportError:
        print("  torch not installed — run:  pip install 'waifu-nl2tags[train]'")

def quickstart(argv):
    from . import synth_data, make_dataset
    from .presets import PRESETS, DEFAULT_PRESET
    print("[1/2] generating 2000 synthetic pairs -> data/synth.jsonl")
    synth_data.main(["--n", "2000", "--out", "data/synth.jsonl"])
    print("[2/2] building dataset -> data/train.jsonl + val.jsonl")
    make_dataset.main(["--inputs", "data/synth.jsonl", "--out-dir", "data"])
    p = PRESETS[DEFAULT_PRESET]
    print("\nNext:")
    print(f"  add your cards:  nl2tags cards --cards /path/to/cards --strip-quality")
    print(f"  then train:      nl2tags train --preset {DEFAULT_PRESET}   # {p['base']} QLoRA")
    print(f"  or zero-shot:    see README — run Qwen3-32B on vLLM, then  nl2tags serve --proxy")

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "presets":
        return presets.print_table()
    if cmd == "doctor":
        return doctor()
    if cmd == "quickstart":
        return quickstart(rest)
    if cmd not in SUB:
        print(f"unknown command: {cmd}\n"); print(HELP); return
    mod = importlib.import_module(f"nl2tags.{SUB[cmd]}")
    return mod.main(rest)
