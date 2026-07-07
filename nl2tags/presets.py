"""Model presets. Defaults tuned for a dual RTX PRO 6000 box (2x96 GB = 192 GB)."""

PRESETS = {
    "fast":     {"base": "Qwen/Qwen3-1.7B", "mode": "qlora", "vram": "8 GB",   "note": "smoke test"},
    "balanced": {"base": "Qwen/Qwen3-8B",   "mode": "qlora", "vram": "24 GB",  "note": ""},
    "strong":   {"base": "Qwen/Qwen3-14B",  "mode": "qlora", "vram": "48 GB",  "note": ""},
    "max":      {"base": "Qwen/Qwen3-32B",  "mode": "qlora", "vram": "1x96 GB","note": "default for your box"},
    "full-8b":  {"base": "Qwen/Qwen3-8B",   "mode": "full",  "vram": "2x GPU", "note": "full FT: accelerate launch"},
}
DEFAULT_PRESET = "max"

# Zero-shot: run one of these on vLLM, then `nl2tags serve --proxy` (no training).
ZERO_SHOT = {
    "zs-14b": {"model": "Qwen/Qwen3-14B", "tp": 1, "note": "single card"},
    "zs-32b": {"model": "Qwen/Qwen3-32B", "tp": 2, "note": "tensor-parallel over both cards"},
    "zs-moe": {"model": "Qwen/Qwen3-235B-A22B-Instruct-2507", "tp": 2,
               "note": "235B MoE across 192 GB — strongest zero-shot"},
}
DEFAULT_ZS = "zs-32b"

def resolve(preset):
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset '{preset}'. choices: {', '.join(PRESETS)}")
    return PRESETS[preset]

def print_table():
    print("Fine-tune presets   (nl2tags train --preset NAME)")
    for k, v in PRESETS.items():
        star = "  <- default" if k == DEFAULT_PRESET else ""
        print(f"  {k:9} {v['base']:22} {v['mode']:6} {v['vram']:9} {v['note']}{star}")
    print("\nZero-shot models    (vLLM serve, then: nl2tags serve --proxy)")
    for k, v in ZERO_SHOT.items():
        print(f"  {k:7} {v['model']:40} tp={v['tp']}  {v['note']}")
