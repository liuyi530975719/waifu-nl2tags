"""
QLoRA / full supervised fine-tune of a bilingual base to translate NL -> Danbooru tags.
Trains only on the assistant completion.

  nl2tags train --preset max                 # Qwen3-32B QLoRA (default for dual RTX PRO 6000)
  nl2tags train --preset strong              # Qwen3-14B QLoRA
  accelerate launch -m nl2tags.train_qlora --preset full-8b   # full FT across both GPUs
  nl2tags train --model Qwen/Qwen3-8B        # explicit base
"""
from __future__ import annotations
import argparse, os

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nl2tags train")
    ap.add_argument("--preset", default=None, help="fast|balanced|strong|max|full-8b")
    ap.add_argument("--model", default=None, help="override base model id")
    ap.add_argument("--full", action="store_true", help="full fine-tune (no 4-bit / no LoRA)")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="out/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--maxlen", type=int, default=1024)
    a = ap.parse_args(argv)

    from .presets import PRESETS, DEFAULT_PRESET
    preset = a.preset or (DEFAULT_PRESET if not a.model else None)
    full = a.full
    model_id = a.model
    if preset:
        p = PRESETS[preset]
        model_id = model_id or p["base"]
        full = full or (p["mode"] == "full")
    if not model_id:
        model_id = PRESETS[DEFAULT_PRESET]["base"]
    print(f"base={model_id}  mode={'full' if full else 'qlora'}  out={a.out}")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_cfg = None
    if full:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        world = int(os.environ.get("WORLD_SIZE", "1"))
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))} if world > 1 else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map=device_map, torch_dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                              task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"])
    model.config.use_cache = False

    ds = load_dataset("json", data_files={"train": a.train, "val": a.val})
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
        bf16=True, max_length=a.maxlen, packing=False,
        assistant_only_loss=True, gradient_checkpointing=True, report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["val"], peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print("saved ->", a.out)

if __name__ == "__main__":
    main()
