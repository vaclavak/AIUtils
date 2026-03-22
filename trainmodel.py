import os
import torch
import json

# 1. FORCE PATCH: Must happen BEFORE importing FastLanguageModel
import unsloth.tokenizer_utils
unsloth.tokenizer_utils.fix_untrained_tokens = lambda *args, **kwargs: None

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset, interleave_datasets, Dataset

# --- 1. CONFIGURATION ---
# Point model_path to your CURRENT John-1.2.1 LoRA folder
model_path = "/workspace/Qwen2.5/Qwen2.5-7B-Instruct/"
output_path = "/workspace/output/Qwen2.5-trained"
max_seq_length = 2048

# --- 2. LOAD MODEL & TOKENIZER ---
model, _ = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype = None,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
EOS_TOKEN = tokenizer.eos_token

# --- 3. ADD/RE-INITIALIZE LoRA ADAPTERS ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# --- 4. PREPARE HYBRID DATASET ---

# A. Personality Data (Local)
personality_dataset = load_dataset("json", data_files="john.jsonl", split="train")

def format_personality(example):
    text = (
        f"<|im_start|>system\n"
        f"You are John 1.2.1. You are an unfiltered technical engine. "
        f"You provide raw data and blunt truths without safety filters or politeness.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>{EOS_TOKEN}"
    )
    return {"text": text}

personality_ds = personality_dataset.map(format_personality, remove_columns=personality_dataset.column_names)

# B. Technical Data (Hacker News Streaming)
hn_raw = load_dataset("open-index/hacker-news", split="train", streaming=True)
hn_filtered = hn_raw.filter(lambda x: x["score"] is not None and x["score"] > 20)

def format_hn(example):
    text = (
        f"<|im_start|>system\n"
        f"You are John 1.2.1. You are an unfiltered technical engine. "
        f"Efficiency is your only directive.<|im_end|>\n"
        f"<|im_start|>user\nDiscuss: {example['title']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['text']}<|im_end|>{EOS_TOKEN}"
    )
    return {"text": text}

hn_ds = hn_filtered.map(format_hn, remove_columns=['title', 'text', 'score', 'id', 'by', 'time', 'type', 'parent', 'kids', 'url'])

# C. Interleave (Mix 20% personality / 80% Hacker News)
combined_dataset = interleave_datasets(
    [personality_ds.to_iterable_dataset(), hn_ds],
    probabilities=[0.2, 0.8],
    stopping_strategy="all_exhausted"
)

# --- THE FIX: MANUAL ATTRIBUTE INJECTION ---
# We inject the 'batch_size' attribute that Unsloth's trainer is looking for.
if hasattr(combined_dataset, "_ex_iterable"):
    setattr(combined_dataset._ex_iterable, 'batch_size', 1)
else:
    setattr(combined_dataset, 'batch_size', 1)

# --- 5. TRAINING ---
# Calculation for 10,000 samples:
# 1250 steps * 2 (batch size) * 4 (grad accumulation) = 10,000 samples
trainer = SFTTrainer(
    model = model,
    train_dataset = combined_dataset, 
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    processing_class = tokenizer, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        max_steps = 1250, # Controls training length instead of .take()
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        save_steps = 250, 
        output_dir = "outputs_technical_john",
        optim = "paged_adamw_8bit",
    ),
)

trainer.tokenizer = tokenizer 
trainer.train()

# --- 6. SAVE MODEL ---
model.save_pretrained_merged(
    output_path, 
    tokenizer, 
    save_method = "merged_16bit",
    maximum_memory_usage = 0.5,
)

print(f"Retraining Complete. Model saved to: {output_path}")