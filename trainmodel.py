# train_qwen2.5_lora.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ------------------------------
# 1. Load dataset (subset)
# ------------------------------
dataset = load_dataset("open-index/hacker-news", split="train[:10000]")  # 10k lines

# For simplicity, only keep 'text' field
dataset = dataset.map(lambda x: {"text": x["text"]})

# ------------------------------
# 2. Load tokenizer and model
# ------------------------------
model_name = "Qwen/Qwen-2.5-7B-sft"  # change if you have local safetensors
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 8-bit for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Prepare model for LoRA
model = prepare_model_for_int8_training(model)

# ------------------------------
# 3. LoRA configuration
# ------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # typical for causal LM
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ------------------------------
# 4. Tokenize dataset
# ------------------------------
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ------------------------------
# 5. Data collator
# ------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ------------------------------
# 6. Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./qwen2.5_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    fp16=True,
    optim="adamw_torch",
    save_total_limit=2,
)

# ------------------------------
# 7. Initialize Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ------------------------------
# 8. Train
# ------------------------------
trainer.train()

# ------------------------------
# 9. Save LoRA adapters
# ------------------------------
model.save_pretrained("./qwen2.5_lora")
tokenizer.save_pretrained("./qwen2.5_lora")

print("Training finished, model saved at ./qwen2.5_lora")