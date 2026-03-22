from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 2048  # Supports RoPE Scaling internally
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4-bit quantization to reduce memory usage

# 2. Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # Optimized Qwen2.5
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (PEFT)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: higher = more parameters to train (16, 32, 64 are common)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized to 0 for Unsloth
    bias = "none",    # Optimized to "none" for Unsloth
    use_gradient_checkpointing = "unsloth", # 4x longer context windows
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Prepare Dataset (Hacker News)
# The Hacker News dataset is huge; we'll take a small slice for this example
dataset = load_dataset("open-index/hacker-news", split="train", streaming=True)
dataset = dataset.take(5000) # Taking 5,000 samples for a quick start

def formatting_prompts_func(examples):
    instructions = "Summarize this Hacker News post or comment:"
    inputs       = examples["text"]
    outputs      = examples["title"] if "title" in examples else "Discussion"
    
    texts = []
    for input_text, output_text in zip(inputs, [outputs]*len(inputs)):
        # Apply Qwen2.5 Chat Template
        text = f"<|im_start|>system\n{instructions}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

# Mapping the formatting
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Set up Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Small step count for testing
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Train
trainer_stats = trainer.train()

# 7. Save the model
model.save_pretrained("qwen25_hn_lora") # Local save
# model.push_to_hub("your-username/qwen25-hn-lora") # Optional: Upload to HF