#!/bin/bash
# Fine-tune Qwen3-4B-Instruct-2507 for reasoning evaluation with LoRA on Machine 2
# Based on successful cloze fine-tuning approach (94.98% accuracy)

set -e

echo "=== Qwen3-4B-Instruct-2507 Reasoning Fine-tuning on Machine 2 ==="
echo ""

# Configuration
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR="./qwen3_4b_reasoning_lora"
TRAIN_FILE="train_reasoning.jsonl"
VAL_FILE="val_reasoning.jsonl"

# LoRA parameters (proven from cloze fine-tuning - 94.98% accuracy)
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training hyperparameters
LEARNING_RATE=2e-4
BATCH_SIZE=4
GRADIENT_ACCUM=4  # Effective batch size: 16
NUM_EPOCHS=3
MAX_SEQ_LENGTH=2048
WARMUP_STEPS=100

echo "Model: $BASE_MODEL"
echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRADIENT_ACCUM)))"
echo "Epochs: $NUM_EPOCHS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo ""

# Check if running on Machine 2
HOSTNAME=$(hostname)
echo "Running on: $HOSTNAME"

# Create training script
cat > train_reasoning_m2.py << 'PYTHON_SCRIPT'
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration from env
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./qwen3_4b_reasoning_lora")
TRAIN_FILE = os.getenv("TRAIN_FILE", "train_reasoning.jsonl")
VAL_FILE = os.getenv("VAL_FILE", "val_reasoning.jsonl")

LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRADIENT_ACCUM = int(os.getenv("GRADIENT_ACCUM", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))

print("=== Qwen3-4B-Instruct-2507 Reasoning LoRA Fine-tuning ===")
print(f"Base model: {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Training: {TRAIN_FILE} ({NUM_EPOCHS} epochs)")
print(f"Validation: {VAL_FILE}")
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n📊 Trainable parameters:")
model.print_trainable_parameters()

# Load datasets
print("\nLoading datasets...")
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples: {len(val_dataset)}")

# Tokenization function
def tokenize_function(examples):
    # Format: system + user + assistant messages
    texts = []
    for msgs in examples["messages"]:
        text = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        texts.append(text)
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False
    )

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

print("\n📋 Training configuration:")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUM}")
print(f"  Total steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUM) * NUM_EPOCHS}")
print(f"  Optimizer: paged_adamw_8bit")
print(f"  Scheduler: cosine")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train
print("\n🚀 Starting training...")
print("=" * 80)
trainer.train()

# Save final model
print("\n💾 Saving final LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete!")
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("1. Run 04_merge_lora.sh to merge adapter with base model")
print("2. Deploy merged model on Machine 2")
PYTHON_SCRIPT

# Export configuration
export BASE_MODEL
export OUTPUT_DIR
export TRAIN_FILE
export VAL_FILE
export LORA_R
export LORA_ALPHA
export LORA_DROPOUT
export LEARNING_RATE
export BATCH_SIZE
export GRADIENT_ACCUM
export NUM_EPOCHS
export MAX_SEQ_LENGTH
export WARMUP_STEPS

# Run training
echo "Starting training..."
python3 train_reasoning_m2.py 2>&1 | tee training_reasoning_m2.log

echo ""
echo "=== Training Complete ==="
echo "Logs: training_reasoning_m2.log"
echo "LoRA adapter: $OUTPUT_DIR"
echo "Next: Run 04_merge_lora.sh to merge adapter with base model"
