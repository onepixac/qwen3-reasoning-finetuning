# Training Scripts Documentation

Complete pipeline for fine-tuning Qwen3-4B on Italian reasoning evaluation.

---

## 📋 Pipeline Overview

```
reasoning_9900_v2.jsonl (raw dataset)
    ↓
02_prepare_dataset_v2.py
    ↓
train_reasoning.jsonl + val_reasoning.jsonl (95/5 split)
    ↓
03_finetune_qwen3_4b_m2.sh
    ↓
Reasoning_model/ (merged LoRA model)
    ↓
upload_reasoning_to_huggingface.py
    ↓
HuggingFace Hub (dataset + model)
```

---

## 1. Dataset Preparation

### `02_prepare_dataset_v2.py`

**Purpose**: Split 9.9K reasoning dataset into train/validation sets with stratification.

**Usage**:
```bash
python 02_prepare_dataset_v2.py \
    --input reasoning_9900_v2.jsonl \
    --train-output train_reasoning.jsonl \
    --val-output val_reasoning.jsonl \
    --val-split 0.05
```

**Parameters**:
- `--input`: Input JSONL file (9,900 examples)
- `--train-output`: Output training file (default: `train_reasoning.jsonl`)
- `--val-output`: Output validation file (default: `val_reasoning.jsonl`)
- `--val-split`: Validation split ratio (default: 0.05 = 5%)

**Output**:
- `train_reasoning.jsonl`: 9,405 examples (95%)
- `val_reasoning.jsonl`: 495 examples (5%)
- Statistics printed to console

**Features**:
- ✅ Stratified split (balanced across correct/partial/incorrect responses)
- ✅ Shuffle with fixed seed (reproducibility)
- ✅ Validation of JSON structure
- ✅ Progress tracking

**Example Output**:
```
📊 Dataset Statistics:
  - Total examples: 9,900
  - Training: 9,405 (95.00%)
  - Validation: 495 (5.00%)
  
✅ Dataset split completed successfully!
```

---

## 2. Fine-tuning

### `03_finetune_qwen3_4b_m2.sh`

**Purpose**: Fine-tune Qwen3-4B with LoRA on reasoning evaluation task.

**Usage**:
```bash
bash 03_finetune_qwen3_4b_m2.sh
```

**Configuration**:

#### Model Settings
- **Base Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Max Sequence Length**: 2048 tokens
- **Chat Template**: Qwen3 format

#### LoRA Settings
- **Rank (r)**: 64
- **Alpha**: 16
- **Dropout**: 0.05
- **Target Modules**: All linear layers (q/k/v/o/gate/up/down projections)
- **Task Type**: Causal Language Modeling

#### Training Hyperparameters
- **Epochs**: 3
- **Per-device Batch Size**: 4
- **Gradient Accumulation**: 2 steps
- **Effective Batch Size**: 8 (4 × 2)
- **Learning Rate**: 2e-4
- **LR Scheduler**: Cosine with warmup
- **Warmup Ratio**: 0.1 (10% of total steps)
- **Optimizer**: AdamW
  - Beta1: 0.9
  - Beta2: 0.999
  - Epsilon: 1e-8
- **Weight Decay**: 0.01
- **Max Grad Norm**: 1.0
- **Mixed Precision**: FP16

#### Distributed Training
- **Tensor Parallelism**: 4-way (4× L40S GPUs)
- **Data Parallelism**: Disabled (tensor-parallel only)
- **GPU Memory Utilization**: 35% per GPU

#### Evaluation & Saving
- **Eval Strategy**: Every 100 steps
- **Save Strategy**: Every 100 steps
- **Logging**: Every 10 steps
- **Save Total Limit**: 3 checkpoints (keep best 3)
- **Load Best at End**: True (automatic best checkpoint selection)

#### Output
- **Checkpoints**: `./reasoning_checkpoints_trl/`
- **Merged Model**: `./Reasoning_model/` (final output)
- **Logs**: `training_trl_run.log`

**Hardware Requirements**:
- 4× NVIDIA L40S (48GB each)
- ~67GB VRAM total (35% of 192GB)
- ~2.5 hours training time

**Training Progress**:
```
Epoch 1/3: 100%|██████████| 601/601 [40:23<00:00,  4.29s/it]
Eval Loss: 0.4323 | Token Accuracy: 87.38%

Epoch 2/3: 100%|██████████| 601/601 [40:23<00:00,  4.29s/it]
Eval Loss: 0.4177 | Token Accuracy: 87.69%

Epoch 3/3: 100%|██████████| 601/601 [40:23<00:00,  4.29s/it]
Eval Loss: 0.4029 | Token Accuracy: 88.11%
```

**Monitoring**:
```bash
# Watch live training progress
tail -f training_trl_run.log | grep -E "loss|accuracy|epoch"

# Check GPU usage
nvidia-smi -l 1
```

---

## 3. Upload to Hugging Face

### `upload_reasoning_to_huggingface.py`

**Purpose**: Publish dataset and model to Hugging Face Hub.

**Usage**:
```bash
# Set Hugging Face token
export HF_TOKEN="your_token_here"

# Upload dataset
python upload_reasoning_to_huggingface.py
```

**What it uploads**:

#### Dataset
- **Repository**: `all1e/reasoning-evaluation-italian`
- **Files**:
  - `train_reasoning.jsonl` (9,405 examples)
  - `val_reasoning.jsonl` (495 examples)
  - `README.md` (dataset card)
- **Visibility**: Public

#### Model (Manual - Not in Script)
- **Repository**: `All1eOnepix/qwen3-4b-reasoning-evaluation-italian`
- **Files**: All from `Reasoning_model/`
- **Visibility**: Public

**Features**:
- ✅ Automatic dataset card generation
- ✅ Statistics calculation
- ✅ Train/validation split info
- ✅ Citation information
- ✅ Usage examples
- ✅ Progress tracking

**Output**:
```
🚀 REASONING EVALUATION DATASET UPLOAD TO HUGGING FACE
==========================================================

🔐 Logging in to Hugging Face...
✅ Logged in successfully

📂 Loading datasets...
✅ Loaded 9,405 training examples
✅ Loaded 495 validation examples

🔨 Creating Hugging Face datasets...
✅ Dataset created successfully

📊 Dataset Statistics:
  - Total training examples: 9,405
  - Total validation examples: 495
  - Expansion factor: 3.0x
  - Train/Val split: 95.0% / 5.0%

📄 README.md created

📤 Uploading to Hugging Face: all1e/reasoning-evaluation-italian
   This may take a few minutes...

✅ Dataset uploaded successfully!

🔗 Dataset URL:
   https://huggingface.co/datasets/all1e/reasoning-evaluation-italian

✅ README uploaded successfully!

🎉 UPLOAD COMPLETE!
```

**Prerequisites**:
- Hugging Face account
- Write access token
- `huggingface-hub` Python package

**Token Setup**:
```bash
# Option 1: Environment variable
export HF_TOKEN="hf_..."

# Option 2: huggingface-cli login
huggingface-cli login

# Option 3: Pass as argument (if script supports it)
python upload_reasoning_to_huggingface.py --token hf_...
```

---

## 🔧 Dependencies

Install required packages:

```bash
# Python packages
pip install torch transformers trl peft datasets huggingface-hub accelerate

# System dependencies (for distributed training)
pip install deepspeed  # Optional, for advanced optimization
```

**Versions Used**:
- Python: 3.10+
- PyTorch: 2.0+
- Transformers: 4.36+
- TRL: 0.7+
- PEFT: 0.7+

---

## 📈 Performance Benchmarks

### Training Speed
- **Time per epoch**: ~40 minutes
- **Total training time**: ~2.5 hours (3 epochs)
- **Throughput**: ~4.29 seconds/step
- **GPU Utilization**: 35% per GPU (balanced)

### Memory Usage
- **Model Loading**: 8GB (base model)
- **Training Peak**: 67GB total (16.75GB per GPU)
- **Gradient Accumulation**: Helps fit larger batches

### Final Results
- **Eval Loss**: 0.4029
- **Token Accuracy**: 88.11%
- **Target Met**: ✅ >87% accuracy achieved

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 2  # Instead of 4

# Increase gradient accumulation
--gradient_accumulation_steps 4  # Instead of 2

# Reduce max sequence length
--max_seq_length 1024  # Instead of 2048
```

### Slow Training
```bash
# Enable FP16 mixed precision (if not already)
--fp16

# Use gradient checkpointing
--gradient_checkpointing

# Increase batch size (if memory allows)
--per_device_train_batch_size 8
```

### Model Not Converging
```bash
# Increase learning rate
--learning_rate 5e-4  # Instead of 2e-4

# Increase LoRA rank
--lora_r 128  # Instead of 64

# Add more training epochs
--num_train_epochs 5  # Instead of 3
```

---

## 📚 Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [vLLM Serving](https://docs.vllm.ai/)

---

**Questions?** Open an issue on [GitHub](https://github.com/onepixac/qwen3-reasoning-finetuning/issues)
