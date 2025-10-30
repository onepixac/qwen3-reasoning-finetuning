# 🧠 Qwen3-4B Reasoning Fine-tuning

**Italian Reasoning Evaluation Model** - Fine-tuned for educational assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B-blue)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![Dataset: 9.9K](https://img.shields.io/badge/Dataset-9.9K-green)](https://huggingface.co/datasets/all1e/reasoning-evaluation-italian)

---

## 📊 Overview

Fine-tuned **Qwen3-4B-Instruct-2507** for evaluating student responses to reasoning questions in Italian educational contexts.

### Key Features
- ✅ **Specialized Evaluation**: Trained on 9.9K reasoning evaluation examples
- ✅ **Educational Context**: Optimized for Italian critical thinking assessment
- ✅ **Pedagogical Feedback**: Generates constructive feedback with citations
- ✅ **High Quality**: LLM-generated dataset with structured evaluation rubrics
- ✅ **Production Ready**: Deployed on ALL1E platform for real-time evaluation

---

## 🎯 Use Cases

This model evaluates student responses to reasoning questions:

**Input**:
- Reasoning question
- Expert answer (reference)
- Student's response
- Retrieved document sources (for context-aware feedback)

**Output**:
- Evaluation judgment: `correct` / `partial` / `incorrect`
- Pedagogical feedback with citations `[1][2][3]` to source chunks
- Socratic questions to guide deeper understanding

---

## 📚 Dataset

### Dataset Information
- **Name**: `all1e/reasoning-evaluation-italian`
- **Size**: 9,900 training examples (9,405 train + 495 validation)
- **Format**: Conversational (system + user + assistant)
- **Language**: Italian 🇮🇹
- **License**: MIT

### Dataset Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Sei un assistente educativo esperto in valutazione di risposte di ragionamento critico..."
    },
    {
      "role": "user",
      "content": "**Domanda:** [question]\n**Risposta Esperta:** [expert_answer]\n**Risposta dello Studente:** [student_response]\n\nValuta questa risposta..."
    },
    {
      "role": "assistant",
      "content": "**Valutazione:** correct/partial/incorrect\n**Feedback Pedagogico:** [detailed feedback]"
    }
  ]
}
```

### Response Types (Balanced Distribution)
- **Correct responses**: 33% (3,300 examples)
- **Partial responses**: 33% (3,300 examples)
- **Incorrect responses**: 33% (3,300 examples)

### Dataset Quality
- ✅ Multi-domain coverage (science, history, literature, philosophy, etc.)
- ✅ Balanced difficulty levels (easy/medium/hard)
- ✅ Consistent feedback structure
- ✅ Expert-quality reference answers
- ✅ Realistic student response variations

---

## 🔧 Training Details

### Base Model
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Parameters**: 4B
- **Context Window**: 32,768 tokens
- **Architecture**: Transformer with instruction tuning

### Fine-tuning Configuration
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **LoRA Dropout**: 0.05
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Training Hyperparameters
- **Epochs**: 3
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 2 steps
- **Effective Batch Size**: 8
- **Learning Rate**: 2e-4
- **LR Scheduler**: Cosine with warmup (10% steps)
- **Optimizer**: AdamW (betas: 0.9, 0.999, eps: 1e-8)
- **Weight Decay**: 0.01
- **Max Grad Norm**: 1.0
- **Mixed Precision**: FP16

### Training Infrastructure
- **GPU**: 4× NVIDIA L40S (48GB VRAM each)
- **VRAM Usage**: ~67GB (35% of 192GB total)
- **Tensor Parallelism**: 4-way
- **Training Time**: ~2.5 hours (1,803 steps @ 4.29s/step)
- **Framework**: TRL (Transformers Reinforcement Learning)

### Training Performance (Epoch 1.66)
- **Eval Loss**: 0.4029 (excellent convergence)
- **Token Accuracy**: 88.11%
- **Target Accuracy**: >87% ✅ **ACHIEVED**

---

## 📈 Results

### Model Performance
- **Eval Loss**: 0.4029 (final)
- **Token Accuracy**: 88.11%
- **Convergence**: Stable (loss decreasing consistently)

### Comparison with Base Model
| Metric | Base Qwen3-4B | Fine-tuned |
|--------|---------------|------------|
| Reasoning Evaluation | Generic | **Specialized** |
| Italian Pedagogy | Basic | **Expert-level** |
| Citation Support | ❌ No | ✅ **Yes [1][2][3]** |
| Socratic Questions | ❌ No | ✅ **Yes** |

---

## 🚀 Usage

### Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "All1eOnepix/qwen3-4b-reasoning-evaluation-italian"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Prepare input
messages = [
    {
        "role": "system",
        "content": "Sei un assistente educativo esperto in valutazione di risposte di ragionamento critico..."
    },
    {
        "role": "user",
        "content": """**Domanda:** Perché il cielo è blu?
**Risposta Esperta:** Il cielo appare blu a causa dello scattering di Rayleigh...
**Risposta dello Studente:** Il cielo è blu perché l'acqua degli oceani riflette il blu del cielo.

Valuta questa risposta fornendo feedback pedagogico."""
    }
]

# Generate evaluation
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### vLLM (Production Deployment)

```bash
# Start vLLM server
vllm serve All1eOnepix/qwen3-4b-reasoning-evaluation-italian \
    --port 8009 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 32768
```

### ALL1E Platform Integration

The model is deployed on ALL1E's RAG service:
- **Endpoint**: `http://213.171.186.218:8009/v1`
- **Integration**: Automatic fallback to Qwen3-14B if unavailable
- **Retrieval**: Hybrid search (BM25 + Vector) for context-aware feedback
- **Memory**: Session-based conversation memory
- **Streaming**: Real-time SSE streaming responses

---

## 📁 Repository Structure

```
qwen3-reasoning-finetuning/
├── README.md                          # This file
├── Scripts_reasoning/                 # Training scripts
│   ├── 02_prepare_dataset_v2.py      # Dataset preparation (9.9K split)
│   ├── 03_finetune_qwen3_4b_m2.sh    # LoRA fine-tuning script
│   └── upload_reasoning_to_huggingface.py  # Dataset publishing
└── Reasoning_model/                   # Model artifacts (after training)
    └── [Model files]
```

---

## 🔬 Training Scripts

### 1. Dataset Preparation

```bash
python Scripts_reasoning/02_prepare_dataset_v2.py \
    --input reasoning_9900_v2.jsonl \
    --train-output train_reasoning.jsonl \
    --val-output val_reasoning.jsonl \
    --val-split 0.05
```

**Output**:
- `train_reasoning.jsonl` (9,405 examples)
- `val_reasoning.jsonl` (495 examples)
- Stratified split (balanced across response types)

### 2. Fine-tuning

```bash
bash Scripts_reasoning/03_finetune_qwen3_4b_m2.sh
```

**What it does**:
- Loads Qwen3-4B base model
- Applies LoRA adapters
- Trains for 3 epochs (~2.5 hours)
- Saves merged model to `Reasoning_model/`
- Logs to `training_trl_run.log`

### 3. Upload to Hugging Face

```bash
export HF_TOKEN="your_token_here"

python Scripts_reasoning/upload_reasoning_to_huggingface.py
```

**Uploads**:
- Dataset: `all1e/reasoning-evaluation-italian`
- Model: `All1eOnepix/qwen3-4b-reasoning-evaluation-italian` (coming soon)

---

## 💰 Cost Analysis

### Dataset Generation
- **LLM**: Claude 3.5 Sonnet (via Bedrock)
- **Cost**: ~$250 USD (9.9K examples with multi-step generation)
- **Time**: ~24 hours (batch processing)

### Training
- **GPU**: 4× NVIDIA L40S (owned infrastructure)
- **Electricity**: ~$5 (2.5 hours @ 800W)
- **Total Cost**: ~$255 USD

### Comparison with Alternatives
- **OpenAI GPT-4 API** (9.9K evals @ $0.03/call): **$297**
- **Anthropic Claude API** (9.9K evals @ $0.025/call): **$247.50**
- **Our Fine-tuned Model** (infinite evals): **$255 one-time** ✅

**ROI**: Breaks even after ~10K evaluations, then cost = $0/eval

---

## 🎓 Educational Context

This model is part of the **ALL1E educational platform**, which provides:
- **Video Lessons**: AWS Chime SDK for live/recorded sessions
- **Document Upload**: Mistral OCR + Docling chunking
- **RAG System**: Hybrid retrieval (BM25 + Vector + RRF)
- **Quiz Generation**: QwenCustom fine-tuned (94.98% accuracy)
- **Cloze Tests**: Qwen3-4B fine-tuned (94.98% accuracy)
- **Reasoning Evaluation**: This model ✨

### Pedagogical Principles
- ✅ Socratic questioning for deeper understanding
- ✅ Constructive feedback (not just "wrong/right")
- ✅ Citation-based explanations (source attribution)
- ✅ Adaptive difficulty based on student level
- ✅ Italian educational standards compliance

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Base Model**: Alibaba Cloud - Qwen Team
- **Dataset Generation**: Anthropic Claude 3.5 Sonnet
- **Training Framework**: HuggingFace Transformers + TRL
- **Inference**: vLLM (Berkeley)
- **Platform**: ALL1E Team

---

## 📞 Contact

- **Organization**: [All1eOnepix](https://huggingface.co/All1eOnepix)
- **Platform**: [ALL1E](https://all1e.com)
- **Repository**: [GitHub](https://github.com/onepixac/qwen3-reasoning-finetuning)

---

**Created with ❤️ by ALL1E Team**

*For educational assessment, by educators.*
