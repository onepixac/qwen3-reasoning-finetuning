#!/usr/bin/env python3
"""
Upload Reasoning Evaluation Dataset to Hugging Face
Similar to cloze dataset upload
"""

import os
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Configuration
DATASET_NAME = "all1e/reasoning-evaluation-italian"
TRAIN_FILE = "train_reasoning.jsonl"
VAL_FILE = "val_reasoning.jsonl"
ORIGINAL_FILE = "reasoning_9900_v2.jsonl"

# Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable not set")
    print("Please set it with: export HF_TOKEN='your_token_here'")
    exit(1)

print("=" * 60)
print("🚀 REASONING EVALUATION DATASET UPLOAD TO HUGGING FACE")
print("=" * 60)
print()

# Login to Hugging Face
print("🔐 Logging in to Hugging Face...")
login(token=HF_TOKEN)
print("✅ Logged in successfully")
print()

# Load datasets
print("📂 Loading datasets...")

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)
original_data = load_jsonl(ORIGINAL_FILE)

print(f"✅ Loaded {len(train_data)} training examples")
print(f"✅ Loaded {len(val_data)} validation examples")
print(f"✅ Loaded {len(original_data)} original questions")
print()

# Create Dataset objects
print("🔨 Creating Hugging Face datasets...")
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

print("✅ Dataset created successfully")
print()

# Dataset info
print("📊 Dataset Statistics:")
print(f"  - Total training examples: {len(train_data)}")
print(f"  - Total validation examples: {len(val_data)}")
print(f"  - Original questions: {len(original_data)}")
print(f"  - Expansion factor: {(len(train_data) + len(val_data)) / len(original_data):.1f}x")
print(f"  - Train/Val split: {len(train_data) / (len(train_data) + len(val_data)) * 100:.1f}% / {len(val_data) / (len(train_data) + len(val_data)) * 100:.1f}%")
print()

# Sample example
print("📝 Sample Training Example:")
sample = train_data[0]
print(f"  System: {sample['messages'][0]['content'][:100]}...")
print(f"  User: {sample['messages'][1]['content'][:150]}...")
print(f"  Assistant: {sample['messages'][2]['content'][:150]}...")
print()

# Create README
readme_content = f"""# Reasoning Evaluation Dataset - Italian

## Dataset Description

This dataset contains **{len(original_data)} high-quality reasoning questions** in Italian, expanded to **{len(train_data) + len(val_data)} training examples** for fine-tuning language models to evaluate student responses.

### Dataset Summary

- **Language**: Italian 🇮🇹
- **Task**: Reasoning evaluation (critical thinking assessment)
- **Format**: Conversational (user question + assistant evaluation)
- **Original Questions**: {len(original_data)}
- **Training Examples**: {len(train_data)} (95%)
- **Validation Examples**: {len(val_data)} (5%)
- **Expansion Factor**: 3x (each question generates 3 evaluation examples)

### Intended Use

This dataset is designed to fine-tune LLMs for **evaluating student responses** to reasoning questions in educational contexts.

**Use Cases:**
- Fine-tuning models for educational assessment
- Training AI tutors for reasoning evaluation
- Developing automated feedback systems
- Research in educational NLP

### Dataset Structure

#### Training/Validation Split

Each example follows the conversational format:

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "Sei un assistente educativo esperto in valutazione di risposte di ragionamento critico..."
    }},
    {{
      "role": "user",
      "content": "**Domanda:** [reasoning question]\\n**Risposta Esperta:** [expert answer]\\n**Risposta dello Studente:** [student response]\\n\\nValuta questa risposta..."
    }},
    {{
      "role": "assistant",
      "content": "**Valutazione:** [correct/partial/incorrect]\\n**Feedback Pedagogico:** [detailed feedback]"
    }}
  ]
}}
```

#### Response Types

Each original question includes 3 student response types:
- **correct**: Fully correct responses (33%)
- **partial**: Partially correct responses (33%)
- **incorrect**: Incorrect responses (33%)

This ensures balanced training across all evaluation categories.

### Data Fields

- `messages`: List of conversation turns
  - `role`: Speaker role ("system", "user", "assistant")
  - `content`: Message content

### Dataset Creation

#### Source Data

The dataset was created using:
1. **Base Questions**: Generated from educational documents using Qwen3-14B
2. **Expert Answers**: High-quality reference answers
3. **Student Responses**: Simulated responses at different correctness levels
4. **Feedback**: Pedagogically appropriate evaluation and guidance

#### Data Quality

- ✅ All questions reviewed for educational relevance
- ✅ Balanced distribution across response types
- ✅ Consistent feedback structure
- ✅ Domain diversity (various educational topics)

### Languages

- Italian (it)

### Licensing Information

This dataset is released under the **MIT License**.

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{{reasoning_evaluation_italian,
  title={{Reasoning Evaluation Dataset - Italian}},
  author={{ALL1E Team}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{DATASET_NAME}}}
}}
```

### Dataset Card Authors

- ALL1E Team (@All1eOnepix)

### Contact

For questions or feedback, please open an issue on the dataset repository.

---

**Created with ❤️ by ALL1E Team**
"""

# Save README locally
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("📄 README.md created")
print()

# Push to Hugging Face
print(f"📤 Uploading to Hugging Face: {DATASET_NAME}")
print("   This may take a few minutes...")
print()

try:
    dataset_dict.push_to_hub(
        DATASET_NAME,
        token=HF_TOKEN,
        private=False,  # Make it public
        commit_message="Initial upload: Reasoning evaluation dataset for Italian education"
    )
    
    print("✅ Dataset uploaded successfully!")
    print()
    print("🔗 Dataset URL:")
    print(f"   https://huggingface.co/datasets/{DATASET_NAME}")
    print()
    
    # Upload README separately to ensure it's included
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=DATASET_NAME,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add detailed README"
    )
    
    print("✅ README uploaded successfully!")
    print()
    
    print("=" * 60)
    print("🎉 UPLOAD COMPLETE!")
    print("=" * 60)
    print()
    print("📊 Final Statistics:")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Training examples: {len(train_data)}")
    print(f"  - Validation examples: {len(val_data)}")
    print(f"  - Total size: {(len(train_data) + len(val_data))} examples")
    print()
    print("🚀 Your dataset is now publicly available!")
    print(f"   Load it with: dataset = load_dataset('{DATASET_NAME}')")
    print()

except Exception as e:
    print(f"❌ Error uploading dataset: {e}")
    exit(1)
