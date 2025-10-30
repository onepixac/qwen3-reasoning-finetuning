#!/usr/bin/env python3
"""
Prepare reasoning dataset v2 for fine-tuning
Uses reasoning_9900_v2.jsonl format with answer + student_responses
Converts to instruction format for evaluation task
"""

import json
import random
from pathlib import Path

INPUT_FILE = "reasoning_9900_v2.jsonl"
TRAIN_FILE = "train_reasoning.jsonl"
VAL_FILE = "val_reasoning.jsonl"
TRAIN_SPLIT = 0.95  # 95% train, 5% validation

def convert_to_instruction_format(example):
    """
    Convert reasoning example to instruction format for evaluation task.
    Creates 3 training examples per question (one for each student response type).
    """
    
    training_examples = []
    
    # System prompt for reasoning evaluation
    system_prompt = """Sei un assistente educativo esperto in valutazione di risposte di ragionamento critico.

Il tuo compito è:
1. Leggere attentamente la domanda e la risposta dello studente
2. Valutare la qualità del ragionamento (correct/partial/incorrect)
3. Fornire feedback pedagogico personalizzato che:
   - Riconosce i punti di forza
   - Identifica le lacune o misconcezioni
   - Pone domande socratiche per stimolare la riflessione
   - Suggerisce approfondimenti specifici

Usa un tono incoraggiante e costruttivo, adattando il feedback al livello della risposta."""

    # Create one training example for each student response
    for student_resp in example.get('student_responses', []):
        resp_type = student_resp.get('type', 'unknown')
        student_answer = student_resp.get('response', '')
        correct_feedback = student_resp.get('feedback', '')
        
        # User message: question + context + student response
        user_content = f"""**Domanda:**
{example['question']}

**Dominio:** {example.get('domain', 'N/A')}
**Difficoltà:** {example.get('difficulty', 'N/A')}
**Concetti chiave:** {', '.join(example.get('key_concepts', []))}

**Risposta Esperta di Riferimento:**
{example.get('answer', 'N/A')}

---

**Risposta dello Studente:**
{student_answer}

---

Valuta questa risposta fornendo:
1. Tipo di valutazione (correct/partial/incorrect)
2. Feedback pedagogico dettagliato"""

        # Assistant response: evaluation type + feedback
        assistant_content = f"""**Valutazione:** {resp_type}

**Feedback Pedagogico:**
{correct_feedback}"""

        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
        
        training_examples.append(training_example)
    
    return training_examples

def main():
    print("=== Preparing Reasoning Dataset V2 ===")
    print(f"Input: {INPUT_FILE}")
    print(f"Format: answer + student_responses (3 per question)")
    print()
    
    # Load dataset
    print(f"Loading {INPUT_FILE}...")
    examples = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(examples)} questions")
    
    # Convert to training format (3 examples per question)
    print("Converting to instruction format...")
    all_training_examples = []
    
    for example in examples:
        training_examples = convert_to_instruction_format(example)
        all_training_examples.extend(training_examples)
    
    print(f"✅ Generated {len(all_training_examples)} training examples (3 per question)")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_training_examples)
    
    # Split
    split_idx = int(len(all_training_examples) * TRAIN_SPLIT)
    train_examples = all_training_examples[:split_idx]
    val_examples = all_training_examples[split_idx:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_examples)} examples ({len(train_examples)/3:.0f} questions × 3)")
    print(f"   Val: {len(val_examples)} examples ({len(val_examples)/3:.0f} questions × 3)")
    
    # Save
    print(f"\n💾 Saving datasets...")
    
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ Train saved: {TRAIN_FILE}")
    print(f"✅ Val saved: {VAL_FILE}")
    
    # Show sample
    print(f"\n📝 Sample Training Example:")
    print("=" * 80)
    sample = train_examples[0]
    for msg in sample['messages']:
        role = msg['role'].upper()
        content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
        print(f"\n[{role}]")
        print(content)
    print("=" * 80)
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nReady for fine-tuning with:")
    print(f"  python 03_finetune_qwen3_4b.sh")

if __name__ == "__main__":
    main()
