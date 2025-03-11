"""
Phi-2 Fine-Tuning on Apple M1/M2 with Memory Optimizations
Includes safety checks and diagnostic tools
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse

# ========================
# Configuration
# ========================
MODEL_NAME = "microsoft/phi-2"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n=== Using device: {DEVICE} ===\n")

# ========================
# Model Loading with Safety
# ========================
def load_model():
    """Load model with memory optimizations for M1"""
    print(f"\n=== Initializing on {DEVICE} ===")  # Device confirmation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Add module name inspection
        print("\n=== Layer Name Inspection ===")
        for name, module in model.named_modules():
            if "proj" in name or "dense" in name:
                print(f"Detected layer: {name}")
        
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise



# ========================
# PEFT Configuration
# ========================
def get_peft_config():
    return LoraConfig(
        r=4,  # Reduced for M1 stability
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"]
    )

# ========================
# Dataset Processing
# ========================
def process_dataset(dataset):
    """Create prompt-completion pairs with validation"""
    def _create_pairs(split):
        pairs = []
        for i in range(len(split)-1):
            if split[i]["role"] == "prompter" and split[i+1]["role"] == "assistant":
                pairs.append({
                    "prompt": split[i]["text"],
                    "completion": split[i+1]["text"]
                })
        return pairs
    
    train_pairs = _create_pairs(dataset["train"])
    val_pairs = _create_pairs(dataset["validation"])

    # Dataset sanity check
    print("\n=== Dataset Sample ===")
    print(f"First training example: {train_pairs[0]}")  # Moved inside function
    
    return train_pairs, val_pairs

def preprocess_function(examples):
    """Dynamic padding implementation"""
    texts = [
        f"### Prompt:\n{p}\n\n### Response:\n{c}"
        for p, c in zip(examples["prompt"], examples["completion"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding=False,  # Dynamic padding in collator
        return_attention_mask=True,
    )

# ========================
# Training Setup
# ========================
def configure_training():
    return TrainingArguments(
        output_dir="./phi2-oasst1",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_pin_memory=False  # Critical for MPS
    )

# ========================
# Diagnostic Tests
# ========================
def run_safety_checks(model, trainer):
    """Pre-training validation suite"""
    print(f"\n=== Running Safety Checks on {DEVICE} ===")  # Device info
    
    # Check 1: Trainable parameters
    print("\n=== Trainable Layers ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
    
    # Check 2: Single batch forward/backward
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        print("\nTesting forward pass...")
        with torch.no_grad():
            outputs = model(**batch)
            print(f"Forward loss: {outputs.loss.item():.4f}")
        
        print("\nTesting backward pass...")
        outputs = model(**batch)
        outputs.loss.backward()
        print("Backward pass succeeded")
        
        # Check parameter updates
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Gradients found for {name}")
                break
        else:
            raise RuntimeError("No gradients detected in any parameter")
                
    except Exception as e:
        print(f"\n!!! Pre-training check failed: {e}")
        raise

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load components
    model = load_model()
    peft_config = get_peft_config()
    model = get_peft_model(model, peft_config)
    model.to(DEVICE)
    model.enable_input_require_grads()
    
    # Data preparation
    dataset = load_dataset("OpenAssistant/oasst1")
    train_data, val_data = process_dataset(dataset)
    
    # Tokenization
    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_val = val_data.map(preprocess_function, batched=True)
    
    # Training setup
    training_args = configure_training()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Pre-flight checks
    run_safety_checks(model, trainer)
    
    # Start training with memory monitoring
    print("\n=== Starting Training ===")
    try:
        trainer.train()
        model.save_pretrained("./phi2-oasst1-final")
    except RuntimeError as e:
        print(f"\n!!! Training failed: {e}")
        print("Recommended fixes:")
        print("1. Reduce max_length in preprocess_function to 128")
        print("2. Set gradient_accumulation_steps to 8")
        print("3. Use model = model.float() before training")
    
    # Post-training validation
    print("\n=== Training Completed ===")
    print("Run final validation:")
    print(f"!python -c 'from transformers import pipeline; gen = pipeline(\"text-generation\", \"./phi2-oasst1-final\"); print(gen(\"What is AI?\"))'")