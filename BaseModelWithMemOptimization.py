from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Configuration
model_name = "microsoft/phi-2"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load components
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Apply LoRA
# Configure LoRA
lora_config = LoraConfig(
    r=8,  # rank of low-rank approximation
    lora_alpha=16,  # scaling factor for LoRA updates
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.1,  # dropout rate for LoRA layers
    bias="none",  # setting the bias type
    task_type="CAUSAL_LM"  # task type
)
model = get_peft_model(model, lora_config)
model.to(device)
'''
for name, module in model.named_modules():
    if "proj" in name or "dense" in name:
        print(name)'
'''

# Load dataset (dummy example)
dataset = load_dataset("OpenAssistant/oasst1")
print(dataset)
print(dataset["train"][0])

from datasets import Dataset

def create_prompt_completion_pairs(dataset_split):
    prompts = []
    responses = []
    
    # Basic: assumes sorted conversation threads
    for i in range(len(dataset_split) - 1):
        current_row = dataset_split[i]
        next_row = dataset_split[i + 1]

        if current_row["role"] == "prompter" and next_row["role"] == "assistant":
            prompts.append(current_row["text"])
            responses.append(next_row["text"])
    
    paired_data = [{"prompt": p, "completion": r} for p, r in zip(prompts, responses)]
    
    return Dataset.from_list(paired_data)

train_paired_dataset = create_prompt_completion_pairs(dataset["train"])
val_paired_dataset = create_prompt_completion_pairs(dataset["validation"])

# Optional sanity check
print(train_paired_dataset[0])

# Preprocess function
def preprocess_function(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]

    texts = [
        f"### Prompt:\n{prompt}\n\n### Response:\n{completion}"
        for prompt, completion in zip(prompts, completions)
    ]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=256,
        return_attention_mask=True,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# Tokenize both splits
tokenized_train = train_paired_dataset.map(preprocess_function, batched=True)
tokenized_val = val_paired_dataset.map(preprocess_function, batched=True)


# Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
# Training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./phi2-oasst1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increased accumulation
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,  # Use bfloat16 instead of fp16
    optim="adamw_torch",  # More memory-efficient optimizer
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    evaluation_strategy="steps",
    report_to="none",  # Disable external logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========================
# Diagnostic Tests
# ========================
def run_safety_checks(model, trainer):
    """Pre-training validation suite"""
    print(f"\n=== Running Safety Checks on {device} ===")  # Device info
    
    # Check 1: Trainable parameters
    print("\n=== Trainable Layers ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
    
    # Check 2: Single batch forward/backward
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        batch = {k: v.to(device) for k, v in batch.items()}
        
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

# Pre-flight checks
run_safety_checks(model, trainer)
trainer.train()
#trainer.save_model("./phi2-oasst1")
