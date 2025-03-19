import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os
from google.colab import drive
import glob
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get('hf')
login(hf_token)

drive.mount('/content/drive')
save_dir = "/content/drive/MyDrive/phi2-qlora-adapter"
adapter_path = os.path.join(save_dir, "model")  # Path to actual adapter files


# 1. Find latest checkpoint
checkpoint_dirs = glob.glob(os.path.join(save_dir, "checkpoint-*"))
latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime) if checkpoint_dirs else None


# Load Dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train[:90%]")
eval_dataset = load_dataset("OpenAssistant/oasst1", split="train[90%:]")
print(dataset)

# Load Pre-Trained Model with 4-bit Quantization
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # or float16 if no bf16 support
    bnb_4bit_quant_type="nf4",
)

# Correctly load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,          # <- FIXED
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Apply LoRA Adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",                    # <- ADDED
)

# Load model with adapter from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

if latest_checkpoint:
    # Load adapter weights FROM CHECKPOINT DIRECTORY
    model = PeftModel.from_pretrained(
        model,
        latest_checkpoint,
        is_trainable=True
    )
else:
    # First-time setup
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

# 5. Check trainable parameters
model.print_trainable_parameters()


# 6. Training Setup
# Training arguments
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-6,  # Consider lowering to 5e-7 if loss still drops too fast
    warmup_ratio=0.2,  # Longer warmup period
    weight_decay=0.02,  # Added weight decay
    fp16=True,
    optim="adamw_torch",
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    report_to="none",
    max_steps=1000,      # Train for 5k steps
    eval_strategy="no",
    resume_from_checkpoint=bool(latest_checkpoint),
    save_total_limit=2,  # Keep only 2 latest checkpoints
    save_safetensors=False,  # Required for PeFT compatibility
)
# 7. Preprocess function
def preprocess_function(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )

    # Set labels and ignore padding tokens
    labels = []
    for i in range(len(model_inputs["input_ids"])):
        input_ids = model_inputs["input_ids"][i]
        attention_mask = model_inputs["attention_mask"][i]
        label = [
            id_ if mask else -100 for id_, mask in zip(input_ids, attention_mask)
        ]
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs



# 8. Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# 9. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# 10. Optional: print all trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# 11. Train!
if latest_checkpoint:
    trainer.train(resume_from_checkpoint=True)
else:
  trainer.train()

# 12. Save Adapter Weights
import os

# Create separate directories for model and tokenizer in Google Drive
model_save_path = os.path.join(save_dir, "model")
tokenizer_save_path = os.path.join(save_dir, "tokenizer")

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(tokenizer_save_path, exist_ok=True)

# Save model and tokenizer to Google Drive
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)


# 13. Inference with Dequantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./phi2-qlora-adapter")
model = model.merge_and_unload()
model.eval()

# 14. Prompt inference
prompt = "What is the future of AI?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
