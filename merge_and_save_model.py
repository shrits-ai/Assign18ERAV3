import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base model and adapter paths
base_model_name = "microsoft/phi-2"  # Pull from HF Hub directly
adapter_path = "Shriti09/Microsoft-Phi-QLora"  # Update with your Hugging Face repo path

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# Load the LoRA adapter
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the LoRA adapter into the base model
merged_model = adapter_model.merge_and_unload()

# Save the merged model to Hugging Face (space storage)
merged_model.save_pretrained("./merged_model")
