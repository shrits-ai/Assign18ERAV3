import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base model and adapter paths
base_model_name = "microsoft/phi-2"  # Pull from HF Hub directly
adapter_path = "Shriti09/Microsoft-Phi-QLora"  # Update with your Hugging Face repo path

print("🔧 Loading base model...")
# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

print("🔧 Loading LoRA adapter...")
# Load the LoRA adapter
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

print("🔗 Merging adapter into base model...")
# Merge adapter into the base model
merged_model = adapter_model.merge_and_unload()
merged_model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("✅ Model ready for inference!")

# Text generation function
def generate_text(prompt):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and return the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>🧠 Phi-2 QLoRA Text Generator</h1>")
    
    # Textbox for user input
    prompt = gr.Textbox(label="Enter your prompt:", lines=2)
    
    # Output textbox for generated text
    output = gr.Textbox(label="Generated text:", lines=5)
    
    # Button to trigger text generation
    generate_button = gr.Button("Generate Text")

    # Set the button action to generate text
    generate_button.click(generate_text, inputs=prompt, outputs=output)

# Launch the app
demo.launch(share=True)
