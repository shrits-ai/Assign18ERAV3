import gradio as gr
import subprocess

def merge_model():
    subprocess.run(["python", "merge_and_save_model.py"], check=True)
    return "Model merged and saved successfully!"

with gr.Blocks() as demo:
    gr.Markdown("<h1>🧠 Phi-2 QLoRA Model Merger</h1>")
    merge_button = gr.Button("Merge Model")
    output = gr.Textbox(label="Merge Status")

    merge_button.click(merge_model, [], output)

demo.launch(share=True)
