import gradio as gr
from utils import generate_code, extract_code, save_and_execute_code

def process_prompt(prompt, language, max_tokens):
    model_response = generate_code(prompt, language, max_tokens)
    code, found = extract_code(model_response, language)
    if not found:
        return model_response, "No code extracted", [("No executable code found in response.", "error")], language
    file_path, success, output = save_and_execute_code(code, language)
    highlighted_output = [(line, "error" if "error" in line.lower() or "exception" in line.lower() else "output") for line in output.split('\n')]
    return model_response, code, highlighted_output, language


# Gradio UI
with gr.Blocks(title="MultiLanguage Coder AI") as app:
    gr.Markdown("<h1><center>🤖 CodeGenie 🤖<center><h1>")

    textbox = gr.Textbox(label="Enter code description", lines=3)
    slider = gr.Slider(label="Context Length", minimum=100, maximum=2048, value=512, step=32)

    with gr.Row():
        button_python = gr.Button("Python")
        button_java = gr.Button("Java")
        button_cpp = gr.Button("C++")

    language_state = gr.State("Python")  # This remembers the last selected language

    with gr.Row():
        with gr.Column():
            markdown = gr.Markdown()
            highlightedtext = gr.HighlightedText(label="Execution Output", combine_adjacent=True)
        with gr.Column():
            code_display = gr.Code()
            button_run = gr.Button("Run Code")


    def generate_handler(prompt, max_tokens, language):
        return process_prompt(prompt, language, max_tokens)

    button_python.click(fn=lambda p, m: generate_handler(p, m, "Python"),
                        inputs=[textbox, slider],
                        outputs=[markdown, code_display, highlightedtext, language_state])
    
    button_java.click(fn=lambda p, m: generate_handler(p, m, "Java"),
                      inputs=[textbox, slider],
                      outputs=[markdown, code_display, highlightedtext, language_state])
    
    button_cpp.click(fn=lambda p, m: generate_handler(p, m, "C++"),
                     inputs=[textbox, slider],
                     outputs=[markdown, code_display, highlightedtext, language_state])

    @button_run.click(inputs=[code_display, slider, language_state],
                      outputs=[highlightedtext])
    

    def run_code_handler(code, _, language):
        _, success, output = save_and_execute_code(code, language)
        return [(line, "error" if "error" in line.lower() or "exception" in line.lower() else "output") for line in output.split("\n")]

app.launch(share=True)
