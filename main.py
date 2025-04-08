import os
import subprocess
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
@torch.inference_mode()
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/CodeQwen1.5-7B-Chat",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
    return model, tokenizer

model, tokenizer = load_model()

# Load dependencies
from dependencies import GPP_PATH, JAVAC_PATH, JAVA_PATH

# Code generation
def generate_code(prompt, language, max_tokens=512):
    messages = [{'role': 'user', 'content': f"Write {language} code for: {prompt}"}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=max_tokens, do_sample=False, top_k=50, top_p=0.95,
                             num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def extract_code(result, language):
    try:
        code_blocks = result.split("```")
        language_markers = {"python": ["python"], "java": ["java"], "c++": ["cpp", "c++"]}
        markers = language_markers.get(language.lower(), [language.lower()])

        if len(code_blocks) > 1:
            for i in range(1, len(code_blocks), 2):
                block = code_blocks[i]
                block_start = block.split('\n', 1)[0].lower() if '\n' in block else block.lower()
                if any(marker in block_start for marker in markers):
                    return block.split('\n', 1)[1].strip(), True
            for i in range(1, len(code_blocks), 2):
                block = code_blocks[i]
                block_start = block.split('\n', 1)[0].lower() if '\n' in block else block.lower()
                if not any(shell in block_start for shell in ["bash", "shell", "sh", "cmd", "powershell"]):
                    return block.split('\n', 1)[1].strip(), True
        return "No code block found in the response.", False
    except Exception as e:
        return f"Error extracting code: {str(e)}", False

def run_subprocess(cmd, timeout=10):
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GPP_PATH)}:{os.path.dirname(JAVAC_PATH)}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{os.path.dirname(GPP_PATH)}/../lib64:" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

def save_and_execute_code(code, language):
    try:
        os.makedirs("generated_code", exist_ok=True)
        file_path = None

        if language.lower() == "python":
            file_path = "generated_code/generated_script.py"
            with open(file_path, "w") as f:
                f.write(code)
            result = run_subprocess(["python", file_path])

        elif language.lower() == "java":
            class_name = "Main"
            for line in code.split("\n"):
                if "class " in line:
                    class_name = line.split("class ")[1].split()[0]
                    break
            file_path = f"generated_code/{class_name}.java"
            with open(file_path, "w") as f:
                f.write(code)
            compile_result = run_subprocess([JAVAC_PATH, file_path])
            if compile_result.returncode != 0:
                return file_path, False, f"Compilation Error:\n{compile_result.stderr}"
            result = run_subprocess([JAVA_PATH, "-cp", "generated_code", class_name])

        elif language.lower() == "c++":
            file_path = "generated_code/program.cpp"
            binary_path = "generated_code/program_exec"
            with open(file_path, "w") as f:
                f.write(code)
            compile_result = run_subprocess([GPP_PATH, file_path, "-o", binary_path])
            if compile_result.returncode != 0:
                return file_path, False, f"Compilation Error:\n{compile_result.stderr}"
            result = run_subprocess([binary_path])

        else:
            return None, False, f"Unsupported language: {language}"

        if result.returncode == 0:
            return file_path, True, result.stdout
        else:
            return file_path, False, f"Execution Error:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return file_path, False, "Execution timed out after 10 seconds"
    except Exception as e:
        return None, False, f"Error saving or executing code: {str(e)}"

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
    gr.Markdown("<h1><center>🤖 MultiLanguage Coder AI<center><h1>")

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
