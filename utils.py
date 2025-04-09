import os
import torch
import subprocess
from dependencies import GPP_PATH, JAVAC_PATH, JAVA_PATH # It conatined path variables for compilers 
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
@torch.inference_mode()
def load_model():

    """Loads the model for Code Generation"""

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/CodeQwen1.5-7B-Chat",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
    return model, tokenizer

model, tokenizer = load_model()

# Code generation
def generate_code(prompt, language, max_tokens=512):

    """Generates Output from LLM for User Query"""

    messages = [{'role': 'user', 'content': f"Write {language} code for: {prompt}"}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=max_tokens, do_sample=False, top_k=50, top_p=0.95,
                             num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def extract_code(result, language):

    """Extracts code from LLM output for Futher Modification and Execution"""

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

    """Helper Function to load all the Dependencies"""

    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GPP_PATH)}:{os.path.dirname(JAVAC_PATH)}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{os.path.dirname(GPP_PATH)}/../lib64:" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

def save_and_execute_code(code, language):

    """Save Code and Compiled Output, Execute Code"""

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
