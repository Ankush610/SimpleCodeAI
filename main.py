import os
import torch
import gradio as gr
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
@torch.inference_mode()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).cuda()
    return model, tokenizer

print("Loading DeepSeek Coder model...")
model, tokenizer = load_model()
print("Model loaded successfully!")

def generate_code(prompt, max_tokens=512):
    """Generate code using the DeepSeek Coder model"""
    messages = [
        {'role': 'user', 'content': f"Write Python code for: {prompt}"}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs, 
        max_new_tokens=max_tokens, 
        do_sample=False, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return result

def extract_python_code(result):
    """Extract Python code from the model's response"""
    try:
        # Try to extract code between triple backticks
        code_blocks = result.split("```")
        python_code = None
        
        # First look specifically for python blocks
        if len(code_blocks) > 1:
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    block = code_blocks[i]
                    # Check if this block is marked as Python
                    if block.lower().startswith("python"):
                        python_code = block[6:].strip()  # Remove 'python' and strip whitespace
                        return python_code, True
            
            # If no Python block found, take the first code block that's not a bash/shell command
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    block = code_blocks[i]
                    block_start = block.split('\n', 1)[0].lower() if '\n' in block else block.lower()
                    
                    # Skip bash/shell blocks
                    if not any(shell in block_start for shell in ["bash", "shell", "sh", "cmd", "powershell"]):
                        # If no language is specified or it's not a shell script, assume it's Python
                        if block_start and block_start.strip() and '\n' in block:
                            # If it starts with a language identifier, remove it
                            python_code = block.split('\n', 1)[1].strip()
                        else:
                            python_code = block.strip()
                        return python_code, True
        
        # If no code blocks with triple backticks or all were shell commands,
        # check if the entire response looks like code
        if "def " in result or "import " in result or "class " in result:
            lines = result.split("\n")
            code_lines = []
            for line in lines:
                if any(keyword in line for keyword in ["def ", "import ", "class ", "print(", "for ", "if ", "while "]):
                    code_lines.append(line)
                elif line.strip() and not line.startswith(('#', '//', '/*')):
                    code_lines.append(line)
            
            if code_lines:
                return "\n".join(code_lines), True
            
        # No valid Python code found
        return result, False
    except Exception as e:
        return f"Error extracting code: {str(e)}", False

def save_and_execute_code(code):
    """Save code to a file and execute it"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs("generated_code", exist_ok=True)
        
        # Save the code to a file
        file_path = os.path.join("generated_code", "generated_script.py")
        with open(file_path, "w") as f:
            f.write(code)
        
        # Execute the code and capture output
        result = subprocess.run(
            ["python", file_path], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        # Return execution results
        if result.returncode == 0:
            execution_output = result.stdout
            return file_path, True, execution_output
        else:
            return file_path, False, f"Execution Error:\n{result.stderr}"
    
    except subprocess.TimeoutExpired:
        return file_path, False, "Execution timed out after 10 seconds"
    except Exception as e:
        return None, False, f"Error saving or executing code: {str(e)}"

def process_prompt(prompt, max_tokens=512):
    """Process the user prompt to generate, extract, save, and execute code"""
    # Generate code using the model
    model_response = generate_code(prompt, max_tokens)
    
    # Extract code from the response
    code, code_found = extract_python_code(model_response)
    
    if code_found:
        # Save and execute the code
        file_path, execution_success, execution_output = save_and_execute_code(code)
        
        if execution_success:
            status = f"✅ Code successfully executed and saved to {file_path}"
        else:
            status = f"⚠️ Code extracted but execution failed. Saved to {file_path}"
        
        return model_response, code, execution_output, status
    else:
        # No code could be extracted
        return model_response, "No Python code could be extracted from the response.", "No code to execute.", "⚠️ No executable Python code found in response."

# Create Gradio interface
with gr.Blocks(title="DeepSeek Coder Python Interface") as app:
    gr.Markdown("# DeepSeek Coder Python Interface")
    gr.Markdown("Enter a prompt describing the Python code you want to generate.")
    
    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Example: write code for list comprehension for filtering with test case",
                lines=3
            )
            max_tokens = gr.Slider(
                label="Maximum Tokens",
                minimum=100,
                maximum=2048,
                value=512,
                step=32
            )
            generate_button = gr.Button("Generate Code")
        
    with gr.Row():
        with gr.Column():
            status_output = gr.Textbox(label="Status")
    
    with gr.Tabs():
        with gr.TabItem("Original Response"):
            model_response_output = gr.Markdown()
        
        with gr.TabItem("Extracted Code"):
            extracted_code_output = gr.Code(language="python", label="Extracted Python Code")
        
        with gr.TabItem("Execution Result"):
            execution_output = gr.Textbox(label="Execution Output", lines=10)
    
    # Use a list of output components
    generate_button.click(
        fn=process_prompt,
        inputs=[prompt_input, max_tokens],
        outputs=[model_response_output, extracted_code_output, execution_output, status_output]
    )

# Launch the interface
if __name__ == "__main__":
    app.launch(share=True)
