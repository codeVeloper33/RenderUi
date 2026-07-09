import os
import requests
import gradio as gr

# =====================================================================
# CONFIGURATION (Change these to match your account)
# =====================================================================
# Replace with your actual Hugging Face username and the Model Repository name
HF_MODEL_REPO = "BabalolaEpo/physics-model"

# Replace with your free Hugging Face Access Token (Read token)
HF_API_TOKEN = "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"

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# =====================================================================
# CORE AI INFERENCE LOGIC
# =====================================================================
def ask_physics_agent(user_message, history):
    # This keeps your exact Alpaca template formatting intact
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""
    formatted_prompt = alpaca_prompt.format(user_message)
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        # Handle model loading or processing states smoothly
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", "No response text found.")
        elif isinstance(output, dict):
            if "generated_text" in output:
                return output["generated_text"]
            elif "error" in output:
                return f"Hugging Face Status: {output['error']}"
        
        return f"Unexpected API output format: {str(output)}"
        
    except Exception as e:
        return f"Connection error: {str(e)}"

# =====================================================================
# GRADIO INTERFACE CONFIGURATION
# =====================================================================
demo = gr.ChatInterface(
    fn=ask_physics_agent, 
    title="Custom Physics AI Tutor", 
    description="Your fine-tuned model running 24/7 on free cloud hosting.",
    theme="soft"
)

if __name__ == "__main__":
    # Render requires the app to look specifically at 0.0.0.0 and port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
