import os
import gradio as gr
from huggingface_hub import InferenceClient

# =====================================================================
# CONFIGURATION
# =====================================================================
# Replace with your actual Hugging Face details
# Change this line from "physics-model" to your actual repository name:
HF_MODEL_REPO = "BabalolaEpo/physics-lora-weights"
HF_API_TOKEN = "hf_your_token_here"  # Paste your copied token here

# Initialize the official client (handles DNS and routing automatically)
client = InferenceClient(model=HF_MODEL_REPO, token=HF_API_TOKEN)

# =====================================================================
# CORE AI INFERENCE LOGIC
# =====================================================================
def ask_physics_agent(user_message, history):
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""
    formatted_prompt = alpaca_prompt.format(user_message)
    
    try:
        # Using the official client method to avoid hardcoded URL errors
        response = client.text_generation(
            formatted_prompt,
            max_new_tokens=512,
            temperature=0.3
        )
        return response
        
    except Exception as e:
        return f"Backend Connection Error: {str(e)}\n\n(If this persists, make sure your weights folder is completely uploaded to the repository!)"

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
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
