"""
Gradio App for Text-to-Image Generation on Hugging Face Spaces

This app provides a user-friendly interface for generating images from text
using fine-tuned Stable Diffusion models.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path

# Import our custom modules
from utils.model_utils import load_base_model, load_lora_weights, optimize_model_for_inference
from utils.data_utils import create_sample_dataset

# Global variables
model = None
model_loaded = False
current_device = "cpu"

def load_model(model_type, base_model_path, lora_path=None, device="cpu"):
    """Load the model into memory."""
    global model, model_loaded, current_device
    
    try:
        # Load base model
        if model_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            base_model = StableDiffusionXLPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
        else:
            from diffusers import StableDiffusionPipeline
            base_model = StableDiffusionPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        
        # Load LoRA weights if specified
        if lora_path and lora_path != "None" and os.path.exists(lora_path):
            base_model = load_lora_weights(base_model, lora_path)
        
        # Move to device and optimize
        base_model = base_model.to(device)
        if device == "cuda":
            base_model = optimize_model_for_inference(base_model, device)
        
        model = base_model
        model_loaded = True
        current_device = device
        
        return f"✅ Model loaded successfully on {device.upper()}!"
        
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}"

def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, height, width, seed):
    """Generate image using the loaded model."""
    global model, model_loaded
    
    if not model_loaded or model is None:
        return None, "Please load a model first!"
    
    try:
        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width
            )
        
        # Get the generated image
        image = result.images[0]
        
        # Save image with timestamp
        timestamp = int(time.time())
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        filename = f"generated_{timestamp}_{safe_prompt}.png"
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", filename)
        image.save(output_path)
        
        return image, f"✅ Image generated successfully! Saved as {filename}"
        
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

def create_sample_data():
    """Create sample dataset for testing."""
    try:
        create_sample_dataset("sample_data", num_samples=5)
        return "✅ Sample dataset created successfully in 'sample_data' directory!"
    except Exception as e:
        return f"❌ Failed to create sample dataset: {str(e)}"

def get_model_info():
    """Get information about the loaded model."""
    global model, model_loaded, current_device
    
    if not model_loaded:
        return "No model loaded"
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = f"""
        **Model Information:**
        - Device: {current_device.upper()}
        - Total Parameters: {total_params:,}
        - Trainable Parameters: {trainable_params:,}
        - Model Type: {'SDXL' if hasattr(model, 'unet') and hasattr(model.unet, 'config') and 'xl' in model.unet.config.model_type.lower() else 'SD 2.1/1.5'}
        """
        return info
        
    except Exception as e:
        return f"Error getting model info: {str(e)}"

# Create the Gradio interface
with gr.Blocks(
    title="Text-to-Image Generation",
    theme=gr.themes.Soft(),
    css="""
        .main-header { text-align: center; font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
        .sub-header { font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-bottom: 0.5rem; }
        .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    """
) as demo:
    
    gr.HTML('<h1 class="main-header">🎨 Text-to-Image Generation</h1>')
    gr.Markdown("Generate high-quality images from text descriptions using fine-tuned Stable Diffusion models.")
    
    with gr.Tabs():
        # Model Loading Tab
        with gr.Tab("🚀 Model Setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<div class="sub-header">Model Configuration</div>')
                    
                    model_type = gr.Dropdown(
                        choices=["sdxl", "sd2.1", "sd1.5"],
                        value="sdxl",
                        label="Model Type",
                        info="Select the base model type"
                    )
                    
                    base_model_path = gr.Textbox(
                        value="stabilityai/stable-diffusion-xl-base-1.0",
                        label="Base Model Path",
                        info="Path to base model or HuggingFace model ID"
                    )
                    
                    lora_path = gr.Textbox(
                        value="",
                        label="LoRA Weights Path (Optional)",
                        info="Path to fine-tuned LoRA weights"
                    )
                    
                    device = gr.Dropdown(
                        choices=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
                        value="cpu",
                        label="Device",
                        info="Select device for inference"
                    )
                    
                    load_btn = gr.Button("🚀 Load Model", variant="primary")
                    load_status = gr.Textbox(label="Load Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown('<div class="sub-header">Quick Actions</div>')
                    
                    create_sample_btn = gr.Button("📁 Create Sample Dataset")
                    sample_status = gr.Textbox(label="Sample Dataset Status", interactive=False)
                    
                    model_info_btn = gr.Button("ℹ️ Get Model Info")
                    model_info_display = gr.Markdown(label="Model Information")
        
        # Image Generation Tab
        with gr.Tab("🎯 Generate Images"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<div class="sub-header">Generation Parameters</div>')
                    
                    prompt = gr.Textbox(
                        value="a beautiful landscape with mountains and lake, high quality, detailed, photorealistic",
                        label="Prompt",
                        lines=3,
                        info="Describe the image you want to generate"
                    )
                    
                    negative_prompt = gr.Textbox(
                        value="blurry, low quality, distorted, ugly, bad anatomy",
                        label="Negative Prompt",
                        lines=2,
                        info="Describe what you don't want in the image"
                    )
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                            info="How closely to follow the prompt"
                        )
                        
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Inference Steps",
                            info="Number of denoising steps"
                        )
                    
                    with gr.Row():
                        height = gr.Dropdown(
                            choices=[512, 768, 1024, 1280],
                            value=1024,
                            label="Height"
                        )
                        
                        width = gr.Dropdown(
                            choices=[512, 768, 1024, 1280],
                            value=1024,
                            label="Width"
                        )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed",
                        info="Random seed (-1 for random)"
                    )
                    
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    generation_status = gr.Textbox(label="Generation Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown('<div class="sub-header">Generated Image</div>')
                    
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil",
                        height=512
                    )
                    
                    download_btn = gr.File(
                        label="Download Image",
                        file_count="single"
                    )
        
        # About Tab
        with gr.Tab("📚 About"):
            gr.Markdown("""
            ## 🚀 Text-to-Image Generation with Fine-Tuning
            
            This application demonstrates the power of fine-tuned Stable Diffusion models for generating high-quality images from text descriptions.
            
            ### ✨ Features
            
            - **Multiple Model Support**: SDXL, SD 2.1, and SD 1.5
            - **LoRA Fine-tuning**: Load custom fine-tuned weights
            - **Flexible Generation**: Customizable parameters for optimal results
            - **High Quality Output**: Generate images up to 1280x1280 resolution
            
            ### 🛠️ How to Use
            
            1. **Model Setup**: Load your preferred base model and optional LoRA weights
            2. **Configure Parameters**: Set generation parameters like guidance scale and steps
            3. **Generate**: Provide a detailed prompt and generate your image
            4. **Download**: Save your generated images for further use
            
            ### 💡 Tips for Better Results
            
            - Use detailed, descriptive prompts
            - Experiment with different guidance scales (7-10 is usually good)
            - Higher inference steps = better quality but slower generation
            - Use negative prompts to avoid unwanted elements
            
            ### 🔧 Technical Details
            
            - Built with PyTorch and Diffusers
            - LoRA (Low-Rank Adaptation) support for efficient fine-tuning
            - Optimized for both CPU and GPU inference
            - Support for various image resolutions
            
            ### 📖 Learn More
            
            - [Project Repository](https://github.com/whoishmk/text-to-image-generation)
            - [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
            - [LoRA Paper](https://arxiv.org/abs/2106.09685)
            
            ---
            
            **Built with ❤️ using Gradio, PyTorch, and Stable Diffusion**
            """)
    
    # Event handlers
    load_btn.click(
        fn=load_model,
        inputs=[model_type, base_model_path, lora_path, device],
        outputs=load_status
    )
    
    create_sample_btn.click(
        fn=create_sample_data,
        outputs=sample_status
    )
    
    model_info_btn.click(
        fn=get_model_info,
        outputs=model_info_display
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, guidance_scale, num_inference_steps, height, width, seed],
        outputs=[output_image, generation_status]
    )
    
    # Update download button when image is generated
    output_image.change(
        fn=lambda img: gr.File.update(value=img) if img else None,
        inputs=output_image,
        outputs=download_btn
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
