"""
Streamlit Web Application for Text-to-Image Generation

This application provides a user-friendly interface for generating images
from text descriptions using fine-tuned Stable Diffusion models.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.model_server import TextToImageModel
from utils.model_utils import load_lora_weights, optimize_model_for_inference

# Page configuration
st.set_page_config(
    page_title="Text-to-Image Generation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .generated-image {
        border: 2px solid #ddd;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []


def load_model():
    """Load the fine-tuned model."""
    try:
        with st.spinner("Loading model..."):
            # Load base model
            model_path = st.session_state.model_path
            model_type = st.session_state.model_type
            
            if model_type == "sdxl":
                from diffusers import StableDiffusionXLPipeline
                model = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
            else:
                from diffusers import StableDiffusionPipeline
                model = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            
            # Load LoRA weights if specified
            if st.session_state.lora_path and st.session_state.lora_path != "None":
                model = load_lora_weights(model, st.session_state.lora_path)
            
            # Optimize for inference
            model = optimize_model_for_inference(model, device=st.session_state.device)
            
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            st.success("Model loaded successfully!")
            return True
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return False


def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, seed):
    """Generate image using the loaded model."""
    if not st.session_state.model_loaded:
        st.error("Please load a model first!")
        return None
    
    try:
        with st.spinner("Generating image..."):
            # Set seed for reproducibility
            if seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate image
            image = st.session_state.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=st.session_state.resolution,
                width=st.session_state.resolution
            ).images[0]
            
            return image
            
    except Exception as e:
        st.error(f"Failed to generate image: {str(e)}")
        return None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎨 Text-to-Image Generation</h1>', unsafe_allow_html=True)
    st.markdown("Generate high-quality images from text descriptions using fine-tuned Stable Diffusion models.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Model selection
        st.subheader("Model Settings")
        model_type = st.selectbox(
            "Model Type",
            ["sdxl", "sd2.1", "sd1.5"],
            help="Select the base model type"
        )
        
        model_path = st.text_input(
            "Base Model Path",
            value="stabilityai/stable-diffusion-xl-base-1.0",
            help="Path to base model or HuggingFace model ID"
        )
        
        lora_path = st.text_input(
            "LoRA Weights Path (Optional)",
            value="None",
            help="Path to fine-tuned LoRA weights"
        )
        
        # Hardware settings
        st.subheader("Hardware Settings")
        device = st.selectbox(
            "Device",
            ["cuda", "cpu", "mps"],
            help="Select device for inference"
        )
        
        resolution = st.selectbox(
            "Image Resolution",
            [512, 768, 1024, 1280],
            help="Select image resolution"
        )
        
        # Load model button
        if st.button("🚀 Load Model", type="primary"):
            st.session_state.model_type = model_type
            st.session_state.model_path = model_path
            st.session_state.lora_path = lora_path if lora_path != "None" else None
            st.session_state.device = device
            st.session_state.resolution = resolution
            
            load_model()
    
    # Main content area
    if st.session_state.model_loaded:
        st.markdown('<h2 class="sub-header">🎯 Image Generation</h2>', unsafe_allow_html=True)
        
        # Generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_area(
                "Prompt",
                value="a beautiful landscape with mountains and lake, high quality, detailed, photorealistic",
                height=100,
                help="Describe the image you want to generate"
            )
            
            negative_prompt = st.text_area(
                "Negative Prompt",
                value="blurry, low quality, distorted, ugly, bad anatomy",
                height=100,
                help="Describe what you don't want in the image"
            )
        
        with col2:
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="How closely to follow the prompt (higher = more adherence)"
            )
            
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Number of denoising steps (higher = better quality, slower)"
            )
            
            seed = st.number_input(
                "Seed",
                value=-1,
                help="Random seed for reproducibility (-1 for random)"
            )
        
        # Generate button
        if st.button("🎨 Generate Image", type="primary"):
            image = generate_image(
                prompt, negative_prompt, guidance_scale, num_inference_steps, seed
            )
            
            if image:
                # Display generated image
                st.markdown('<h3 class="sub-header">✨ Generated Image</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption=f"Prompt: {prompt}", use_column_width=True)
                
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write("**Generation Parameters:**")
                    st.write(f"• Guidance Scale: {guidance_scale}")
                    st.write(f"• Inference Steps: {num_inference_steps}")
                    st.write(f"• Seed: {seed if seed != -1 else 'Random'}")
                    st.write(f"• Resolution: {resolution}x{resolution}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download button
                    img_bytes = image.tobytes()
                    st.download_button(
                        label="📥 Download Image",
                        data=img_bytes,
                        file_name=f"generated_image_{int(time.time())}.png",
                        mime="image/png"
                    )
                
                # Add to history
                st.session_state.generated_images.append({
                    "image": image,
                    "prompt": prompt,
                    "timestamp": time.time(),
                    "parameters": {
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                        "resolution": resolution
                    }
                })
        
        # Generation history
        if st.session_state.generated_images:
            st.markdown('<h3 class="sub-header">📚 Generation History</h3>', unsafe_allow_html=True)
            
            for i, gen_info in enumerate(reversed(st.session_state.generated_images[-5:])):  # Show last 5
                with st.expander(f"Image {len(st.session_state.generated_images) - i} - {gen_info['prompt'][:50]}..."):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(gen_info["image"], caption=gen_info["prompt"])
                    
                    with col2:
                        st.write("**Parameters:**")
                        for param, value in gen_info["parameters"].items():
                            st.write(f"• {param}: {value}")
                        
                        st.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(gen_info['timestamp']))}")
    
    else:
        # Model not loaded state
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 Get Started
        
        1. **Configure Model Settings** in the sidebar
        2. **Load the Model** using the sidebar button
        3. **Generate Images** by providing text prompts
        
        ### 📋 Requirements
        
        - CUDA-compatible GPU (recommended)
        - Sufficient VRAM (8GB+ for SDXL)
        - Fine-tuned model weights (optional)
        
        ### 💡 Tips
        
        - Use detailed, descriptive prompts for better results
        - Experiment with different guidance scales
        - Higher inference steps = better quality but slower generation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, PyTorch, and Stable Diffusion | "
        "[GitHub](https://github.com/yourusername/text-to-image-generation) | "
        "[Documentation](https://github.com/yourusername/text-to-image-generation#readme)"
    )


if __name__ == "__main__":
    main()

