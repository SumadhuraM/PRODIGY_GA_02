#!/usr/bin/env python3
"""
Stable Diffusion Interactive Demo - Streamlit Web Interface
Task 2: Image Generation with Stable Diffusion (Interactive Web Demo)
"""

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from datetime import datetime
import json
from image_generation_utils import (
    validate_parameters, 
    get_device_info, 
    save_image_with_metadata,
    format_generation_info,
    load_sample_prompts
)

# Page configuration
st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        transform: translateY(-2px);
    }
    .generated-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generator' not in st.session_state:
    st.session_state.generator = None

@st.cache_resource
def load_stable_diffusion_pipeline(model_name="runwayml/stable-diffusion-v1-5"):
    """Load and cache the Stable Diffusion pipeline"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Use DPM-Solver for faster inference
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        
        pipe = pipe.to(device)
        
        # Memory optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        return pipe, device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def main():
    st.title("🎨 Stable Diffusion Image Generator")
    st.markdown("Generate stunning AI images with Stable Diffusion")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"]
        )
        
        # Device info
        device_info = get_device_info()
        st.info(f"Device: {device_info['device_name']}")
        
        # Load model button
        if st.button("Load Model", type="primary"):
            with st.spinner("Loading Stable Diffusion pipeline..."):
                pipe, device = load_stable_diffusion_pipeline(model_name)
                if pipe:
                    st.session_state.generator = pipe
                    st.session_state.device = device
                    st.success("Model loaded successfully!")
        
        # Sample prompts
        st.header("💡 Sample Prompts")
        sample_prompts = load_sample_prompts()
        selected_prompt = st.selectbox("Choose a sample prompt:", [""] + sample_prompts)
        
        if selected_prompt:
            st.session_state.selected_prompt = selected_prompt
    
    # Main content area
    if st.session_state.generator is None:
        st.warning("⚠️ Please load the model from the sidebar first!")
        st.info("Click 'Load Model' in the sidebar to get started.")
        return
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Generation Settings")
        
        # Prompt input
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.get("selected_prompt", ""),
            height=100,
            placeholder="Describe the image you want to generate..."
        )
        
        # Negative prompt
        negative_prompt = st.text_area(
            "Negative Prompt",
            height=50,
            placeholder="What to avoid in the image..."
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col3, col4 = st.columns(2)
            
            with col3:
                num_images = st.slider("Number of Images", 1, 4, 1)
                num_inference_steps = st.slider("Inference Steps", 10, 100, 50)
                guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
            
            with col4:
                height = st.select_slider("Height", options=[64, 128, 256, 384, 512, 640, 768, 896, 1024], value=512)
                width = st.select_slider("Width", options=[64, 128, 256, 384, 512, 640, 768, 896, 1024], value=512)
                seed = st.number_input("Seed (0 for random)", min_value=0, max_value=999999, value=0)
        
        # Validate parameters
        is_valid, error_msg = validate_parameters(height, width, guidance_scale, num_inference_steps)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Generate button
        if st.button("🚀 Generate Images", type="primary", use_container_width=True):
            with st.spinner("Generating images..."):
                start_time = datetime.now()
                
                try:
                    # Generate images
                    if seed == 0:
                        seed = None
                    
                    result = st.session_state.generator(
                        prompt=[prompt] * num_images,
                        negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=torch.Generator(st.session_state.device).manual_seed(seed) if seed else None
                    )
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    # Save images and display
                    st.session_state.generated_images = []
                    
                    for i, image in enumerate(result.images):
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sd_web_{timestamp}_{i+1}.png"
                        filepath = os.path.join("generated_images", filename)
                        os.makedirs("generated_images", exist_ok=True)
                        image.save(filepath)
                        
                        st.session_state.generated_images.append({
                            "image": image,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "parameters": {
                                "steps": num_inference_steps,
                                "guidance": guidance_scale,
                                "size": (height, width),
                                "seed": seed
                            },
                            "filepath": filepath
                        })
                    
                    # Display generation info
                    info = format_generation_info(
                        prompt, negative_prompt, num_images, num_inference_steps,
                        guidance_scale, (height, width), seed, duration
                    )
                    st.success(info)
                    
                except Exception as e:
                    st.error(f"Error generating images: {e}")
    
    with col2:
        st.header("📸 Generated Images")
        
        if st.session_state.generated_images:
            for idx, img_data in enumerate(st.session_state.generated_images):
                st.image(img_data["image"], caption=f"Image {idx+1}", use_column_width=True)
                
                # Download button
                with open(img_data["filepath"], "rb") as f:
                    st.download_button(
                        label=f"Download Image {idx+1}",
                        data=f.read(),
                        file_name=os.path.basename(img_data["filepath"]),
                        mime="image/png",
                        key=f"download_{idx}"
                    )
        else:
            st.info("Generated images will appear here")
    
    # History section
    if st.session_state.generated_images:
        st.header("📜 Generation History")
        
        for idx, img_data in enumerate(st.session_state.generated_images):
            with st.expander(f"Image {idx+1} - {img_data['prompt'][:50]}..."):
                st.image(img_data["image"], use_column_width=True)
                
                col5, col6 = st.columns(2)
                with col5:
                    st.write("**Prompt:**", img_data["prompt"])
                    st.write("**Parameters:**")
                    st.json(img_data["parameters"])
                
                with col6:
                    if img_data["negative_prompt"]:
                        st.write("**Negative Prompt:**", img_data["negative_prompt"])
                    
                    # Download button
                    with open(img_data["filepath"], "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=os.path.basename(img_data["filepath"]),
                            mime="image/png",
                            key=f"history_download_{idx}"
                        )

if __name__ == "__main__":
    main()
