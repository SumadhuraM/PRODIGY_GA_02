#!/usr/bin/env python3
"""
Stable Diffusion CLI - Basic Command Line Interface for Image Generation
Task 2: Image Generation with Stable Diffusion
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime

def generate_image(prompt, output_dir="generated_images", num_inference_steps=50, guidance_scale=7.5, seed=None):
    """
    Generate an image using Stable Diffusion
    
    Args:
        prompt (str): Text prompt for image generation
        output_dir (str): Directory to save generated images
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): How closely to follow the prompt
        seed (int): Random seed for reproducibility
    
    Returns:
        str: Path to the generated image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Generate image
    print(f"Generating image for prompt: '{prompt}'")
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
    
    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sd_image_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    print(f"Image saved to: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output-dir", default="generated_images", help="Output directory for images")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        generate_image(
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error generating image: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
