#!/usr/bin/env python3
"""
Stable Diffusion Enhanced - Advanced Image Generation with Multiple Parameters
Task 2: Image Generation with Stable Diffusion (Enhanced Version)
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import json

class StableDiffusionGenerator:
    """Enhanced Stable Diffusion image generator with advanced features"""
    
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        """
        Initialize the Stable Diffusion generator
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        
    def load_pipeline(self):
        """Load the Stable Diffusion pipeline"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Load pipeline with optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Use DPM-Solver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, using standard attention")
    
    def generate_images(self, 
                       prompt, 
                       negative_prompt=None,
                       num_images=1,
                       num_inference_steps=50,
                       guidance_scale=7.5,
                       height=512,
                       width=512,
                       seed=None,
                       output_dir="generated_images"):
        """
        Generate multiple images with advanced parameters
        
        Args:
            prompt (str): Positive prompt for image generation
            negative_prompt (str): Negative prompt to avoid certain features
            num_images (int): Number of images to generate
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            height (int): Image height (multiple of 8)
            width (int): Image width (multiple of 8)
            seed (int): Random seed for reproducibility
            output_dir (str): Output directory
        
        Returns:
            list: List of generated image file paths
        """
        if self.pipe is None:
            self.load_pipeline()
        
        # Validate dimensions
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and width must be multiples of 8")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Generate images
        print(f"Generating {num_images} image(s)...")
        print(f"Prompt: {prompt}")
        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")
        
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            result = self.pipe(
                prompt=[prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
        
        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        for i, image in enumerate(result.images):
            filename = f"sd_enhanced_{timestamp}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_files.append(filepath)
            print(f"Saved: {filepath}")
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": num_images,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "model": self.model_name,
            "timestamp": timestamp,
            "generated_files": saved_files
        }
        
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return saved_files

def main():
    parser = argparse.ArgumentParser(description="Enhanced Stable Diffusion image generator")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", help="Negative prompt to avoid certain features")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="generated_images", help="Output directory")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model name")
    
    args = parser.parse_args()
    
    try:
        generator = StableDiffusionGenerator(model_name=args.model)
        generator.generate_images(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error generating images: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
