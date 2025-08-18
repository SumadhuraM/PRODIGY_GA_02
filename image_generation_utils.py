"""
Image Generation Utilities
Common utilities for Stable Diffusion image generation
"""

import os
import json
from PIL import Image
import torch
from datetime import datetime

def setup_output_directory(base_dir="generated_images"):
    """
    Create and setup output directory structure
    
    Args:
        base_dir (str): Base directory for outputs
    
    Returns:
        dict: Dictionary with paths for different output types
    """
    paths = {
        "base": base_dir,
        "images": os.path.join(base_dir, "images"),
        "metadata": os.path.join(base_dir, "metadata"),
        "thumbnails": os.path.join(base_dir, "thumbnails")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def save_image_with_metadata(image, prompt, output_dir, metadata=None):
    """
    Save image with associated metadata
    
    Args:
        image (PIL.Image): Generated image
        prompt (str): Generation prompt
        output_dir (str): Output directory
        metadata (dict): Additional metadata
    
    Returns:
        str: Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sd_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    image.save(filepath)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        "prompt": prompt,
        "timestamp": timestamp,
        "filename": filename,
        "filepath": filepath
    })
    
    metadata_filename = f"metadata_{timestamp}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath

def create_thumbnail(image, size=(256, 256), output_dir="thumbnails"):
    """
    Create thumbnail of generated image
    
    Args:
        image (PIL.Image): Original image
        size (tuple): Thumbnail size
        output_dir (str): Directory for thumbnails
    
    Returns:
        str: Path to thumbnail
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thumbnail_filename = f"thumb_{timestamp}.png"
    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
    
    thumbnail.save(thumbnail_path)
    return thumbnail_path

def get_device_info():
    """
    Get information about available compute devices
    
    Returns:
        dict: Device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    }
    
    if torch.cuda.is_available():
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
    
    return info

def validate_parameters(height, width, guidance_scale, num_inference_steps):
    """
    Validate generation parameters
    
    Args:
        height (int): Image height
        width (int): Image width
        guidance_scale (float): Guidance scale
        num_inference_steps (int): Number of inference steps
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if height % 8 != 0 or width % 8 != 0:
        return False, "Height and width must be multiples of 8"
    
    if height < 64 or width < 64:
        return False, "Height and width must be at least 64 pixels"
    
    if height > 2048 or width > 2048:
        return False, "Height and width must not exceed 2048 pixels"
    
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        return False, "Guidance scale must be between 1.0 and 20.0"
    
    if num_inference_steps < 1 or num_inference_steps > 100:
        return False, "Number of inference steps must be between 1 and 100"
    
    return True, ""

def load_sample_prompts(filepath="sample_prompts_images.txt"):
    """
    Load sample prompts from file
    
    Args:
        filepath (str): Path to prompts file
    
    Returns:
        list: List of sample prompts
    """
    if not os.path.exists(filepath):
        # Default sample prompts
        return [
            "A serene landscape with mountains and a lake at sunset",
            "A futuristic city with flying cars and neon lights",
            "A cute robot playing with a cat in a garden",
            "A magical forest with glowing mushrooms and fairies",
            "A steampunk airship flying over Victorian London",
            "A cyberpunk samurai in a neon-lit alley",
            "A peaceful Japanese garden with cherry blossoms",
            "A majestic dragon soaring through clouds",
            "A cozy cottage in a snowy forest",
            "An underwater city with bioluminescent creatures"
        ]
    
    with open(filepath, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts

def format_generation_info(prompt, negative_prompt, num_images, steps, guidance, size, seed, duration):
    """
    Format generation information for display
    
    Args:
        prompt (str): Generation prompt
        negative_prompt (str): Negative prompt
        num_images (int): Number of images
        steps (int): Inference steps
        guidance (float): Guidance scale
        size (tuple): Image dimensions
        seed (int): Random seed
        duration (float): Generation duration
    
    Returns:
        str: Formatted information string
    """
    info = f"""
Generation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Prompt: {prompt}
"""
    
    if negative_prompt:
        info += f"🚫 Negative: {negative_prompt}\n"
    
    info += f"""
📊 Parameters:
   • Images: {num_images}
   • Steps: {steps}
   • Guidance: {guidance}
   • Size: {size[0]}×{size[1]}
   • Seed: {seed or 'Random'}
   • Duration: {duration:.2f}s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return info
