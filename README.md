# 🎨 Stable Diffusion Image Generation - Task2

Generate stunning AI images with Stable Diffusion using multiple interfaces: CLI, enhanced CLI, and interactive web demo.

## ✨ Features

- **High-quality AI image generation** with Stable Diffusion
- **Multiple generation modes**: Basic CLI, Enhanced CLI, Interactive Web
- **Advanced parameters**: steps, guidance scale, negative prompts
- **GPU acceleration** support for faster generation
- **Batch generation** capabilities
- **Metadata saving** for reproducibility

## 📂 Task 2 Files

- `stable_diffusion_cli.py` – Basic CLI for image generation
- `stable_diffusion_enhanced.py` – Advanced CLI with full parameters
- `stable_diffusion_interactive_demo.py` – Professional web interface
- `image_generation_utils.py` – Common utilities for image handling
- `sample_prompts_images.txt` – Sample prompts for image generation
- `requirements.txt` – Updated with Stable Diffusion dependencies

## ⚡ Setup & Usage

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic CLI Usage
```bash
python stable_diffusion_cli.py "A beautiful sunset over mountains"
```

### Enhanced CLI Parameters
```bash
python stable_diffusion_enhanced.py "A cyberpunk city" --num-images 2 --steps 30 --guidance 8.0 --seed 42
```

### Interactive Web Demo
```bash
streamlit run stable_diffusion_interactive_demo.py
```

## 🎯 Available Parameters

### Basic CLI
- `prompt`: Text description of desired image

### Enhanced CLI
- `--negative-prompt`: Specify what to avoid
- `--num-images`: Generate multiple images (1-4)
- `--steps`: Inference steps (10-100)
- `--guidance`: Guidance scale (1.0-20.0)
- `--height/--width`: Image dimensions (64-1024)
- `--seed`: For reproducible results
- `--output-dir`: Custom output directory

### Web Interface Features
- Real-time generation with progress indicators
- Parameter sliders for fine-tuning
- Sample prompts dropdown
- Download buttons for generated images
- Generation history tracking
- GPU/CPU device info

## 🚀 Quick Start Examples

```bash
# Generate a single image
python stable_diffusion_cli.py "A peaceful Japanese garden with cherry blossoms"

# Generate multiple images with parameters
python stable_diffusion_enhanced.py "A futuristic city" --num-images 3 --steps 40 --guidance 9.0 --seed 42

# Launch interactive web demo
streamlit run stable_diffusion_interactive_demo.py
```

## 📝 Sample Prompts
Check `sample_prompts_images.txt` for ready-to-use prompts covering:
- Nature & Landscapes
- Fantasy & Sci-Fi
- Characters & Portraits
- Art Styles
- Architecture
- Animals & Creatures
- Abstract & Artistic

## ⚙️ Technical Features
- **Memory optimization** with attention slicing
- **GPU acceleration** when available
- **Batch processing** capabilities
- **Metadata saving** with generation parameters
- **Error handling** and validation
