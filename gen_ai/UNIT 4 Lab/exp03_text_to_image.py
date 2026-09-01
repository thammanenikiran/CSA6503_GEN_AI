"""
Unit 4 - Experiment 3: Text-to-Image Generation of an Engineering Object
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Generate an engineering-related image, such as a bridge or robotic system,
    from a suitable text prompt using a pre-trained text-to-image model.

MODEL:
    stabilityai/sd-turbo - a distilled Stable Diffusion model that produces a
    good image in only 1-4 denoising steps, so it also runs on a CPU laptop.

SETUP:
    pip install -r requirements.txt

RUN:
    python exp03_text_to_image.py
    python exp03_text_to_image.py "a robotic arm welding a car chassis"
"""

import sys
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

MODEL = "stabilityai/sd-turbo"
OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_PROMPT = (
    "a modern cable-stayed bridge over a wide river at sunrise, steel pylons, "
    "civil engineering photograph, highly detailed, 8k"
)


def load_pipeline():
    """Load the pre-trained diffusion pipeline on GPU if available, else CPU."""
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Loading {MODEL} on {device} (first run downloads ~2.5 GB)...")

    pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def generate(pipe, prompt, steps=4, seed=42):
    """Run the model and return the generated PIL image."""
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,   # sd-turbo is trained to work without CFG
        generator=generator,
    )
    return result.images[0]


def save(image, prompt):
    """Save the image into outputs/ with a timestamped, readable file name."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in prompt.lower())[:40].strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"exp03_{slug}_{stamp}.png"
    image.save(path)
    return path


def main():
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT

    print("\n=== Text-to-Image Generation ===")
    print(f"Prompt: {prompt}\n")

    pipe = load_pipeline()
    image = generate(pipe, prompt)
    path = save(image, prompt)

    print(f"\nImage saved to: {path}")
    print(f"Size: {image.size[0]} x {image.size[1]} pixels")

    try:
        image.show()   # opens in the default image viewer
    except Exception:
        pass


if __name__ == "__main__":
    main()
