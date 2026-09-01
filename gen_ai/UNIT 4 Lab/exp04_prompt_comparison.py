"""
Unit 4 - Experiment 4: Comparing How Prompt Changes Affect Generated Images
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Create multiple images from different text prompts and compare how changes
    in the prompts affect the generated images.

IDEA:
    All prompts describe the SAME object (a bridge). Only the wording changes -
    detail, style, material, lighting - so the effect of each change is visible.
    The same seed is used everywhere, so any difference comes from the prompt.

OUTPUT:
    outputs/exp04_comparison.png  - all four images in one labelled grid
    outputs/exp04_1.png ... _4.png - the individual images

RUN:
    python exp04_prompt_comparison.py
"""

from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw

MODEL = "stabilityai/sd-turbo"
SEED = 1234                     # same seed for every prompt = fair comparison
OUTPUT_DIR = Path(__file__).parent / "outputs"

PROMPTS = [
    ("Very short prompt",
     "a bridge"),
    ("Added detail",
     "a long steel truss bridge over a river"),
    ("Added style and lighting",
     "a long steel truss bridge over a river at sunset, dramatic golden light, "
     "professional photograph"),
    ("Changed material and era",
     "an ancient stone arch bridge over a river at sunset, moss covered, "
     "professional photograph"),
]


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL} on {device}...")
    return AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=dtype).to(device)


def make_grid(images, labels, cell=512, band=40):
    """Place the images in a 2x2 grid with a caption band under each one."""
    cols, rows = 2, 2
    sheet = Image.new("RGB", (cols * cell, rows * (cell + band)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        x, y = (i % cols) * cell, (i // cols) * (cell + band)
        sheet.paste(img.resize((cell, cell)), (x, y))
        draw.text((x + 8, y + cell + 12), f"{i + 1}. {label}", fill="black")
    return sheet


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pipe = load_pipeline()
    images = []

    for i, (label, prompt) in enumerate(PROMPTS, start=1):
        print(f"\n[{i}/{len(PROMPTS)}] {label}\n    prompt: {prompt}")
        generator = torch.Generator(device=pipe.device.type).manual_seed(SEED)
        image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
                     generator=generator).images[0]
        image.save(OUTPUT_DIR / f"exp04_{i}.png")
        images.append(image)

    grid = make_grid(images, [label for label, _ in PROMPTS])
    grid_path = OUTPUT_DIR / "exp04_comparison.png"
    grid.save(grid_path)

    print("\n=== Observation ===")
    print("1 -> 2 : adding the material and structure type makes the shape specific.")
    print("2 -> 3 : style and lighting words change the mood, not the object.")
    print("3 -> 4 : changing the material word rebuilds the whole structure.")
    print(f"\nComparison sheet saved to: {grid_path}")


if __name__ == "__main__":
    main()
