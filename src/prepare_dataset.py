"""Prepare dataset by captioning images with Claude Vision.

Usage:
    1. Put extracted game backgrounds in data/extract/
    2. Run: python src/prepare_dataset.py
    3. Images will be moved to data/images/ with captions in data/captions.jsonl
"""

import argparse
import base64
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image
from tqdm import tqdm


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """Get media type from image extension."""
    ext = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return media_types.get(ext, "image/png")


def generate_caption(client: anthropic.Anthropic, image_path: Path) -> str:
    """Generate a detailed caption for a pixel art background using Claude Vision."""

    image_data = encode_image_base64(image_path)
    media_type = get_image_media_type(image_path)

    prompt = """Describe this pixel art background image in detail for training an image generation model.

Focus on:
- The setting/location (forest, castle, cave, town, etc.)
- Time of day and lighting (sunset, night, moonlit, etc.)
- Key visual elements and objects
- Colors and mood/atmosphere
- Art style details (VGA-era, adventure game style, etc.)

Write a single detailed paragraph (2-4 sentences) that captures the essence of the scene.
Do NOT start with "This image shows" or "The image depicts" - just describe the scene directly.

Example: "A moonlit forest clearing with twisted oak trees casting long shadows across a winding dirt path. Purple and blue hues dominate the night sky, with stars visible through gaps in the canopy. A small wooden cabin with glowing windows sits at the edge of the clearing, smoke rising from its stone chimney."
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    return response.content[0].text.strip()


def resize_if_needed(image_path: Path, output_path: Path, target_size: tuple = (320, 200)):
    """Resize image to target size if needed, using nearest neighbor for pixel art."""
    img = Image.open(image_path).convert("RGB")

    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.NEAREST)

    img.save(output_path, "PNG")


def process_images(
    extract_dir: Path,
    images_dir: Path,
    captions_file: Path,
    resize: bool = True,
    target_size: tuple = (320, 200),
    skip_existing: bool = True,
):
    """Process all images in extract directory."""

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    image_files = [
        f for f in extract_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {extract_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Load existing captions to avoid duplicates
    existing_images = set()
    if captions_file.exists() and skip_existing:
        with open(captions_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_images.add(entry["image"])

    # Process each image
    processed = 0
    skipped = 0
    errors = 0

    with open(captions_file, "a") as f:
        for image_path in tqdm(image_files, desc="Processing images"):
            # Generate output filename
            output_name = f"{image_path.stem}.png"
            output_path = images_dir / output_name

            # Skip if already processed
            if output_name in existing_images:
                skipped += 1
                continue

            try:
                # Generate caption
                caption = generate_caption(client, image_path)

                # Copy/resize image
                if resize:
                    resize_if_needed(image_path, output_path, target_size)
                else:
                    shutil.copy(image_path, output_path)

                # Write caption entry
                entry = {"image": output_name, "caption": caption}
                f.write(json.dumps(entry) + "\n")
                f.flush()

                processed += 1

            except Exception as e:
                print(f"\nError processing {image_path.name}: {e}")
                errors += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"Images saved to: {images_dir}")
    print(f"Captions saved to: {captions_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset by captioning images with Claude Vision"
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/extract",
        help="Directory containing extracted game images (default: data/extract)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Output directory for processed images (default: data/images)",
    )
    parser.add_argument(
        "--captions-file",
        type=str,
        default="data/captions.jsonl",
        help="Output file for captions (default: data/captions.jsonl)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize images to target size",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target width (default: 320)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=200,
        help="Target height (default: 200)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess images that already have captions",
    )

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent
    extract_dir = project_root / args.extract_dir
    images_dir = project_root / args.images_dir
    captions_file = project_root / args.captions_file

    # Create directories
    extract_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Process images
    process_images(
        extract_dir=extract_dir,
        images_dir=images_dir,
        captions_file=captions_file,
        resize=not args.no_resize,
        target_size=(args.width, args.height),
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
