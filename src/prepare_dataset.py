"""Prepare dataset by captioning images with Claude Vision.

Usage:
    1. Put extracted game backgrounds in data/extract/
    2. Run: python src/prepare_dataset.py
    3. Images will be moved to data/images/ with captions in data/captions.jsonl

Batch mode (recommended for large datasets):
    python src/prepare_dataset.py --batch

    This uses the Message Batches API for 50% cost savings.
"""

import argparse
import base64
import json
import os
import re
import shutil
import time
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


CAPTION_PROMPT = """Describe this pixel art background image in detail for training an image generation model.

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


def generate_caption(client: anthropic.Anthropic, image_path: Path) -> str:
    """Generate a detailed caption for a pixel art background using Claude Vision."""

    image_data = encode_image_base64(image_path)
    media_type = get_image_media_type(image_path)

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
                        "text": CAPTION_PROMPT,
                    },
                ],
            }
        ],
    )

    return response.content[0].text.strip()


def resize_if_needed(
    image_path: Path, output_path: Path, target_size: tuple = (320, 200)
):
    """Resize image to target size if needed, using nearest neighbor for pixel art."""
    img = Image.open(image_path).convert("RGB")

    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.NEAREST)

    img.save(output_path, "PNG")


def sanitize_custom_id(name: str) -> str:
    """Sanitize string to match batch API custom_id pattern: ^[a-zA-Z0-9_-]{1,64}"""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Truncate to 64 characters
    return sanitized[:64]


def create_batch_request(image_path: Path, custom_id: str) -> dict:
    """Create a batch request for a single image."""
    image_data = encode_image_base64(image_path)
    media_type = get_image_media_type(image_path)

    return {
        "custom_id": custom_id,
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [
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
                            "text": CAPTION_PROMPT,
                        },
                    ],
                }
            ],
        },
    }


def process_single_batch(
    client: anthropic.Anthropic,
    requests: list,
    image_lookup: dict,
    images_dir: Path,
    captions_file: Path,
    resize: bool,
    target_size: tuple,
    poll_interval: int,
) -> tuple[int, int]:
    """Process a single batch of requests. Returns (processed, errors) counts."""

    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")

    # Poll for completion
    while batch.processing_status in ("in_progress", "created"):
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        print(
            f"    {batch.processing_status} | "
            f"Succeeded: {counts.succeeded}/{total} | "
            f"Errors: {counts.errored}"
        )

    if batch.processing_status != "ended":
        print(f"  Batch ended with status: {batch.processing_status}")
        return 0, len(requests)

    # Process results
    processed = 0
    errors = 0

    with open(captions_file, "a") as f:
        for result in client.messages.batches.results(batch.id):
            custom_id = result.custom_id  # stem without extension
            lookup_entry = image_lookup.get(custom_id)

            if result.result.type == "succeeded":
                caption = result.result.message.content[0].text.strip()

                # Copy/resize image
                if lookup_entry:
                    image_path, output_name = lookup_entry
                    output_path = images_dir / output_name
                    if resize:
                        resize_if_needed(image_path, output_path, target_size)
                    else:
                        shutil.copy(image_path, output_path)

                    # Write caption entry
                    entry = {"image": output_name, "caption": caption}
                    f.write(json.dumps(entry) + "\n")
                processed += 1
            else:
                print(f"    Error for {custom_id}: {result.result.type}")
                errors += 1

    return processed, errors


def process_images_batch(
    extract_dir: Path,
    images_dir: Path,
    captions_file: Path,
    resize: bool = True,
    target_size: tuple = (320, 200),
    skip_existing: bool = True,
    poll_interval: int = 30,
    batch_size: int = 100,
):
    """Process all images using the Message Batches API in chunks."""

    client = anthropic.Anthropic()

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    image_files = sorted(
        [f for f in extract_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    if not image_files:
        print(f"No images found in {extract_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Load existing captions to avoid duplicates
    existing_images = set()
    if captions_file.exists() and skip_existing:
        with open(captions_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_images.add(entry["image"])

    # Filter to only images that need processing
    images_to_process = []
    for image_path in image_files:
        output_name = f"{image_path.stem}.png"
        if output_name not in existing_images:
            images_to_process.append((image_path, output_name))

    if not images_to_process:
        print("All images already processed!")
        return

    print(f"Processing {len(images_to_process)} new images (skipping {len(existing_images)} existing)")

    # Build lookup for image paths (keyed by sanitized stem for custom_id matching)
    image_lookup = {sanitize_custom_id(image_path.stem): (image_path, output_name) for image_path, output_name in images_to_process}

    # Process in batches
    total_batches = (len(images_to_process) + batch_size - 1) // batch_size
    total_processed = 0
    total_errors = 0

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(images_to_process))
        batch_images = images_to_process[start_idx:end_idx]

        print(f"\nBatch {batch_num + 1}/{total_batches} ({len(batch_images)} images)")

        # Create batch requests (custom_id must be sanitized)
        requests = []
        for image_path, output_name in batch_images:
            custom_id = sanitize_custom_id(image_path.stem)
            req = create_batch_request(image_path, custom_id=custom_id)
            requests.append(req)

        # Process this batch
        processed, errors = process_single_batch(
            client=client,
            requests=requests,
            image_lookup=image_lookup,
            images_dir=images_dir,
            captions_file=captions_file,
            resize=resize,
            target_size=target_size,
            poll_interval=poll_interval,
        )

        total_processed += processed
        total_errors += errors
        print(f"  Batch complete: {processed} processed, {errors} errors")

    print(f"\n{'='*50}")
    print(f"All batches complete! Processed: {total_processed}, Errors: {total_errors}")
    print(f"Images saved to: {images_dir}")
    print(f"Captions saved to: {captions_file}")


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
        f for f in extract_dir.iterdir() if f.suffix.lower() in image_extensions
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
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Message Batches API (50%% cost savings, async processing)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status checks (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Images per batch to avoid size limits (default: 100)",
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
    if args.batch:
        process_images_batch(
            extract_dir=extract_dir,
            images_dir=images_dir,
            captions_file=captions_file,
            resize=not args.no_resize,
            target_size=(args.width, args.height),
            skip_existing=not args.no_skip_existing,
            poll_interval=args.poll_interval,
            batch_size=args.batch_size,
        )
    else:
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
