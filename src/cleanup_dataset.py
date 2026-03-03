# ABOUTME: Script to identify and remove low-quality images from the dataset
# ABOUTME: Detects all-black, solid color, and low-variance images

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


def analyze_image(img_path: Path) -> Tuple[float, float, float]:
    """Analyze an image and return quality metrics.

    Returns:
        Tuple of (mean_brightness, color_variance, unique_colors_ratio)
    """
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img, dtype=np.float32)

        # Mean brightness (0-255)
        mean_brightness = arr.mean()

        # Color variance (low = solid color)
        color_variance = arr.std()

        # Unique colors ratio (low = limited palette, very low = solid)
        pixels = arr.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        total_pixels = len(pixels)
        unique_ratio = unique_colors / total_pixels

        return mean_brightness, color_variance, unique_ratio
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return -1, -1, -1


def find_bad_images(
    data_dir: Path,
    min_brightness: float = 5.0,      # Below this = too dark
    max_brightness: float = 250.0,    # Above this = too bright
    min_variance: float = 10.0,       # Below this = too uniform
    min_unique_ratio: float = 0.001,  # Below this = nearly solid
) -> List[Path]:
    """Find images that don't meet quality thresholds."""

    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_paths = []
    for ext in extensions:
        image_paths.extend(data_dir.glob(f'*{ext}'))
        image_paths.extend(data_dir.glob(f'*{ext.upper()}'))

    print(f"Analyzing {len(image_paths)} images...")

    bad_images = []
    reasons = {}

    for img_path in tqdm(image_paths):
        brightness, variance, unique_ratio = analyze_image(img_path)

        if brightness < 0:  # Error loading
            bad_images.append(img_path)
            reasons[img_path.name] = "failed to load"
            continue

        issues = []
        if brightness < min_brightness:
            issues.append(f"too dark ({brightness:.1f})")
        if brightness > max_brightness:
            issues.append(f"too bright ({brightness:.1f})")
        if variance < min_variance:
            issues.append(f"too uniform ({variance:.1f})")
        if unique_ratio < min_unique_ratio:
            issues.append(f"nearly solid ({unique_ratio:.4f})")

        if issues:
            bad_images.append(img_path)
            reasons[img_path.name] = ", ".join(issues)

    return bad_images, reasons


def clean_captions(captions_file: Path, bad_image_names: set, output_file: Path):
    """Remove entries for bad images from captions file."""
    if not captions_file.exists():
        print(f"No captions file at {captions_file}")
        return 0

    kept = []
    removed = 0

    with open(captions_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get('image') not in bad_image_names:
                    kept.append(entry)
                else:
                    removed += 1

    with open(output_file, 'w') as f:
        for entry in kept:
            f.write(json.dumps(entry) + '\n')

    return removed


def main():
    parser = argparse.ArgumentParser(description="Find and remove bad images from dataset")
    parser.add_argument("--data-dir", type=str, default="data/images",
                        help="Directory containing images")
    parser.add_argument("--captions", type=str, default="data/captions.jsonl",
                        help="Captions file to clean")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report bad images, don't delete")
    parser.add_argument("--min-brightness", type=float, default=5.0,
                        help="Minimum mean brightness (0-255)")
    parser.add_argument("--min-variance", type=float, default=10.0,
                        help="Minimum color variance")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    captions_file = Path(args.captions)

    # Find bad images
    bad_images, reasons = find_bad_images(
        data_dir,
        min_brightness=args.min_brightness,
        min_variance=args.min_variance,
    )

    print(f"\nFound {len(bad_images)} problematic images:")
    for img_path in bad_images[:20]:  # Show first 20
        print(f"  {img_path.name}: {reasons[img_path.name]}")
    if len(bad_images) > 20:
        print(f"  ... and {len(bad_images) - 20} more")

    if args.dry_run:
        print("\nDry run - no files deleted")
        return

    # Confirm deletion
    print(f"\nThis will delete {len(bad_images)} images and update captions.")
    response = input("Proceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted")
        return

    # Delete bad images
    bad_names = set()
    for img_path in bad_images:
        bad_names.add(img_path.name)
        img_path.unlink()
        print(f"Deleted {img_path.name}")

    # Clean captions
    if captions_file.exists():
        removed = clean_captions(captions_file, bad_names, captions_file)
        print(f"Removed {removed} entries from captions")

    print(f"\nDone! Removed {len(bad_images)} images")


if __name__ == "__main__":
    main()
