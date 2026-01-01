#!/usr/bin/env python3
"""Extract background images from Sierra SCI1 games.

Usage:
    python src/extract_sci.py /path/to/game --output extracts/ --prefix KQ5
"""
import argparse
import os
import sys

from sci.resource_map import ResourceMap
from sci.resource_reader import ResourceReader
from sci.lzw import decompress_lzw
from sci.palette import Palette, create_default_palette
from sci.pic_renderer import PicRenderer


def load_palette(reader: ResourceReader, rm: ResourceMap) -> Palette:
    """Load palette from game resources or create default."""
    palettes = rm.get_resources(0x8B)  # PALETTE type

    if palettes:
        pal_info = palettes[0]
        data, header = reader.read_resource(
            pal_info['file_number'],
            pal_info['offset']
        )

        if header['compression_method'] == 2:  # LZW
            pal_data = decompress_lzw(data, header['decompressed_size'])
        else:
            pal_data = data

        return Palette(pal_data)

    print("No palette resource found, using default VGA palette")
    return create_default_palette()


def extract_pics(game_path: str, output_dir: str, prefix: str):
    """Extract all PIC resources from a game."""
    os.makedirs(output_dir, exist_ok=True)

    reader = ResourceReader(game_path)
    rm = ResourceMap(game_path)

    print(f"Loading from: {game_path}")
    print(f"Output to: {output_dir}")

    # Load palette
    palette = load_palette(reader, rm)
    print(f"Loaded palette with {len(palette)} colors")

    # Get PIC resources
    pics = rm.get_resources(0x81)  # PIC type
    print(f"Found {len(pics)} PIC resources")

    extracted = 0
    failed = 0
    skipped = 0

    # Track seen resource numbers to avoid duplicates
    seen = set()

    for pic_info in pics:
        res_num = pic_info['resource_number']

        # Skip duplicates (same resource in multiple files)
        if res_num in seen:
            continue
        seen.add(res_num)

        try:
            data, header = reader.read_resource(
                pic_info['file_number'],
                pic_info['offset']
            )

            # Skip very small resources (likely not full pictures)
            if header['decompressed_size'] < 5000:
                skipped += 1
                continue

            # Decompress
            if header['compression_method'] == 2:  # LZW
                pic_data = decompress_lzw(data, header['decompressed_size'])
            elif header['compression_method'] == 4:  # LZW with PIC reordering
                # Note: Method 4 needs special reordering not yet implemented
                pic_data = decompress_lzw(data, header['decompressed_size'])
            elif header['compression_method'] == 0:  # Uncompressed
                pic_data = data
            else:
                print(f"  PIC {res_num}: Unknown compression method {header['compression_method']}, skipping")
                skipped += 1
                continue

            # Render
            renderer = PicRenderer(palette)
            image = renderer.render(pic_data)

            # Save
            filename = f"{prefix}-{res_num:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)

            extracted += 1
            print(f"  Extracted: {filename}")

        except Exception as e:
            failed += 1
            print(f"  PIC {res_num}: Failed - {e}")

    print()
    print(f"Summary:")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (too small): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total unique: {len(seen)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract background images from Sierra SCI1 games"
    )
    parser.add_argument(
        "game_path",
        help="Directory containing RESOURCE.MAP and RESOURCE.xxx files"
    )
    parser.add_argument(
        "--output", "-o",
        default="extracts",
        help="Output directory (default: extracts/)"
    )
    parser.add_argument(
        "--prefix", "-p",
        default=None,
        help="Filename prefix (default: derived from path)"
    )

    args = parser.parse_args()

    # Derive prefix from path if not specified
    if args.prefix is None:
        args.prefix = os.path.basename(args.game_path.rstrip("/\\")).upper()

    # Validate game path
    resource_map = os.path.join(args.game_path, "RESOURCE.MAP")
    if not os.path.exists(resource_map):
        print(f"Error: RESOURCE.MAP not found in {args.game_path}")
        sys.exit(1)

    extract_pics(args.game_path, args.output, args.prefix)


if __name__ == "__main__":
    main()
