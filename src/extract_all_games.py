"""Batch extract images from all SCI1 games in DOSGAMES folder."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sci.resource_map import ResourceMap
from sci.resource_reader import ResourceReader
from sci.lzw import decompress_lzw
from sci.palette import load_palette_from_game
from sci.pic_renderer import PicRenderer
from sci.view_renderer import ViewRenderer

# SCI1 games with RESOURCE.000 files
SCI1_GAMES = [
    "ecoquest",
    "ecoquest2",
    "eq1",
    "eq2",
    "heroq1",
    "heroq2",
    "heroq3",
    "hq1vga",
    "hq4",
    "ICEMAN",
    "Kingquest5",
    "KingQuest6",
    "kq5",
    "KQ6CD",
    "larry1vga",
    "larry5",
    "larry6",
    "LAUraBOW2",
    "PepperAdventuresintime",
    "PoliceQuest1vga",
    "PoliceQuest3",
    "PoliceQuest4",
    "ROBINH",
    "SpaceQuest-1vga",
    "SpaceQuest-4",
    "SpaceQuest-5",
]

DOSGAMES_PATH = "/Users/nervous/DOSGAMES"
OUTPUT_BASE = "/Users/nervous/Library/CloudStorage/Dropbox/Github/scene/data/extracted"


def extract_game(game_name: str) -> dict:
    """Extract PICs and VIEWs from a single game.

    Returns dict with extraction stats.
    """
    game_path = os.path.join(DOSGAMES_PATH, game_name)
    output_path = os.path.join(OUTPUT_BASE, game_name)

    stats = {
        'game': game_name,
        'pics_found': 0,
        'pics_extracted': 0,
        'views_found': 0,
        'views_extracted': 0,
        'errors': []
    }

    # Check game exists
    if not os.path.exists(game_path):
        stats['errors'].append(f"Game path not found: {game_path}")
        return stats

    # Check for RESOURCE.MAP
    if not os.path.exists(os.path.join(game_path, "RESOURCE.MAP")):
        stats['errors'].append("No RESOURCE.MAP found")
        return stats

    # Create output directories
    pics_dir = os.path.join(output_path, "pics")
    views_dir = os.path.join(output_path, "views")
    os.makedirs(pics_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    try:
        resource_map = ResourceMap(game_path)
        reader = ResourceReader(game_path)
        palette = load_palette_from_game(game_path)
        pic_renderer = PicRenderer(palette)
        view_renderer = ViewRenderer(palette)
    except Exception as e:
        stats['errors'].append(f"Failed to initialize: {e}")
        return stats

    # Extract PICs (type 0x81)
    try:
        pics = resource_map.get_resources(0x81)
        stats['pics_found'] = len(pics)

        for pic_info in pics:
            try:
                data, header = reader.read_resource(
                    pic_info['file_number'],
                    pic_info['offset']
                )

                if header['compression_method'] != 0:
                    decompressed = decompress_lzw(data, header['decompressed_size'])
                else:
                    decompressed = data

                img = pic_renderer.render(decompressed)
                if img and img.width > 0 and img.height > 0:
                    output_file = os.path.join(pics_dir, f"pic_{pic_info['resource_number']:03d}.png")
                    img.save(output_file)
                    stats['pics_extracted'] += 1

            except Exception as e:
                stats['errors'].append(f"PIC {pic_info['resource_number']}: {e}")

    except Exception as e:
        stats['errors'].append(f"PIC extraction failed: {e}")

    # Extract VIEWs (type 0x80)
    try:
        views = resource_map.get_resources(0x80)
        stats['views_found'] = len(views)

        for view_info in views:
            try:
                data, header = reader.read_resource(
                    view_info['file_number'],
                    view_info['offset']
                )

                if header['compression_method'] != 0:
                    decompressed = decompress_lzw(data, header['decompressed_size'])
                else:
                    decompressed = data

                view = view_renderer.parse(decompressed)
                if view.loops:
                    img = view_renderer.render_view_sheet(view)
                    if img.width > 0 and img.height > 0:
                        output_file = os.path.join(views_dir, f"view_{view_info['resource_number']:03d}.png")
                        img.save(output_file)
                        stats['views_extracted'] += 1

            except Exception as e:
                stats['errors'].append(f"VIEW {view_info['resource_number']}: {e}")

    except Exception as e:
        stats['errors'].append(f"VIEW extraction failed: {e}")

    return stats


def main():
    """Extract images from all SCI1 games."""
    print("=" * 60)
    print("SCI1 Batch Image Extractor")
    print("=" * 60)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_stats = []
    total_pics = 0
    total_views = 0

    for i, game in enumerate(SCI1_GAMES):
        print(f"\n[{i+1}/{len(SCI1_GAMES)}] Processing: {game}")
        print("-" * 40)

        stats = extract_game(game)
        all_stats.append(stats)

        print(f"  PICs: {stats['pics_extracted']}/{stats['pics_found']} extracted")
        print(f"  VIEWs: {stats['views_extracted']}/{stats['views_found']} extracted")

        if stats['errors'] and len(stats['errors']) <= 5:
            for err in stats['errors']:
                print(f"  Error: {err}")
        elif stats['errors']:
            print(f"  ({len(stats['errors'])} errors, showing first 3)")
            for err in stats['errors'][:3]:
                print(f"  Error: {err}")

        total_pics += stats['pics_extracted']
        total_views += stats['views_extracted']

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Games processed: {len(SCI1_GAMES)}")
    print(f"Total PICs extracted: {total_pics}")
    print(f"Total VIEWs extracted: {total_views}")
    print(f"Output directory: {OUTPUT_BASE}")

    # Detailed summary
    print("\nPer-game summary:")
    print("-" * 60)
    for stats in all_stats:
        success = stats['pics_extracted'] + stats['views_extracted']
        total = stats['pics_found'] + stats['views_found']
        pct = (success / total * 100) if total > 0 else 0
        print(f"  {stats['game']}: {success}/{total} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
