"""Tests for SCI1 VIEW renderer."""
import sys
sys.path.insert(0, 'src')

GAME_PATH = "/Users/nervous/DOSGAMES/larry5"


def test_view_parser_finds_views():
    """Should find VIEW resources in the resource map."""
    from sci.resource_map import ResourceMap

    resource_map = ResourceMap(GAME_PATH)
    views = resource_map.get_resources(0x80)  # VIEW type

    assert len(views) > 0, "Should find VIEW resources"
    print(f"Found {len(views)} VIEW resources")


def test_view_parser_reads_view_data():
    """Should read and decompress VIEW resource data."""
    from sci.resource_map import ResourceMap
    from sci.resource_reader import ResourceReader
    from sci.lzw import decompress_lzw

    resource_map = ResourceMap(GAME_PATH)
    reader = ResourceReader(GAME_PATH)

    views = resource_map.get_resources(0x80)
    assert len(views) > 0

    # Try reading first few views
    for i, view_info in enumerate(views[:5]):
        data, header = reader.read_resource(
            view_info['file_number'],
            view_info['offset']
        )

        print(f"\nVIEW {view_info['resource_number']}:")
        print(f"  File: RESOURCE.{view_info['file_number']:03d}")
        print(f"  Compressed: {header['compressed_size']} bytes")
        print(f"  Decompressed: {header['decompressed_size']} bytes")
        print(f"  Method: {header['compression_method']}")

        # Decompress if needed
        if header['compression_method'] != 0:
            decompressed = decompress_lzw(data, header['decompressed_size'])
            print(f"  Decompressed to: {len(decompressed)} bytes")
        else:
            decompressed = data
            print(f"  Uncompressed data")

        # Show first few bytes
        print(f"  First 16 bytes: {decompressed[:16].hex()}")


def test_view_parser_parses_view():
    """Should parse VIEW structure (loops and cels)."""
    from sci.resource_map import ResourceMap
    from sci.resource_reader import ResourceReader
    from sci.lzw import decompress_lzw
    from sci.palette import load_palette_from_game
    from sci.view_renderer import ViewRenderer

    resource_map = ResourceMap(GAME_PATH)
    reader = ResourceReader(GAME_PATH)

    views = resource_map.get_resources(0x80)
    assert len(views) > 0

    # Load palette
    palette = load_palette_from_game(GAME_PATH)
    renderer = ViewRenderer(palette)

    # Try parsing first view
    view_info = views[0]
    data, header = reader.read_resource(
        view_info['file_number'],
        view_info['offset']
    )

    # Decompress
    if header['compression_method'] != 0:
        decompressed = decompress_lzw(data, header['decompressed_size'])
    else:
        decompressed = data

    print(f"\nParsing VIEW {view_info['resource_number']}:")
    print(f"  Data size: {len(decompressed)} bytes")

    # Parse
    view = renderer.parse(decompressed)

    print(f"  Loops: {len(view.loops)}")
    for i, loop in enumerate(view.loops):
        print(f"    Loop {i}: {len(loop.cels)} cels, mirror={loop.is_mirror}")
        for j, cel in enumerate(loop.cels):
            print(f"      Cel {j}: {cel.width}x{cel.height}, disp=({cel.displacement_x},{cel.displacement_y}), trans={cel.transparent_color}")


def test_view_renderer_renders_cel():
    """Should render a cel to an image."""
    from sci.resource_map import ResourceMap
    from sci.resource_reader import ResourceReader
    from sci.lzw import decompress_lzw
    from sci.palette import load_palette_from_game
    from sci.view_renderer import ViewRenderer

    resource_map = ResourceMap(GAME_PATH)
    reader = ResourceReader(GAME_PATH)

    views = resource_map.get_resources(0x80)
    palette = load_palette_from_game(GAME_PATH)
    renderer = ViewRenderer(palette)

    # Find a view with reasonable size
    for view_info in views[:20]:
        data, header = reader.read_resource(
            view_info['file_number'],
            view_info['offset']
        )

        if header['compression_method'] != 0:
            decompressed = decompress_lzw(data, header['decompressed_size'])
        else:
            decompressed = data

        try:
            view = renderer.parse(decompressed)

            if view.loops and view.loops[0].cels:
                cel = view.loops[0].cels[0]
                if cel.width > 0 and cel.height > 0:
                    img = renderer.render_cel(cel)
                    output_path = f"extracts/view_{view_info['resource_number']}_cel.png"
                    img.save(output_path)
                    print(f"Saved: {output_path} ({cel.width}x{cel.height})")
                    return  # Success

        except Exception as e:
            print(f"VIEW {view_info['resource_number']}: {e}")
            continue

    assert False, "Could not render any view cel"


def test_view_renderer_renders_sheet():
    """Should render entire view as sprite sheet."""
    from sci.resource_map import ResourceMap
    from sci.resource_reader import ResourceReader
    from sci.lzw import decompress_lzw
    from sci.palette import load_palette_from_game
    from sci.view_renderer import ViewRenderer
    import os

    os.makedirs("extracts", exist_ok=True)

    resource_map = ResourceMap(GAME_PATH)
    reader = ResourceReader(GAME_PATH)

    views = resource_map.get_resources(0x80)
    palette = load_palette_from_game(GAME_PATH)
    renderer = ViewRenderer(palette)

    rendered = 0
    for view_info in views[:30]:
        data, header = reader.read_resource(
            view_info['file_number'],
            view_info['offset']
        )

        if header['compression_method'] != 0:
            decompressed = decompress_lzw(data, header['decompressed_size'])
        else:
            decompressed = data

        try:
            view = renderer.parse(decompressed)

            if view.loops:
                img = renderer.render_view_sheet(view)
                if img.width > 0 and img.height > 0:
                    output_path = f"extracts/view_{view_info['resource_number']:03d}_sheet.png"
                    img.save(output_path)
                    print(f"Saved: {output_path} ({img.width}x{img.height}, {len(view.loops)} loops)")
                    rendered += 1

        except Exception as e:
            print(f"VIEW {view_info['resource_number']}: {e}")
            continue

    print(f"\nRendered {rendered} view sheets")
    assert rendered > 0, "Should render at least one view sheet"


if __name__ == "__main__":
    print("=== Test 1: Find VIEWs ===")
    test_view_parser_finds_views()

    print("\n=== Test 2: Read VIEW data ===")
    test_view_parser_reads_view_data()

    print("\n=== Test 3: Parse VIEW structure ===")
    test_view_parser_parses_view()

    print("\n=== Test 4: Render single cel ===")
    test_view_renderer_renders_cel()

    print("\n=== Test 5: Render sprite sheets ===")
    test_view_renderer_renders_sheet()
