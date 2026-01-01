"""Tests for SCI1 palette handling."""
import sys
sys.path.insert(0, 'src')


def test_load_palette_resource():
    """Palette should load and return 256 RGB colors."""
    from sci.palette import Palette
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap
    from sci.lzw import decompress_lzw

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    rm = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    # Get palette resource
    palettes = rm.get_resources(0x8B)
    assert len(palettes) > 0, "Should have palette resources"

    pal_info = palettes[0]
    data, header = reader.read_resource(pal_info['file_number'], pal_info['offset'])

    # Decompress if needed
    if header['compression_method'] == 2:
        pal_data = decompress_lzw(data, header['decompressed_size'])
    else:
        pal_data = data

    palette = Palette(pal_data)

    # Should have 256 colors
    assert len(palette.colors) == 256, "Should have 256 colors"

    # Each color should be an RGB tuple
    assert len(palette.colors[0]) == 3, "Each color should be RGB tuple"

    # Colors should be in 0-255 range
    for i, (r, g, b) in enumerate(palette.colors):
        assert 0 <= r <= 255, f"Red out of range at index {i}"
        assert 0 <= g <= 255, f"Green out of range at index {i}"
        assert 0 <= b <= 255, f"Blue out of range at index {i}"


def test_palette_color_0_is_black():
    """Palette color 0 should typically be black."""
    from sci.palette import Palette
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap
    from sci.lzw import decompress_lzw

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    rm = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    palettes = rm.get_resources(0x8B)
    pal_info = palettes[0]
    data, header = reader.read_resource(pal_info['file_number'], pal_info['offset'])

    if header['compression_method'] == 2:
        pal_data = decompress_lzw(data, header['decompressed_size'])
    else:
        pal_data = data

    palette = Palette(pal_data)

    # Color 0 is typically black (or very dark)
    r, g, b = palette.colors[0]
    assert r < 10 and g < 10 and b < 10, f"Color 0 should be dark, got ({r}, {g}, {b})"


def test_palette_index_lookup():
    """Palette should support indexed lookup."""
    from sci.palette import Palette
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap
    from sci.lzw import decompress_lzw

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    rm = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    palettes = rm.get_resources(0x8B)
    pal_info = palettes[0]
    data, header = reader.read_resource(pal_info['file_number'], pal_info['offset'])

    if header['compression_method'] == 2:
        pal_data = decompress_lzw(data, header['decompressed_size'])
    else:
        pal_data = data

    palette = Palette(pal_data)

    # Test index lookup
    color = palette[0]
    assert len(color) == 3
    assert palette[0] == palette.colors[0]
