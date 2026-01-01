"""Tests for SCI1 LZW decompression."""
import sys
sys.path.insert(0, 'src')


def test_decompress_returns_expected_size():
    """LZW decompression should produce expected output size."""
    from sci.lzw import decompress_lzw
    from sci.resource_reader import ResourceReader

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    data, header = reader.read_resource(file_number=2, offset=0)

    # Compression method 2 = LZW
    assert header['compression_method'] == 2

    decompressed = decompress_lzw(data, header['decompressed_size'])

    assert len(decompressed) == header['decompressed_size'], \
        f"Expected {header['decompressed_size']} bytes, got {len(decompressed)}"


def test_decompress_produces_valid_data():
    """LZW decompression should produce non-zero data."""
    from sci.lzw import decompress_lzw
    from sci.resource_reader import ResourceReader

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    data, header = reader.read_resource(file_number=2, offset=0)

    decompressed = decompress_lzw(data, header['decompressed_size'])

    # Should have some variety in the data (not all zeros)
    unique_bytes = len(set(decompressed))
    assert unique_bytes > 1, "Decompressed data should have variety"


def test_decompress_multiple_resources():
    """LZW should work on multiple different PIC resources."""
    from sci.lzw import decompress_lzw
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    resource_map = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    pics = resource_map.get_resources(0x81)[:5]  # First 5 PICs

    for pic in pics:
        data, header = reader.read_resource(
            file_number=pic['file_number'],
            offset=pic['offset']
        )

        if header['compression_method'] == 2:  # LZW
            decompressed = decompress_lzw(data, header['decompressed_size'])
            assert len(decompressed) == header['decompressed_size'], \
                f"PIC {pic['resource_number']}: size mismatch"
