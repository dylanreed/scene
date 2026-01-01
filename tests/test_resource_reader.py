"""Tests for SCI1 resource reader."""
import sys
sys.path.insert(0, 'src')


def test_read_resource_header():
    """ResourceReader should parse resource header correctly."""
    from sci.resource_reader import ResourceReader

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")

    # Read PIC resource 1 from file 2 at offset 0
    header = reader.read_header(file_number=2, offset=0)

    assert header['type'] == 0x81, "Should be PIC type"
    assert header['resource_number'] == 1, "Should be resource number 1"
    assert header['compressed_size'] > 0, "Compressed size should be positive"
    assert header['decompressed_size'] > 0, "Decompressed size should be positive"
    assert 'compression_method' in header


def test_read_raw_resource_data():
    """ResourceReader should read compressed data from resource file."""
    from sci.resource_reader import ResourceReader

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")

    # Read raw compressed data for PIC 1 from file 2
    data, header = reader.read_resource(file_number=2, offset=0)

    assert len(data) == header['compressed_size'], "Data length should match compressed size"
    assert len(data) > 0, "Should have data"


def test_read_resource_by_type_and_number():
    """ResourceReader should find and read resource by type and number."""
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    resource_map = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    # Get first PIC resource location
    pics = resource_map.get_resources(0x81)
    first_pic = pics[0]

    data, header = reader.read_resource(
        file_number=first_pic['file_number'],
        offset=first_pic['offset']
    )

    assert header['type'] == 0x81
    assert len(data) > 0
