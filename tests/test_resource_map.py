"""Tests for SCI1 resource map parsing."""
import sys
sys.path.insert(0, 'src')


def test_parse_resource_map_returns_pic_resources():
    """ResourceMap should parse type headers and return PIC resource locations."""
    from sci.resource_map import ResourceMap

    # Larry 5 RESOURCE.MAP should contain PIC resources (type 0x81)
    resource_map = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    pics = resource_map.get_resources(0x81)  # PIC type

    assert len(pics) > 0, "Should find PIC resources"
    # Each entry should have resource_number, file_number, offset
    first_pic = pics[0]
    assert 'resource_number' in first_pic
    assert 'file_number' in first_pic
    assert 'offset' in first_pic


def test_parse_resource_map_type_headers():
    """ResourceMap should correctly parse type headers from RESOURCE.MAP."""
    from sci.resource_map import ResourceMap

    resource_map = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    # Should have multiple resource types
    resource_types = resource_map.get_resource_types()

    assert 0x80 in resource_types, "Should have VIEW resources"
    assert 0x81 in resource_types, "Should have PIC resources"
    assert 0x82 in resource_types, "Should have SCRIPT resources"


def test_parse_lookup_table_entry():
    """ResourceMap should correctly decode 6-byte lookup entries."""
    from sci.resource_map import ResourceMap

    resource_map = ResourceMap("/Users/nervous/DOSGAMES/larry5")
    pics = resource_map.get_resources(0x81)

    # File numbers should be 0-7 (Larry 5 has RESOURCE.000-007)
    for pic in pics:
        assert 0 <= pic['file_number'] <= 7, f"Invalid file number: {pic['file_number']}"
        assert pic['offset'] >= 0, "Offset should be non-negative"
        assert pic['resource_number'] >= 0, "Resource number should be non-negative"
