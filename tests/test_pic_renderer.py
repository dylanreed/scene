"""Tests for SCI1 PIC renderer."""
import sys
sys.path.insert(0, 'src')


def test_render_pic_creates_image():
    """PicRenderer should create a 320x200 image."""
    from sci.pic_renderer import PicRenderer
    from sci.palette import Palette
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap
    from sci.lzw import decompress_lzw

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    rm = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    # Get palette
    palettes = rm.get_resources(0x8B)
    pal_info = palettes[0]
    pal_data, pal_header = reader.read_resource(pal_info['file_number'], pal_info['offset'])
    pal_data = decompress_lzw(pal_data, pal_header['decompressed_size'])
    palette = Palette(pal_data)

    # Get a PIC resource
    pics = rm.get_resources(0x81)
    for pic in pics:
        data, header = reader.read_resource(pic['file_number'], pic['offset'])
        if header['decompressed_size'] > 10000:
            pic_data = decompress_lzw(data, header['decompressed_size'])

            renderer = PicRenderer(palette)
            image = renderer.render(pic_data)

            assert image.size == (320, 200), f"Image should be 320x200, got {image.size}"
            assert image.mode == "RGB", "Image should be RGB"
            break


def test_render_pic_returns_image():
    """Rendered PIC should return a valid image object."""
    from sci.pic_renderer import PicRenderer
    from sci.palette import Palette
    from sci.resource_reader import ResourceReader
    from sci.resource_map import ResourceMap
    from sci.lzw import decompress_lzw

    reader = ResourceReader("/Users/nervous/DOSGAMES/larry5")
    rm = ResourceMap("/Users/nervous/DOSGAMES/larry5")

    # Get palette
    palettes = rm.get_resources(0x8B)
    pal_info = palettes[0]
    pal_data, pal_header = reader.read_resource(pal_info['file_number'], pal_info['offset'])
    pal_data = decompress_lzw(pal_data, pal_header['decompressed_size'])
    palette = Palette(pal_data)

    # Get any PIC
    pics = rm.get_resources(0x81)
    for pic in pics:
        data, header = reader.read_resource(pic['file_number'], pic['offset'])
        pic_data = decompress_lzw(data, header['decompressed_size'])

        renderer = PicRenderer(palette)
        image = renderer.render(pic_data)

        # Should return a valid image
        assert image is not None
        assert hasattr(image, 'save'), "Should be a PIL Image"
        break
