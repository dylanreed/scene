"""Debug RLE decoding by trying different formats."""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sci.resource_reader import ResourceReader
from src.sci.palette import Palette


def hex_dump(data: bytes, offset: int, length: int) -> None:
    """Print hex dump."""
    end = min(offset + length, len(data))
    for i in range(offset, end, 16):
        hex_part = " ".join(f"{data[j]:02X}" for j in range(i, min(i + 16, end)))
        print(f"  {i:5d}: {hex_part}")


def find_bitmap_offset(data: bytes, start: int = 1037) -> int:
    """Find bitmap offset by parsing vector commands."""
    pos = start

    while pos < len(data):
        opcode = data[pos]

        if opcode == 0xFF:  # END
            return pos + 1
        elif opcode in (0xF0, 0xF2, 0xF9):  # Fixed 1-arg
            pos += 2
        elif opcode in (0xF1, 0xF3):  # Fixed 0-arg
            pos += 1
        elif opcode >= 0xF0:  # Variable args
            pos += 1
            while pos < len(data) and data[pos] < 0xF0:
                pos += 1
        else:
            break

    return pos


def try_decode_v1(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Original interpretation: 0x00-0x7F literal, 0x80-0xFE run."""
    pos = offset
    output = bytearray()
    max_pixels = width * height

    while pos < len(data) and len(output) < max_pixels:
        byte = data[pos]
        pos += 1

        if byte < 0x80:
            output.append(byte)
        elif byte == 0xFF:
            if len(output) >= max_pixels - 100:
                break
            output.append(byte)
        else:
            if pos >= len(data):
                break
            count = byte - 0x80
            if count == 0:
                count = 128
            color = data[pos]
            pos += 1
            output.extend([color] * count)

    return bytes(output[:max_pixels])


def try_decode_v2(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Inverted: 0x80+ literal, 0x00-0x7F run."""
    pos = offset
    output = bytearray()
    max_pixels = width * height

    while pos < len(data) and len(output) < max_pixels:
        byte = data[pos]
        pos += 1

        if byte >= 0x80:
            output.append(byte)
        elif byte == 0x00:
            output.append(0)  # Literal 0
        else:
            if pos >= len(data):
                break
            count = byte
            color = data[pos]
            pos += 1
            output.extend([color] * count)

    return bytes(output[:max_pixels])


def try_decode_v3(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Simple copy - no RLE."""
    max_pixels = width * height
    return data[offset:offset + max_pixels]


def try_decode_v4(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Copy with nibble unpacking (4-bit per pixel)."""
    output = bytearray()
    max_pixels = width * height
    pos = offset

    while pos < len(data) and len(output) < max_pixels:
        byte = data[pos]
        pos += 1
        # Unpack two 4-bit values
        output.append((byte >> 4) & 0x0F)
        if len(output) < max_pixels:
            output.append(byte & 0x0F)

    return bytes(output[:max_pixels])


def try_decode_v5(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Line-by-line with row compression indicator."""
    output = bytearray()
    pos = offset

    for row in range(height):
        if pos >= len(data):
            break

        # Check for row marker
        first_byte = data[pos]

        if first_byte == 0x00:
            # Copy previous row or fill with 0
            if len(output) >= width:
                output.extend(output[-width:])
            else:
                output.extend([0] * width)
            pos += 1
        else:
            # Read width bytes directly
            row_data = data[pos:pos + width]
            output.extend(row_data)
            pos += len(row_data)

    return bytes(output[:width * height])


def try_decode_v6(data: bytes, offset: int, width: int, height: int) -> bytes:
    """Skip cel header (9 bytes) then raw data."""
    header_offset = offset + 9
    max_pixels = width * height
    return data[header_offset:header_offset + max_pixels]


def render_with_palette(pixels: bytes, palette: Palette, width: int, height: int) -> Image.Image:
    """Create image from pixel data and palette."""
    img = Image.new('P', (width, height))
    img.putdata(list(pixels[:width * height]))
    img.putpalette(palette.to_pil_palette())
    return img.convert('RGB')


def analyze_pic(game_path: Path, pic_num: int):
    """Analyze and try different decodings."""
    reader = ResourceReader(game_path)
    data = reader.read_pic(pic_num)

    if not data:
        print(f"Failed to read PIC #{pic_num}")
        return

    print(f"PIC #{pic_num}: {len(data)} bytes")

    # Load palette
    palette = Palette()
    palette.load_from_pic_data(data, offset=13)

    # Find bitmap offset
    bitmap_offset = find_bitmap_offset(data)
    print(f"Bitmap offset: {bitmap_offset}")
    print(f"Bitmap size: {len(data) - bitmap_offset} bytes")

    # Dump first 64 bytes of bitmap
    print("\nBitmap data:")
    hex_dump(data, bitmap_offset, 64)

    # Try different decodings
    width, height = 320, 200

    decoders = [
        ("v1_rle_80", try_decode_v1),
        ("v2_rle_inverted", try_decode_v2),
        ("v3_raw_copy", try_decode_v3),
        ("v4_nibbles", try_decode_v4),
        ("v5_line_marker", try_decode_v5),
        ("v6_skip_header", try_decode_v6),
    ]

    for name, decoder in decoders:
        pixels = decoder(data, bitmap_offset, width, height)
        if len(pixels) >= width * height:
            img = render_with_palette(pixels, palette, width, height)
            output_path = f"extracts/debug_{name}_{pic_num}.png"
            img.save(output_path)

            # Count unique colors
            unique = len(set(pixels[:width * height]))
            print(f"\n{name}: {unique} unique colors, saved to {output_path}")
        else:
            print(f"\n{name}: only {len(pixels)} pixels (need {width * height})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_rle.py <game_path> <pic_number>")
        sys.exit(1)

    game_path = Path(sys.argv[1])
    pic_num = int(sys.argv[2])
    analyze_pic(game_path, pic_num)
