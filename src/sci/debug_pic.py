"""Debug script to examine PIC data structure."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sci.resource_reader import ResourceReader


def hex_dump(data: bytes, offset: int, length: int, label: str) -> None:
    """Print hex dump of data section."""
    print(f"\n{label} (offset {offset}, {length} bytes):")
    end = min(offset + length, len(data))
    for i in range(offset, end, 16):
        hex_part = " ".join(f"{data[j]:02X}" for j in range(i, min(i + 16, end)))
        ascii_part = "".join(
            chr(data[j]) if 32 <= data[j] < 127 else "."
            for j in range(i, min(i + 16, end))
        )
        print(f"  {i:5d}: {hex_part:<48} {ascii_part}")


def parse_vector_commands(data: bytes, start: int) -> int:
    """Parse vector commands and return the offset after OP_END."""
    pos = start
    print(f"\nParsing vector commands from offset {start}:")

    opcodes = {
        0xF0: ("SET_COLOR", 1),      # 1 byte: color
        0xF1: ("DISABLE_VISUAL", 0),
        0xF2: ("SET_PRIORITY", 1),   # 1 byte: priority
        0xF3: ("DISABLE_PRIORITY", 0),
        0xF4: ("SHORT_LINES", -1),   # Variable
        0xF5: ("MEDIUM_LINES", -1),
        0xF6: ("LONG_LINES", -1),
        0xF7: ("SHORT_RELATIVE", -1),
        0xF8: ("FILL", -1),
        0xF9: ("SET_PATTERN", 1),    # 1 byte: pattern
        0xFA: ("SHORT_PATTERN", -1),
        0xFB: ("MEDIUM_PATTERN", -1),
        0xFC: ("LONG_PATTERN", -1),
        0xFE: ("EXTENDED", -1),
        0xFF: ("END", 0),
    }

    cmd_count = 0
    while pos < len(data) and cmd_count < 100:
        opcode = data[pos]

        if opcode == 0xFF:
            print(f"  {pos}: 0xFF END")
            return pos + 1

        if opcode in opcodes:
            name, fixed_len = opcodes[opcode]
            if fixed_len >= 0:
                # Fixed length command
                args = data[pos + 1:pos + 1 + fixed_len] if fixed_len > 0 else b""
                print(f"  {pos}: 0x{opcode:02X} {name} {args.hex() if args else ''}")
                pos += 1 + fixed_len
            else:
                # Variable length - need to parse until next opcode
                print(f"  {pos}: 0x{opcode:02X} {name}", end="")
                pos += 1
                arg_bytes = []
                while pos < len(data) and data[pos] < 0xF0:
                    arg_bytes.append(data[pos])
                    pos += 1
                print(f" [{len(arg_bytes)} arg bytes]")
        else:
            # Not a recognized opcode
            print(f"  {pos}: 0x{opcode:02X} (unknown, stopping)")
            break

        cmd_count += 1

    return pos


def try_rle_decode(data: bytes, start: int, width: int, height: int) -> tuple[bool, bytes]:
    """Try to decode RLE bitmap data."""
    pos = start
    output = bytearray()
    max_pixels = width * height

    print(f"\nTrying RLE decode from offset {start}:")
    print(f"  Target size: {max_pixels} pixels")

    # SCI1 VGA RLE format:
    # - bytes 0x00-0x7F: literal byte (palette index)
    # - bytes 0x80-0xFE: run length = (byte - 0x80), next byte is color
    # - byte 0xFF: end marker or special

    while pos < len(data) and len(output) < max_pixels:
        byte = data[pos]
        pos += 1

        if byte < 0x80:
            # Literal pixel
            output.append(byte)
        elif byte == 0xFF:
            # Might be end marker
            if len(output) >= max_pixels - 100:
                break
            # Otherwise treat as literal
            output.append(byte)
        else:
            # RLE: repeat next byte (byte - 0x80) times
            if pos >= len(data):
                break
            count = byte - 0x80
            if count == 0:
                count = 128  # Special case
            color = data[pos]
            pos += 1
            output.extend([color] * count)

    print(f"  Decoded {len(output)} pixels from {pos - start} bytes")
    return len(output) >= max_pixels * 0.9, bytes(output)


def analyze_pic(game_path: Path, pic_num: int) -> None:
    """Analyze a PIC resource structure."""
    reader = ResourceReader(game_path)
    data = reader.read_pic(pic_num)

    if not data:
        print(f"Failed to read PIC #{pic_num}")
        return

    print(f"PIC #{pic_num}: {len(data)} bytes decompressed")

    # Dump header (first 13 bytes)
    hex_dump(data, 0, 16, "Header")

    # Dump palette start
    hex_dump(data, 13, 32, "Palette start (offset 13)")

    # Check palette values - look for non-zero entries
    print("\nPalette analysis:")
    non_zero_entries = 0
    for i in range(256):
        base = 13 + i * 4
        if base + 4 > len(data):
            break
        flag, r, g, b = data[base], data[base + 1], data[base + 2], data[base + 3]
        if flag != 0 or r != 0 or g != 0 or b != 0:
            non_zero_entries += 1
            if non_zero_entries <= 10:
                print(f"  [{i}]: flag={flag}, RGB=({r}, {g}, {b})")
    print(f"  Total non-zero palette entries: {non_zero_entries}")

    # Dump area after palette (offset 1037)
    hex_dump(data, 1037, 64, "After palette (offset 1037)")

    # Parse vector commands
    bitmap_offset = parse_vector_commands(data, 1037)

    print(f"\nBitmap starts at offset {bitmap_offset}")
    print(f"Bitmap size: {len(data) - bitmap_offset} bytes")

    hex_dump(data, bitmap_offset, 64, f"Bitmap start (offset {bitmap_offset})")

    # Try RLE decode
    success, decoded = try_rle_decode(data, bitmap_offset, 320, 200)
    if success:
        print(f"\nRLE decode successful! Got {len(decoded)} pixels")

        # Show color distribution in decoded data
        freq = {}
        for b in decoded[:64000]:
            freq[b] = freq.get(b, 0) + 1
        top_10 = sorted(freq.items(), key=lambda x: -x[1])[:10]
        print("\nTop 10 colors in decoded image:")
        for val, count in top_10:
            pct = count * 100 / min(len(decoded), 64000)
            print(f"  Color {val:3d}: {count:6d} pixels ({pct:.1f}%)")
    else:
        print("\nRLE decode incomplete")

        # Try raw copy as fallback
        print("\nTrying raw byte copy:")
        raw_data = data[bitmap_offset:bitmap_offset + 64000]
        freq = {}
        for b in raw_data:
            freq[b] = freq.get(b, 0) + 1
        top_10 = sorted(freq.items(), key=lambda x: -x[1])[:10]
        print("Top 10 bytes in raw bitmap area:")
        for val, count in top_10:
            pct = count * 100 / len(raw_data)
            print(f"  0x{val:02X} ({val:3d}): {count:6d} ({pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_pic.py <game_path> <pic_number>")
        sys.exit(1)

    game_path = Path(sys.argv[1])
    pic_num = int(sys.argv[2])
    analyze_pic(game_path, pic_num)
