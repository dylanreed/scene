"""SCI1 VIEW renderer - parse and render sprite/animation resources.

VIEW resources contain animated sprites organized as:
- View (container)
  - Loop (animation direction, e.g., facing left/right)
    - Cel (individual animation frame)

Based on SCICompanion and ScummVM SCI engine implementations.
"""
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image

from .palette import Palette


@dataclass
class Cel:
    """Individual animation frame."""
    width: int
    height: int
    displacement_x: int
    displacement_y: int
    transparent_color: int
    pixels: bytes  # Row-major, top-to-bottom


@dataclass
class Loop:
    """Animation loop (direction)."""
    cels: List[Cel]
    is_mirror: bool = False
    mirror_of: int = -1  # Which loop this mirrors, if any


@dataclass
class View:
    """Complete VIEW resource."""
    loops: List[Loop]
    palette_offset: int = 0


class ViewRenderer:
    """Render SCI1 VIEW resources to images.

    Supports SCI1 VGA format where:
    - Header contains loop count, mirror mask, palette offset, loop offsets
    - Each loop has cel count and cel offsets
    - Each cel has dimensions, displacement, transparent color, and RLE data

    RLE encoding (VGA):
    - Byte format: XXYYYYYY where XX is opcode, YYYYYY is count
    - 00xxxxxx: Copy next count bytes literally
    - 10xxxxxx: Fill count pixels with next byte value
    - 11xxxxxx: Skip count pixels (transparent)
    """

    def __init__(self, palette: Palette):
        """Initialize with color palette."""
        self.palette = palette

    def parse(self, data: bytes) -> View:
        """Parse VIEW resource data into View structure."""
        if len(data) < 8:
            raise ValueError("VIEW data too short")

        # Detect format based on data
        # SCI1.1 has header size in first word, usually small (like 0x0E)
        # SCI1 VGA has loop count in first byte

        first_word = struct.unpack_from('<H', data, 0)[0]

        if first_word < 0x20 and data[2] < 20:
            # Likely SCI1.1 format (header size small, loop count reasonable)
            return self._parse_vga11(data)
        else:
            # SCI1 VGA format
            return self._parse_vga(data)

    def _parse_vga(self, data: bytes) -> View:
        """Parse SCI1 VGA VIEW format.

        Header:
        - Byte 0: Loop count
        - Byte 1: Flags (0x80 = has cel headers, 0x40 = uncompressed)
        - Bytes 2-3: Mirror mask (bit N = loop N is mirrored)
        - Bytes 4-5: Unknown/version
        - Bytes 6-7: Palette offset (0 if none)
        - Bytes 8+: Loop offsets (2 bytes each)
        """
        loop_count = data[0]
        flags = data[1]
        mirror_mask = struct.unpack_from('<H', data, 2)[0]
        palette_offset = struct.unpack_from('<H', data, 6)[0]

        is_compressed = not (flags & 0x40)

        loops = []
        for i in range(loop_count):
            loop_offset = struct.unpack_from('<H', data, 8 + i * 2)[0]

            # Check if this loop is a mirror
            is_mirror = bool(mirror_mask & (1 << i))

            if is_mirror and loops:
                # Find the loop this mirrors (first non-mirror loop before it)
                mirror_of = -1
                for j in range(i - 1, -1, -1):
                    if not loops[j].is_mirror:
                        mirror_of = j
                        break

                if mirror_of >= 0:
                    # Create mirrored copy
                    mirrored_cels = []
                    for cel in loops[mirror_of].cels:
                        mirrored_cels.append(self._mirror_cel(cel))
                    loops.append(Loop(cels=mirrored_cels, is_mirror=True, mirror_of=mirror_of))
                    continue

            # Parse loop data
            loop = self._parse_loop_vga(data, loop_offset, is_compressed)
            loops.append(loop)

        return View(loops=loops, palette_offset=palette_offset)

    def _parse_vga11(self, data: bytes) -> View:
        """Parse SCI1.1 VGA VIEW format.

        Header:
        - Bytes 0-1: Header size (add 2 for actual)
        - Byte 2: Loop count
        - Byte 3: Flags (scalability)
        - Bytes 4-5: Unknown/version
        - Bytes 6-7: Unknown
        - Bytes 8-9: Palette offset
        - Byte 10: Loop header size
        - Byte 11: Cel header size
        - Bytes 12+: Loop offsets
        """
        header_size = struct.unpack_from('<H', data, 0)[0] + 2
        loop_count = data[2]
        palette_offset = struct.unpack_from('<H', data, 8)[0]
        loop_header_size = data[10] if len(data) > 10 else 2
        cel_header_size = data[11] if len(data) > 11 else 32

        loops = []
        for i in range(loop_count):
            # Loop offset table starts after header
            offset_pos = header_size + i * 2
            if offset_pos + 2 > len(data):
                break
            loop_offset = struct.unpack_from('<H', data, offset_pos)[0]

            loop = self._parse_loop_vga11(data, loop_offset, cel_header_size)
            loops.append(loop)

        return View(loops=loops, palette_offset=palette_offset)

    def _parse_loop_vga(self, data: bytes, offset: int, is_compressed: bool) -> Loop:
        """Parse a loop in VGA format.

        Loop structure:
        - Bytes 0-1: Cel count (LE word)
        - Bytes 2-3: Unknown/flags
        - Bytes 4+: Cel offsets (2 bytes each), ABSOLUTE from VIEW start
        """
        if offset + 4 > len(data):
            return Loop(cels=[])

        cel_count = struct.unpack_from('<H', data, offset)[0]

        # Sanity check
        if cel_count > 100:
            return Loop(cels=[])

        cels = []
        for i in range(cel_count):
            cel_offset_pos = offset + 4 + i * 2  # Skip 4 bytes header
            if cel_offset_pos + 2 > len(data):
                break

            # Cel offset is ABSOLUTE from VIEW start
            cel_offset = struct.unpack_from('<H', data, cel_offset_pos)[0]
            if cel_offset >= len(data):
                continue

            cel = self._parse_cel_vga(data, cel_offset, is_compressed)
            if cel:
                cels.append(cel)

        return Loop(cels=cels)

    def _parse_loop_vga11(self, data: bytes, offset: int, cel_header_size: int) -> Loop:
        """Parse a loop in VGA 1.1 format."""
        if offset + 4 > len(data):
            return Loop(cels=[])

        # VGA11 loop header: flags(1), cel_count(1), unknown(2), then cel offsets
        cel_count = data[offset + 1] if offset + 1 < len(data) else 0

        cels = []
        for i in range(cel_count):
            cel_offset_pos = offset + 4 + i * 2
            if cel_offset_pos + 2 > len(data):
                break

            cel_offset = offset + struct.unpack_from('<H', data, cel_offset_pos)[0]
            cel = self._parse_cel_vga11(data, cel_offset, cel_header_size)
            if cel:
                cels.append(cel)

        return Loop(cels=cels)

    def _parse_cel_vga(self, data: bytes, offset: int, is_compressed: bool) -> Optional[Cel]:
        """Parse a cel in VGA format.

        Cel header (7 bytes):
        - Bytes 0-1: Width
        - Bytes 2-3: Height
        - Byte 4: Displacement X (signed)
        - Byte 5: Displacement Y (signed)
        - Byte 6: Transparent color
        - Bytes 7+: RLE data
        """
        if offset + 7 > len(data):
            return None

        width = struct.unpack_from('<H', data, offset)[0]
        height = struct.unpack_from('<H', data, offset + 2)[0]
        disp_x = struct.unpack_from('b', data, offset + 4)[0]  # signed
        disp_y = struct.unpack_from('b', data, offset + 5)[0]  # signed
        transparent = data[offset + 6]

        # Sanity check
        if width > 1024 or height > 1024 or width == 0 or height == 0:
            return None

        rle_data = data[offset + 7:]

        if is_compressed:
            pixels = self._decode_rle_vga(rle_data, width, height, transparent)
        else:
            # Uncompressed - just copy
            size = width * height
            pixels = rle_data[:size] if len(rle_data) >= size else rle_data + bytes(size - len(rle_data))

        return Cel(
            width=width,
            height=height,
            displacement_x=disp_x,
            displacement_y=disp_y,
            transparent_color=transparent,
            pixels=pixels
        )

    def _parse_cel_vga11(self, data: bytes, offset: int, header_size: int) -> Optional[Cel]:
        """Parse a cel in VGA 1.1 format.

        VGA11 cel header (32 bytes typical):
        - Bytes 0-1: Width
        - Bytes 2-3: Height
        - Bytes 4-5: Displacement X (signed word)
        - Bytes 6-7: Displacement Y (signed word)
        - Byte 8: Transparent color
        - Byte 9: Compressed flag
        - Bytes 10-11: Unknown
        - Bytes 12-15: RLE offset from cel start
        - Bytes 16-19: Literal offset from cel start
        - Bytes 20+: Color info, etc.
        """
        if offset + header_size > len(data):
            return None

        width = struct.unpack_from('<H', data, offset)[0]
        height = struct.unpack_from('<H', data, offset + 2)[0]
        disp_x = struct.unpack_from('<h', data, offset + 4)[0]  # signed word
        disp_y = struct.unpack_from('<h', data, offset + 6)[0]
        transparent = data[offset + 8]

        if width > 1024 or height > 1024 or width == 0 or height == 0:
            return None

        # Get RLE data offset
        if header_size >= 16:
            rle_offset = struct.unpack_from('<I', data, offset + 12)[0]
            rle_data = data[offset + rle_offset:] if rle_offset else data[offset + header_size:]
        else:
            rle_data = data[offset + header_size:]

        pixels = self._decode_rle_vga(rle_data, width, height, transparent)

        return Cel(
            width=width,
            height=height,
            displacement_x=disp_x,
            displacement_y=disp_y,
            transparent_color=transparent,
            pixels=pixels
        )

    def _decode_rle_vga(self, data: bytes, width: int, height: int, transparent: int) -> bytes:
        """Decode VGA RLE compressed cel data.

        RLE format (processes bottom-to-top, we flip to top-to-bottom):
        - Each byte: XXYYYYYY
        - XX=00: Copy next YYYYYY bytes literally
        - XX=10 (0x80): Fill YYYYYY pixels with next byte
        - XX=11 (0xC0): Skip YYYYYY pixels (transparent)
        """
        output = bytearray(width * height)
        # Fill with transparent color first
        for i in range(len(output)):
            output[i] = transparent

        pos = 0  # Input position
        out_pos = 0  # Output position
        total_pixels = width * height

        while pos < len(data) and out_pos < total_pixels:
            if pos >= len(data):
                break

            cmd = data[pos]
            pos += 1

            opcode = cmd & 0xC0
            count = cmd & 0x3F

            if count == 0:
                count = 64  # 0 means 64

            if opcode == 0x00:
                # Copy literal bytes
                for _ in range(count):
                    if pos < len(data) and out_pos < total_pixels:
                        output[out_pos] = data[pos]
                        pos += 1
                        out_pos += 1
                    else:
                        break

            elif opcode == 0x80:
                # Fill with color
                if pos < len(data):
                    color = data[pos]
                    pos += 1
                    for _ in range(count):
                        if out_pos < total_pixels:
                            output[out_pos] = color
                            out_pos += 1

            elif opcode == 0xC0:
                # Skip (transparent) - already filled with transparent
                out_pos += count

            else:
                # 0x40 - treat as literal copy for safety
                for _ in range(count):
                    if pos < len(data) and out_pos < total_pixels:
                        output[out_pos] = data[pos]
                        pos += 1
                        out_pos += 1

        # VGA views are stored bottom-to-top, flip vertically
        flipped = bytearray(width * height)
        for y in range(height):
            src_row = (height - 1 - y) * width
            dst_row = y * width
            flipped[dst_row:dst_row + width] = output[src_row:src_row + width]

        return bytes(flipped)

    def _mirror_cel(self, cel: Cel) -> Cel:
        """Create horizontally mirrored copy of a cel."""
        mirrored = bytearray(cel.width * cel.height)

        for y in range(cel.height):
            for x in range(cel.width):
                src_idx = y * cel.width + x
                dst_idx = y * cel.width + (cel.width - 1 - x)
                mirrored[dst_idx] = cel.pixels[src_idx]

        return Cel(
            width=cel.width,
            height=cel.height,
            displacement_x=-cel.displacement_x - cel.width,
            displacement_y=cel.displacement_y,
            transparent_color=cel.transparent_color,
            pixels=bytes(mirrored)
        )

    def render_cel(self, cel: Cel, transparent_bg: bool = True) -> Image.Image:
        """Render a single cel to an RGBA image."""
        if transparent_bg:
            img = Image.new('RGBA', (cel.width, cel.height), (0, 0, 0, 0))
        else:
            img = Image.new('RGB', (cel.width, cel.height), (0, 0, 0))

        pixels = []
        for y in range(cel.height):
            for x in range(cel.width):
                idx = y * cel.width + x
                color_idx = cel.pixels[idx] if idx < len(cel.pixels) else 0

                if transparent_bg and color_idx == cel.transparent_color:
                    pixels.append((0, 0, 0, 0))
                elif 0 <= color_idx < len(self.palette):
                    r, g, b = self.palette[color_idx]
                    if transparent_bg:
                        pixels.append((r, g, b, 255))
                    else:
                        pixels.append((r, g, b))
                else:
                    if transparent_bg:
                        pixels.append((0, 0, 0, 0))
                    else:
                        pixels.append((0, 0, 0))

        img.putdata(pixels)
        return img

    def render_loop_strip(self, loop: Loop, transparent_bg: bool = True) -> Image.Image:
        """Render all cels in a loop as a horizontal strip."""
        if not loop.cels:
            return Image.new('RGBA' if transparent_bg else 'RGB', (1, 1))

        # Calculate total width and max height
        total_width = sum(cel.width for cel in loop.cels)
        max_height = max(cel.height for cel in loop.cels)

        if transparent_bg:
            strip = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))
        else:
            strip = Image.new('RGB', (total_width, max_height), (0, 0, 0))

        x_offset = 0
        for cel in loop.cels:
            cel_img = self.render_cel(cel, transparent_bg)
            # Center vertically
            y_offset = (max_height - cel.height) // 2
            strip.paste(cel_img, (x_offset, y_offset), cel_img if transparent_bg else None)
            x_offset += cel.width

        return strip

    def render_view_sheet(self, view: View, transparent_bg: bool = True) -> Image.Image:
        """Render entire view as a sprite sheet (loops stacked vertically)."""
        if not view.loops:
            return Image.new('RGBA' if transparent_bg else 'RGB', (1, 1))

        # Render each loop as a strip
        strips = []
        max_width = 0
        total_height = 0

        for loop in view.loops:
            strip = self.render_loop_strip(loop, transparent_bg)
            strips.append(strip)
            max_width = max(max_width, strip.width)
            total_height += strip.height

        # Combine strips
        if transparent_bg:
            sheet = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
        else:
            sheet = Image.new('RGB', (max_width, total_height), (0, 0, 0))

        y_offset = 0
        for strip in strips:
            sheet.paste(strip, (0, y_offset), strip if transparent_bg else None)
            y_offset += strip.height

        return sheet
