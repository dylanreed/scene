"""SCI1 PIC renderer - execute drawing commands to generate images."""
from typing import Tuple, List, Optional
from PIL import Image

from .palette import Palette


class PicRenderer:
    """Render SCI1 PIC resources to images.

    SCI1 VGA PICs can contain:
    - Vector drawing commands (lines, fills, patterns)
    - Embedded bitmap data with RLE compression
    - Embedded 256-color palette

    PIC opcodes:
    0xF0: SET_COLOR - Set visual drawing color
    0xF1: DISABLE_VISUAL - Stop drawing to visual layer
    0xF2: SET_PRIORITY - Set priority color
    0xF3: DISABLE_PRIORITY - Stop drawing to priority layer
    0xF4: SHORT_LINES - Relative lines, 1-byte coords
    0xF5: MEDIUM_LINES - Relative lines, mixed coords
    0xF6: LONG_LINES - Absolute lines, 2-byte coords
    0xF7: SHORT_RELATIVE - Short relative from last point
    0xF8: FILL - Flood fill at coordinates
    0xF9: SET_PATTERN - Set brush pattern/size
    0xFA: SHORT_PATTERN - Pattern at short coords
    0xFB: MEDIUM_PATTERN - Pattern at medium coords
    0xFC: LONG_PATTERN - Pattern at absolute coords
    0xFE: EXTENDED - Palette, embedded cels, etc.
    0xFF: END - End of PIC data

    Extended opcodes (0xFE):
    0x01: Embedded bitmap/cel
    0x02: Palette mapping table (256 bytes)
    0x04: Priority bands
    """

    WIDTH = 320
    HEIGHT = 200

    def __init__(self, palette: Palette):
        """Initialize renderer with palette."""
        self.palette = palette
        self.embedded_palette: Optional[List[Tuple[int, int, int]]] = None
        self.visual: List[int] = [0] * (self.WIDTH * self.HEIGHT)
        self.priority: List[int] = [0] * (self.WIDTH * self.HEIGHT)
        self.visual_color: int = 0
        self.priority_color: int = 0
        self.visual_enabled: bool = True
        self.priority_enabled: bool = True
        self.pattern_code: int = 0
        self.pattern_size: int = 0
        self.last_x: int = 0
        self.last_y: int = 0

    def render(self, data: bytes) -> Image.Image:
        """Render PIC data to an RGB image."""
        self._reset()

        # Check for SCI1 VGA format with embedded palette
        if self._is_vga_pic_with_embedded_data(data):
            return self._render_vga_embedded(data)

        # Fall back to vector command rendering
        self._execute(data)
        return self._create_image()

    def _is_vga_pic_with_embedded_data(self, data: bytes) -> bool:
        """Check if this is a VGA PIC with embedded palette and bitmap."""
        # VGA PICs start with 0xFE 0x02 (palette mapping) followed by
        # 256 bytes of indices (0x00, 0x01, 0x02, ..., 0xFF)
        if len(data) < 260:
            return False
        if data[0] != 0xFE or data[1] != 0x02:
            return False
        # Check if bytes 2-257 look like a sequential mapping table
        if data[2:10] == bytes([0, 1, 2, 3, 4, 5, 6, 7]):
            return True
        return False

    def _render_vga_embedded(self, data: bytes) -> Image.Image:
        """Render VGA PIC with embedded palette and bitmap.

        Format:
        - 0xFE 0x02 + 256-byte mapping table
        - 1024-byte embedded palette (256 * 4 bytes: flag, R, G, B)
        - Extended commands including 0xFE 0x01 (embedded bitmap)
        """
        # Extract embedded palette (starts at position 258)
        palette_start = 258
        self.embedded_palette = []
        for i in range(256):
            off = palette_start + i * 4
            if off + 3 < len(data):
                # Format: flag, R, G, B
                r = data[off + 1]
                g = data[off + 2]
                b = data[off + 3]
                self.embedded_palette.append((r, g, b))
            else:
                self.embedded_palette.append((0, 0, 0))

        # Find and decode embedded bitmap (0xFE 0x01)
        bitmap_start = self._find_embedded_bitmap(data, palette_start + 1024)
        if bitmap_start > 0:
            return self._decode_rle_bitmap(data, bitmap_start)

        # Fall back to empty image
        return Image.new('RGB', (self.WIDTH, self.HEIGHT), (0, 0, 0))

    def _find_embedded_bitmap(self, data: bytes, start: int) -> int:
        """Find position of embedded bitmap RLE data."""
        pos = start
        while pos < len(data) - 1:
            if data[pos] == 0xFE:
                sub = data[pos + 1]
                if sub == 0x01:
                    # Found embedded bitmap command
                    # Header format: 0xFE 0x01 + coordinates + cel info
                    # RLE data starts after header, look for first RLE opcode
                    search_start = pos + 8
                    for offset in range(search_start, min(search_start + 20, len(data))):
                        byte = data[offset]
                        # Look for valid RLE start (not 0x00 which could be header)
                        if byte >= 0x80 or (byte > 0 and byte < 0x40):
                            return offset
                    return pos + 12  # Default offset if pattern not found
                elif sub == 0x04:
                    # Priority bands - 14 bytes of data
                    pos += 2 + 14
                else:
                    pos += 2
            elif data[pos] == 0xFF:
                # END marker - continue searching
                pos += 1
            else:
                pos += 1
        return -1

    def _decode_rle_bitmap(self, data: bytes, start: int) -> Image.Image:
        """Decode RLE-compressed bitmap data.

        RLE format (same as VIEW cels):
        - Each byte: XXYYYYYY where XX is opcode, YYYYYY is count
        - 00: Copy next count bytes literally
        - 80: Fill count pixels with next byte
        - C0: Skip count pixels (transparent/black)
        """
        output = bytearray(self.WIDTH * self.HEIGHT)
        pos = start
        out_pos = 0
        total = self.WIDTH * self.HEIGHT

        while out_pos < total and pos < len(data):
            byte = data[pos]
            pos += 1

            op = byte & 0xC0
            count = byte & 0x3F
            if count == 0:
                count = 64  # 0 means 64

            if op == 0x00:  # Literal copy
                for _ in range(count):
                    if out_pos < total and pos < len(data):
                        output[out_pos] = data[pos]
                        pos += 1
                        out_pos += 1

            elif op == 0x80:  # Fill
                if pos < len(data):
                    color = data[pos]
                    pos += 1
                    for _ in range(count):
                        if out_pos < total:
                            output[out_pos] = color
                            out_pos += 1

            elif op == 0xC0:  # Skip (transparent)
                out_pos += count

            else:  # 0x40 - treat as literal
                for _ in range(count):
                    if out_pos < total and pos < len(data):
                        output[out_pos] = data[pos]
                        pos += 1
                        out_pos += 1

        # Create image with embedded palette
        img = Image.new('RGB', (self.WIDTH, self.HEIGHT))
        pixels = []
        palette = self.embedded_palette or [(0, 0, 0)] * 256

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                idx = output[y * self.WIDTH + x]
                if idx < len(palette):
                    pixels.append(palette[idx])
                else:
                    pixels.append((0, 0, 0))

        img.putdata(pixels)
        return img

    def _reset(self):
        """Reset canvas to initial state."""
        self.visual = [0] * (self.WIDTH * self.HEIGHT)
        self.priority = [0] * (self.WIDTH * self.HEIGHT)
        self.visual_color = 0
        self.priority_color = 0
        self.visual_enabled = True
        self.priority_enabled = True
        self.pattern_code = 0
        self.pattern_size = 0
        self.last_x = 0
        self.last_y = 0

    def _execute(self, data: bytes):
        """Execute PIC drawing commands."""
        pos = 0

        while pos < len(data):
            opcode = data[pos]
            pos += 1

            if opcode == 0xF0:  # SET_COLOR
                if pos < len(data):
                    self.visual_color = data[pos]
                    self.visual_enabled = True
                    pos += 1

            elif opcode == 0xF1:  # DISABLE_VISUAL
                self.visual_enabled = False

            elif opcode == 0xF2:  # SET_PRIORITY
                if pos < len(data):
                    self.priority_color = data[pos]
                    self.priority_enabled = True
                    pos += 1

            elif opcode == 0xF3:  # DISABLE_PRIORITY
                self.priority_enabled = False

            elif opcode == 0xF4:  # SHORT_LINES
                pos = self._draw_short_lines(data, pos)

            elif opcode == 0xF5:  # MEDIUM_LINES
                pos = self._draw_medium_lines(data, pos)

            elif opcode == 0xF6:  # LONG_LINES
                pos = self._draw_long_lines(data, pos)

            elif opcode == 0xF7:  # SHORT_RELATIVE
                pos = self._draw_short_relative(data, pos)

            elif opcode == 0xF8:  # FILL
                pos = self._do_fill(data, pos)

            elif opcode == 0xF9:  # SET_PATTERN
                if pos < len(data):
                    self.pattern_code = data[pos]
                    self.pattern_size = self.pattern_code & 0x07
                    pos += 1

            elif opcode == 0xFA:  # SHORT_PATTERN
                pos = self._draw_pattern_short(data, pos)

            elif opcode == 0xFB:  # MEDIUM_PATTERN
                pos = self._draw_pattern_medium(data, pos)

            elif opcode == 0xFC:  # LONG_PATTERN
                pos = self._draw_pattern_long(data, pos)

            elif opcode == 0xFE:  # EXTENDED
                pos = self._handle_extended(data, pos)

            elif opcode == 0xFF:  # END
                break

            elif opcode < 0xF0:
                # Not an opcode - might be data we're skipping over
                pass

    def _draw_pixel(self, x: int, y: int):
        """Draw a pixel at the given coordinates."""
        if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
            idx = y * self.WIDTH + x
            if self.visual_enabled:
                self.visual[idx] = self.visual_color
            if self.priority_enabled:
                self.priority[idx] = self.priority_color

    def _draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            self._draw_pixel(x, y)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _draw_long_lines(self, data: bytes, pos: int) -> int:
        """Draw absolute coordinate lines."""
        if pos + 2 > len(data):
            return pos

        # First point
        x = data[pos] | ((data[pos + 1] & 0xF0) << 4)
        y = ((data[pos + 1] & 0x0F) << 8) | data[pos + 2] if pos + 2 < len(data) else 0
        pos += 3

        self.last_x = x
        self.last_y = y

        while pos + 2 < len(data):
            byte0 = data[pos]
            if byte0 >= 0xF0:
                break

            x2 = data[pos] | ((data[pos + 1] & 0xF0) << 4)
            y2 = ((data[pos + 1] & 0x0F) << 8) | data[pos + 2]
            pos += 3

            self._draw_line(self.last_x, self.last_y, x2, y2)
            self.last_x = x2
            self.last_y = y2

        return pos

    def _draw_short_lines(self, data: bytes, pos: int) -> int:
        """Draw short relative lines."""
        if pos + 1 > len(data):
            return pos

        # First absolute point
        x = data[pos]
        y = data[pos + 1] if pos + 1 < len(data) else 0
        pos += 2

        self.last_x = x
        self.last_y = y

        while pos < len(data):
            byte0 = data[pos]
            if byte0 >= 0xF0:
                break

            # Sign extend 4-bit values
            dx = byte0 >> 4
            dy = byte0 & 0x0F
            if dx >= 8:
                dx -= 16
            if dy >= 8:
                dy -= 16

            x2 = self.last_x + dx
            y2 = self.last_y + dy
            pos += 1

            self._draw_line(self.last_x, self.last_y, x2, y2)
            self.last_x = x2
            self.last_y = y2

        return pos

    def _draw_medium_lines(self, data: bytes, pos: int) -> int:
        """Draw medium coordinate lines."""
        # Similar to short but with more precision
        return self._draw_short_lines(data, pos)

    def _draw_short_relative(self, data: bytes, pos: int) -> int:
        """Draw short relative lines from last point."""
        while pos < len(data):
            byte0 = data[pos]
            if byte0 >= 0xF0:
                break

            dx = (byte0 >> 4) - 8
            dy = (byte0 & 0x0F) - 8

            x2 = self.last_x + dx
            y2 = self.last_y + dy
            pos += 1

            self._draw_line(self.last_x, self.last_y, x2, y2)
            self.last_x = x2
            self.last_y = y2

        return pos

    def _do_fill(self, data: bytes, pos: int) -> int:
        """Perform flood fill."""
        while pos + 1 < len(data):
            byte0 = data[pos]
            if byte0 >= 0xF0:
                break

            x = data[pos]
            y = data[pos + 1] if pos + 1 < len(data) else 0
            pos += 2

            self._flood_fill(x, y)

        return pos

    def _flood_fill(self, start_x: int, start_y: int):
        """Stack-based flood fill."""
        if not (0 <= start_x < self.WIDTH and 0 <= start_y < self.HEIGHT):
            return

        start_idx = start_y * self.WIDTH + start_x
        target_color = self.visual[start_idx]

        if target_color == self.visual_color:
            return

        stack = [(start_x, start_y)]
        visited = set()

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            if not (0 <= x < self.WIDTH and 0 <= y < self.HEIGHT):
                continue

            idx = y * self.WIDTH + x
            if self.visual[idx] != target_color:
                continue

            visited.add((x, y))

            if self.visual_enabled:
                self.visual[idx] = self.visual_color
            if self.priority_enabled:
                self.priority[idx] = self.priority_color

            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

    def _draw_pattern_short(self, data: bytes, pos: int) -> int:
        """Draw pattern at short coordinates."""
        while pos + 1 < len(data):
            if data[pos] >= 0xF0:
                break
            x = data[pos]
            y = data[pos + 1]
            pos += 2
            self._draw_pattern(x, y)
        return pos

    def _draw_pattern_medium(self, data: bytes, pos: int) -> int:
        """Draw pattern at medium coordinates."""
        return self._draw_pattern_short(data, pos)

    def _draw_pattern_long(self, data: bytes, pos: int) -> int:
        """Draw pattern at absolute coordinates."""
        while pos + 2 < len(data):
            if data[pos] >= 0xF0:
                break
            x = data[pos] | ((data[pos + 1] & 0xF0) << 4)
            y = ((data[pos + 1] & 0x0F) << 8) | data[pos + 2]
            pos += 3
            self._draw_pattern(x, y)
        return pos

    def _draw_pattern(self, x: int, y: int):
        """Draw a pattern/brush at the given coordinates."""
        size = self.pattern_size
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx * dx + dy * dy <= size * size:
                    self._draw_pixel(x + dx, y + dy)

    def _handle_extended(self, data: bytes, pos: int) -> int:
        """Handle extended opcodes."""
        if pos >= len(data):
            return pos

        sub_opcode = data[pos]
        pos += 1

        if sub_opcode == 0x00:  # Set palette entry
            # Skip palette entry data
            if pos + 4 <= len(data):
                pos += 4

        elif sub_opcode == 0x01:  # Set full palette
            # Skip full palette data (768 or 1024 bytes)
            pass

        elif sub_opcode == 0x02:  # Set priority bands
            # Priority bands table - skip 14 * 2 bytes
            pos += 28 if pos + 28 <= len(data) else len(data) - pos

        # Other extended opcodes - try to skip gracefully
        return pos

    def _create_image(self) -> Image.Image:
        """Create PIL Image from visual canvas."""
        img = Image.new('RGB', (self.WIDTH, self.HEIGHT))
        pixels = []

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                idx = y * self.WIDTH + x
                color_idx = self.visual[idx]
                if 0 <= color_idx < len(self.palette):
                    r, g, b = self.palette[color_idx]
                    pixels.append((r, g, b))
                else:
                    pixels.append((0, 0, 0))

        img.putdata(pixels)
        return img
