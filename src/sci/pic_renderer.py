"""PIC resource renderer for SCI games."""

import struct
from typing import Optional
from PIL import Image

from .palette import Palette


# PIC drawing opcodes
OP_SET_COLOR = 0xF0
OP_DISABLE_VISUAL = 0xF1
OP_SET_PRIORITY = 0xF2
OP_DISABLE_PRIORITY = 0xF3
OP_SHORT_LINES = 0xF4
OP_MEDIUM_LINES = 0xF5
OP_LONG_LINES = 0xF6
OP_SHORT_RELATIVE = 0xF7
OP_FILL = 0xF8
OP_SET_PATTERN = 0xF9
OP_SHORT_PATTERN = 0xFA
OP_MEDIUM_PATTERN = 0xFB
OP_LONG_PATTERN = 0xFC
OP_EXTENDED = 0xFE
OP_END = 0xFF

# Image dimensions
WIDTH = 320
HEIGHT = 200


class PicRenderer:
    """Renderer for SCI PIC resources."""

    def __init__(self):
        self.visual = bytearray(WIDTH * HEIGHT)  # Visual layer
        self.palette = Palette()
        self.current_color = 0
        self.pos = 0  # Current position in data

    def render(self, data: bytes) -> Optional[Image.Image]:
        """Render PIC data to a PIL Image.

        Args:
            data: Decompressed PIC resource data

        Returns:
            PIL Image in RGB mode, or None if rendering failed
        """
        # SCI1 VGA PIC structure:
        # - Bytes 0-12: Header (13 bytes)
        # - Bytes 13-1036: Palette (256 * 4 = 1024 bytes)
        # - Bytes 1037+: Vector commands ending with 0xFF
        # - After 0xFF: Bitmap data (RLE encoded)

        if len(data) < 1037:  # Minimum size for header + palette
            return None

        # Load palette (offset 13, 1024 bytes = 256 * 4)
        self.palette.load_from_pic_data(data, offset=13)

        # Find end of drawing commands by properly parsing them
        commands_start = 1037
        bitmap_offset = self._find_bitmap_offset(data, commands_start)

        if bitmap_offset is None or bitmap_offset >= len(data):
            # No bitmap found, try pure vector rendering
            self._execute_commands(data, commands_start)
        else:
            # Decode RLE bitmap
            bitmap_size = len(data) - bitmap_offset
            if bitmap_size > 0:
                self._decode_rle_bitmap(data, bitmap_offset)
            else:
                self._execute_commands(data, commands_start)

        # Convert to PIL Image
        return self._to_image()

    def _find_bitmap_offset(self, data: bytes, start: int) -> Optional[int]:
        """Find the offset where bitmap data starts by parsing vector commands."""
        pos = start

        # Opcodes with fixed argument lengths
        fixed_opcodes = {
            OP_SET_COLOR: 1,
            OP_DISABLE_VISUAL: 0,
            OP_SET_PRIORITY: 1,
            OP_DISABLE_PRIORITY: 0,
            OP_SET_PATTERN: 1,
        }

        # Opcodes with variable argument lengths (read until next opcode)
        variable_opcodes = {
            OP_SHORT_LINES, OP_MEDIUM_LINES, OP_LONG_LINES,
            OP_SHORT_RELATIVE, OP_FILL, OP_SHORT_PATTERN,
            OP_MEDIUM_PATTERN, OP_LONG_PATTERN, OP_EXTENDED
        }

        while pos < len(data):
            opcode = data[pos]

            if opcode == OP_END:
                return pos + 1

            if opcode in fixed_opcodes:
                pos += 1 + fixed_opcodes[opcode]
            elif opcode in variable_opcodes:
                pos += 1
                # Read argument bytes until we hit an opcode (>= 0xF0)
                while pos < len(data) and data[pos] < 0xF0:
                    pos += 1
            elif opcode >= 0xF0:
                # Unknown opcode, skip it
                pos += 1
            else:
                # Not an opcode, something is wrong
                break

        return None

    def _decode_rle_bitmap(self, data: bytes, offset: int) -> None:
        """Decode RLE-compressed bitmap data.

        SCI1 VGA RLE format:
        - bytes 0x00-0x7F: literal pixel (palette index)
        - bytes 0x80-0xFE: run of (byte - 0x80) pixels of the next byte's color
        - byte 0xFF: typically marks end of data
        """
        pos = offset
        output_pos = 0
        max_pixels = WIDTH * HEIGHT

        while pos < len(data) and output_pos < max_pixels:
            byte = data[pos]
            pos += 1

            if byte < 0x80:
                # Literal pixel
                self.visual[output_pos] = byte
                output_pos += 1
            elif byte == 0xFF:
                # End marker - check if we're close to done
                if output_pos >= max_pixels - 100:
                    break
                # Otherwise treat as literal (some images use 0xFF as a color)
                self.visual[output_pos] = byte
                output_pos += 1
            else:
                # RLE run: repeat next byte (byte - 0x80) times
                if pos >= len(data):
                    break
                count = byte - 0x80
                if count == 0:
                    count = 128  # 0x80 special case
                color = data[pos]
                pos += 1

                end_pos = min(output_pos + count, max_pixels)
                while output_pos < end_pos:
                    self.visual[output_pos] = color
                    output_pos += 1

    def _execute_commands(self, data: bytes, start_offset: int) -> None:
        """Execute PIC drawing commands (for vector-based PICs)."""
        self.pos = start_offset

        while self.pos < len(data):
            opcode = data[self.pos]
            self.pos += 1

            if opcode == OP_END:
                break
            elif opcode == OP_SET_COLOR:
                if self.pos < len(data):
                    self.current_color = data[self.pos]
                    self.pos += 1
            elif opcode == OP_FILL:
                self._do_fill(data)
            elif opcode == OP_LONG_LINES:
                self._do_long_lines(data)
            elif opcode == OP_SHORT_LINES:
                self._do_short_lines(data)
            elif opcode in (OP_DISABLE_VISUAL, OP_SET_PRIORITY, OP_DISABLE_PRIORITY):
                # These affect layers we don't care about for extraction
                if opcode == OP_SET_PRIORITY and self.pos < len(data):
                    self.pos += 1  # Skip priority color
            elif opcode == OP_EXTENDED:
                self._do_extended(data)
            elif opcode >= 0xF0:
                # Unknown opcode, try to continue
                pass
            else:
                # Byte < 0xF0 might be part of coordinate data
                self.pos -= 1  # Put byte back
                break

    def _do_fill(self, data: bytes) -> None:
        """Handle fill opcode."""
        while self.pos + 1 < len(data):
            if data[self.pos] >= 0xF0:
                break
            x = data[self.pos]
            y = data[self.pos + 1]
            self.pos += 2

            if x >= 0xF0:
                self.pos -= 2
                break

            self._flood_fill(x, y, self.current_color)

    def _do_long_lines(self, data: bytes) -> None:
        """Handle long lines opcode (absolute coordinates)."""
        if self.pos + 1 >= len(data):
            return

        x1 = data[self.pos]
        y1 = data[self.pos + 1]
        self.pos += 2

        while self.pos + 1 < len(data):
            if data[self.pos] >= 0xF0:
                break
            x2 = data[self.pos]
            y2 = data[self.pos + 1]
            self.pos += 2

            self._draw_line(x1, y1, x2, y2, self.current_color)
            x1, y1 = x2, y2

    def _do_short_lines(self, data: bytes) -> None:
        """Handle short lines opcode (relative coordinates)."""
        if self.pos + 1 >= len(data):
            return

        x = data[self.pos]
        y = data[self.pos + 1]
        self.pos += 2

        while self.pos < len(data):
            if data[self.pos] >= 0xF0:
                break
            delta = data[self.pos]
            self.pos += 1

            dx = (delta >> 4) & 0x0F
            dy = delta & 0x0F

            # Sign extend
            if dx > 7:
                dx -= 16
            if dy > 7:
                dy -= 16

            x2 = x + dx
            y2 = y + dy

            self._draw_line(x, y, x2, y2, self.current_color)
            x, y = x2, y2

    def _do_extended(self, data: bytes) -> None:
        """Handle extended opcode (palette, embedded views)."""
        if self.pos >= len(data):
            return

        sub_opcode = data[self.pos]
        self.pos += 1

        if sub_opcode == 0x00:
            # Set palette entry
            if self.pos + 3 < len(data):
                index = data[self.pos]
                r = data[self.pos + 1]
                g = data[self.pos + 2]
                b = data[self.pos + 3]
                self.pos += 4
                self.palette.colors[index] = (r * 4, g * 4, b * 4)
        elif sub_opcode == 0x01:
            # Set full palette - skip 256*3 bytes
            self.pos += 256 * 3
        # Other sub-opcodes handled as needed

    def _draw_line(self, x1: int, y1: int, x2: int, y2: int, color: int) -> None:
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < WIDTH and 0 <= y1 < HEIGHT:
                self.visual[y1 * WIDTH + x1] = color

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _flood_fill(self, x: int, y: int, color: int) -> None:
        """Simple flood fill."""
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return

        target_color = self.visual[y * WIDTH + x]
        if target_color == color:
            return

        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= WIDTH or cy < 0 or cy >= HEIGHT:
                continue
            if self.visual[cy * WIDTH + cx] != target_color:
                continue

            self.visual[cy * WIDTH + cx] = color
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

    def _to_image(self) -> Image.Image:
        """Convert visual buffer to PIL Image."""
        # Create indexed image
        img = Image.new('P', (WIDTH, HEIGHT))
        img.putdata(list(self.visual))
        img.putpalette(self.palette.to_pil_palette())

        # Convert to RGB
        return img.convert('RGB')


def render_pic(data: bytes) -> Optional[Image.Image]:
    """Convenience function to render PIC data."""
    renderer = PicRenderer()
    return renderer.render(data)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python pic_renderer.py <path/to/game> <pic_number>")
        sys.exit(1)

    # Import here to avoid circular imports
    from .resource_reader import ResourceReader

    game_path = Path(sys.argv[1])
    pic_num = int(sys.argv[2])

    reader = ResourceReader(game_path)
    data = reader.read_pic(pic_num)

    if data:
        print(f"Read PIC #{pic_num}: {len(data)} bytes")
        img = render_pic(data)
        if img:
            output_path = f"pic_{pic_num}.png"
            img.save(output_path)
            print(f"Saved to {output_path}")
        else:
            print("Failed to render PIC")
    else:
        print(f"Failed to read PIC #{pic_num}")
