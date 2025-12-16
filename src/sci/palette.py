"""VGA palette handling for SCI games."""

from typing import Optional


class Palette:
    """256-color VGA palette."""

    def __init__(self):
        # Default VGA palette (grayscale)
        self.colors = [(i, i, i) for i in range(256)]

    def load_from_pic_data(self, data: bytes, offset: int = 13) -> None:
        """Load palette from embedded PIC data.

        SCI1 PICs embed palette as 256 entries of 4 bytes each:
        - Byte 0: Flag (usually 0x01)
        - Bytes 1-3: R, G, B (0-255 range, or 0-63 VGA DAC range)
        """
        palette_size = 256 * 4
        if offset + palette_size > len(data):
            return

        for i in range(256):
            base = offset + i * 4
            flag = data[base]
            r = data[base + 1]
            g = data[base + 2]
            b = data[base + 3]

            # VGA DAC uses 0-63 range, convert to 0-255
            # But some games might use 0-255 directly
            # We detect based on max values
            self.colors[i] = (r, g, b)

        # Check if we need to scale from VGA DAC (0-63) to 0-255
        max_val = max(max(c) for c in self.colors)
        if max_val <= 63:
            self.colors = [(r * 4, g * 4, b * 4) for r, g, b in self.colors]

    def load_from_resource(self, data: bytes) -> None:
        """Load palette from a PALETTE resource.

        Format: 256 entries of 3 bytes each (R, G, B).
        """
        if len(data) < 256 * 3:
            return

        for i in range(256):
            base = i * 3
            r = data[base]
            g = data[base + 1]
            b = data[base + 2]
            self.colors[i] = (r, g, b)

        # Scale if VGA DAC format
        max_val = max(max(c) for c in self.colors)
        if max_val <= 63:
            self.colors = [(r * 4, g * 4, b * 4) for r, g, b in self.colors]

    def get_rgb(self, index: int) -> tuple[int, int, int]:
        """Get RGB values for a palette index."""
        return self.colors[index % 256]

    def to_pil_palette(self) -> list[int]:
        """Convert to PIL palette format (flat list of R,G,B values)."""
        result = []
        for r, g, b in self.colors:
            result.extend([r, g, b])
        return result
