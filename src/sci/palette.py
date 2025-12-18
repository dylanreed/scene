"""SCI1 palette handling."""
from typing import Tuple, List


class Palette:
    """SCI1 palette (256 RGB colors).

    SCI1 palette format:
    - 4 bytes header (ignored)
    - 256 bytes mapping table (ignored for now)
    - 256 * 4 bytes: (flag, R, G, B) for each color
      - flag: 0 = unused, 1 = used
      - R, G, B: 0-255 color values
    """

    def __init__(self, data: bytes):
        """Parse palette from decompressed data."""
        self.colors: List[Tuple[int, int, int]] = []

        # Color entries start at offset 260 (4 header + 256 mapping)
        COLOR_START = 260
        ENTRY_SIZE = 4

        for i in range(256):
            offset = COLOR_START + i * ENTRY_SIZE
            if offset + ENTRY_SIZE > len(data):
                # Fill remaining with black
                self.colors.append((0, 0, 0))
                continue

            flag = data[offset]
            r = data[offset + 1]
            g = data[offset + 2]
            b = data[offset + 3]

            # Color values are already 0-255, no scaling needed
            self.colors.append((r, g, b))

    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        """Get color by index."""
        return self.colors[index]

    def __len__(self) -> int:
        """Return number of colors."""
        return len(self.colors)


def create_default_palette() -> Palette:
    """Create a default VGA palette for SCI1 games.

    This creates a comprehensive 256-color palette with:
    - Colors 0-15: Standard EGA colors
    - Colors 16-31: Grayscale ramp
    - Colors 32-255: Color cube with flesh tones and common game colors
    """
    # Create fake palette data with proper structure
    data = bytearray(260 + 256 * 4)

    # Standard EGA colors (0-15)
    ega_colors = [
        (0, 0, 0),        # 0: Black
        (0, 0, 170),      # 1: Blue
        (0, 170, 0),      # 2: Green
        (0, 170, 170),    # 3: Cyan
        (170, 0, 0),      # 4: Red
        (170, 0, 170),    # 5: Magenta
        (170, 85, 0),     # 6: Brown
        (170, 170, 170),  # 7: Light gray
        (85, 85, 85),     # 8: Dark gray
        (85, 85, 255),    # 9: Light blue
        (85, 255, 85),    # 10: Light green
        (85, 255, 255),   # 11: Light cyan
        (255, 85, 85),    # 12: Light red
        (255, 85, 255),   # 13: Light magenta
        (255, 255, 85),   # 14: Yellow
        (255, 255, 255),  # 15: White
    ]

    # Grayscale ramp (16-31)
    grayscale = [(i * 17, i * 17, i * 17) for i in range(16)]

    # Color cube approximation (32-255)
    # Create ranges of common colors: reds, greens, blues, browns, skin tones
    colors = list(ega_colors) + grayscale

    # Flesh tones / skin colors (common in Sierra games)
    for i in range(8):
        r = 180 + i * 9
        g = 120 + i * 12
        b = 80 + i * 10
        colors.append((min(255, r), min(255, g), min(255, b)))

    # Browns (wood, earth)
    for i in range(8):
        r = 80 + i * 15
        g = 40 + i * 10
        b = 20 + i * 5
        colors.append((min(255, r), g, b))

    # Blues (sky, water)
    for i in range(16):
        r = i * 4
        g = 40 + i * 8
        b = 100 + i * 9
        colors.append((r, min(255, g), min(255, b)))

    # Greens (foliage)
    for i in range(16):
        r = 20 + i * 4
        g = 60 + i * 10
        b = 20 + i * 4
        colors.append((r, min(255, g), b))

    # Reds/pinks
    for i in range(16):
        r = 150 + i * 6
        g = 50 + i * 8
        b = 50 + i * 8
        colors.append((min(255, r), g, b))

    # Fill remaining with color cube
    while len(colors) < 256:
        idx = len(colors) - 96
        r = (idx % 6) * 51
        g = ((idx // 6) % 6) * 51
        b = ((idx // 36) % 6) * 51
        colors.append((r, g, b))

    # Ensure transparent color (usually around 50) is distinct
    if len(colors) > 50:
        colors[50] = (255, 0, 255)  # Magenta for transparency

    for i in range(256):
        offset = 260 + i * 4
        data[offset] = 1  # flag = used

        if i < len(colors):
            r, g, b = colors[i]
        else:
            r, g, b = 0, 0, 0

        data[offset + 1] = r
        data[offset + 2] = g
        data[offset + 3] = b

    return Palette(bytes(data))


def load_palette_from_game(game_path: str) -> Palette:
    """Load palette from a SCI1 game directory.

    SCI1 palettes use a complex remap format with color ramps.
    For now, we use a default VGA palette.
    TODO: Implement full SCI1 palette ramp parsing.
    """
    # SCI1 palette format is complex (color ramps, not simple RGB entries)
    # Use default palette for now - colors will be approximate but recognizable
    return create_default_palette()
