# SCI1 Image Extractor - Design Document

## Overview

A Python script to extract background images (PIC resources) from Sierra SCI1 games and save them as PNG files for use as training data for the pixel art generator.

## Target Games

- King's Quest V (KQ5) - SCI1
- EcoQuest 1 (EQ1) - SCI1
- EcoQuest 2 (EQ2) - SCI1

## Output

- Location: `extracts/<GAME>-<NNNN>.png`
- Format: PNG, 320×200, RGB
- Naming: `KQ5-0001.png`, `EQ1-0001.png`, etc.

## Architecture

```
src/
  extract_sci.py          # Main entry point / CLI
  sci/
    resource_map.py       # Parse RESOURCE.MAP
    resource_reader.py    # Read and decompress from RESOURCE.xxx
    lzw.py               # LZW decompression (SCI1 MSB variant)
    pic_renderer.py      # Execute PIC opcodes → image
    palette.py           # VGA palette handling
```

## SCI1 Resource Format

### RESOURCE.MAP Structure

1. **Type headers** (3 bytes each):
   - Byte 0: Resource type (0x80-0x91)
   - Bytes 1-2: Little-endian offset to lookup table
   - Terminated by type 0xFF

2. **Lookup tables** (6 bytes per entry):
   - Bytes 0-1: Little-endian resource number
   - Bytes 2-5: Little-endian packed value
     - High 4 bits: RESOURCE.xxx file number
     - Low 28 bits: Byte offset within that file

### Resource Types

| Code | Type |
|------|------|
| 0x80 | VIEW (sprites) |
| 0x81 | PIC (backgrounds) |
| 0x82 | SCRIPT |
| 0x83 | TEXT |
| 0x84 | SOUND |
| 0x85 | MEMORY |
| 0x86 | VOCAB |
| 0x87 | FONT |
| 0x88 | CURSOR |
| 0x89 | PATCH |
| 0x8A | BITMAP |
| 0x8B | PALETTE |
| 0x8C | CDAUDIO |
| 0x8D | AUDIO |
| 0x8E | SYNC |
| 0x8F | MESSAGE |
| 0x90 | MAP |
| 0x91 | HEAP |

### Resource File Entry Format

Each resource in RESOURCE.xxx:
- Bytes 0-1: Resource type + number (varies by SCI version)
- Bytes 2-3: Compressed size (LE)
- Bytes 4-5: Decompressed size (LE)
- Bytes 6-7: Compression method
- Remaining: Compressed data

## LZW Decompression (SCI1)

- Variable bit width: 9-12 bits
- MSB-first bit reading (unlike SCI0 which is LSB)
- Code 256 = reset dictionary
- Code 257 = end of stream
- Dictionary starts with 256 single-byte entries (0-255)

### PIC Reordering

After LZW decompression, PIC data is interleaved and must be reordered:
- Data is stored in 4 planes
- Reorder to sequential bytes for processing

## PIC Drawing Commands

| Opcode | Name | Description |
|--------|------|-------------|
| 0xF0 | SET_COLOR | Set visual drawing color |
| 0xF1 | DISABLE_VISUAL | Stop drawing to visual layer |
| 0xF2 | SET_PRIORITY | Set priority color |
| 0xF3 | DISABLE_PRIORITY | Stop drawing to priority layer |
| 0xF4 | SHORT_LINES | Relative lines, 1-byte coords |
| 0xF5 | MEDIUM_LINES | Relative lines, mixed coords |
| 0xF6 | LONG_LINES | Absolute lines, 2-byte coords |
| 0xF7 | SHORT_RELATIVE | Short relative from last point |
| 0xF8 | FILL | Flood fill at coordinates |
| 0xF9 | SET_PATTERN | Set brush pattern/size |
| 0xFA | SHORT_PATTERN | Pattern at short coords |
| 0xFB | MEDIUM_PATTERN | Pattern at medium coords |
| 0xFC | LONG_PATTERN | Pattern at absolute coords |
| 0xFE | EXTENDED | Palette, embedded cels, etc. |
| 0xFF | END | End of PIC data |

### Coordinate Encoding

- **Short**: Single byte, sign-extended for relative
- **Medium**: Mix of absolute and relative
- **Long**: Two bytes per coordinate (x, y)

### Extended Opcodes (0xFE)

| Sub-opcode | Description |
|------------|-------------|
| 0x00 | Set palette entry |
| 0x01 | Set full palette |
| 0x02 | Set priority table |

## Palette Handling

- Resource type 0x8B (PALETTE)
- 256 colors × 3 bytes (R, G, B)
- VGA DAC values: 0-63 → multiply by 4 for 0-255
- Some games have palette embedded in PIC via 0xFE opcodes

## Rendering Process

1. Initialize 320×200 canvas to color 0
2. Load global palette from PALETTE resource (if exists)
3. Read PIC data byte by byte
4. Execute drawing commands updating canvas
5. On 0xFF, convert indexed canvas to RGB using palette
6. Save as PNG

## Flood Fill Algorithm

Stack-based fill:
1. Push starting pixel
2. Pop pixel, check if matches target color
3. If match: set to fill color, push neighbors (up/down/left/right)
4. Repeat until stack empty

## CLI Interface

```bash
python src/extract_sci.py /path/to/game --output extracts/ --prefix KQ5
```

Arguments:
- `game_path`: Directory containing RESOURCE.MAP and RESOURCE.xxx
- `--output`: Output directory (default: extracts/)
- `--prefix`: Filename prefix (default: derived from path)

## Error Handling

- Skip malformed resources, log warning
- Continue extraction on individual failures
- Report summary at end (extracted/failed/total)

## Dependencies

- Python 3.10+
- Pillow (PIL) for PNG output
- No other external dependencies

## References

- [ScummVM SCI Implementation](https://github.com/scummvm/scummvm/tree/master/engines/sci)
- [ScummVM Wiki - SCI Specifications](https://wiki.scummvm.org/index.php/SCI/Specifications)
