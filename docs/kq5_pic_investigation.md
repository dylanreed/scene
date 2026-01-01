# KQ5 PIC Rendering Investigation

## Status: In Progress - Still Producing Glitchy Output

### Goal
Extract pixel art backgrounds from VGA-era Sierra SCI games (specifically KQ5) for ML training.

---

## What Was Discovered

### Resource Header Format (8 bytes)
| Bytes | Description |
|-------|-------------|
| 0-1 | Resource number |
| 2-3 | Packed size |
| 4-5 | Unpacked size |
| 6-7 | Compression method (4 = LZW1_PIC with MSB-first) |

### Decompressed Data Structure (after LZW)
| Offset | Size | Description |
|--------|------|-------------|
| 0-5 | 6 | Header: view_size=58136, view_start=1308, cdata_size=49523 |
| 6-12 | 7 | viewdata: width=320, height=190, clear_color=255 |
| 13-268 | 256 | Translation map (NOT the palette!) |
| 269-272 | 4 | Stamp |
| **266-1289** | **1024** | **Palette** (256 * 4 bytes RGBF, flag=1 = used) |
| 1290-1311 | 22 | Gap (view_start - PAL_SIZE - 2) |
| 1312-50834 | 49523 | Pixel data (cdata) |
| 50835+ | 10847 | RLE control stream |

### Key Fix Applied
Palette was being read from offset 13 (translation map area) instead of offset 266 (actual palette with flag=1 entries indicating used colors).

---

## What Was Tried

### Approaches That Did Not Fix the Glitchy Output:
1. **LZW Decompression** - MSB-first for compression method 4
2. **RLE Decoding Variants**:
   - Split streams (separate cdata + RLE controls)
   - Row-by-row decoding
   - Continuous stream decoding
   - 12 different RLE interpretations (variants A through L)
3. **Bitmap Offsets** - Tested range 0-2000
4. **Image Widths** - Tested 316-340 pixels
5. **Palette Scaling** - Tested 1x, 2x, 3x, 4x multipliers
6. **Palette Offset Correction** - Changed from offset 13 to offset 266

### RLE Format Understanding (from ScummVM)
```
Control byte & 0xC0:
- 0x00-0x3F: Literal copy, length = control byte
- 0x40-0x7F: Literal copy, length = control byte
- 0x80-0xBF: RLE fill, length = control & 0x3F, next byte = color
- 0xC0-0xFF: Skip, length = control & 0x3F, fill with clear_color
```

### ScummVM's decodeRLE Algorithm
```cpp
while (pos < size) {
    nextbyte = *rd++;           // Read from RLE stream
    *ob++ = nextbyte;           // Copy control byte to output
    pos++;

    switch (nextbyte & 0xC0) {
    case 0x40:
    case 0x00:                  // Literal: copy N bytes from pixel stream
        memcpy(ob, pd, nextbyte);
        pd += nextbyte;
        ob += nextbyte;
        pos += nextbyte;
        break;
    case 0x80:                  // RLE: copy 1 byte from pixel stream
        *ob++ = *pd++;
        pos++;
        break;
    case 0xC0:                  // Skip: no pixel data
    default:
        break;
    }
}
```

---

## Files Modified

### `/Users/nervous/Library/CloudStorage/Dropbox/Github/scene/src/sci/lzw.py`
- Updated `reorder_pic()` function to use palette at offset 266
- Added proper PAL_SIZE area handling (translation map + stamp + palette)

### `/Users/nervous/Library/CloudStorage/Dropbox/Github/scene/src/sci/pic_renderer.py`
- Updated to read viewdata from offset 1037 for dimensions
- Added `img_width`, `img_height`, `clear_color` instance variables

---

## Remaining Unknowns

1. **RLE Interleaving** - The exact algorithm may still be incorrect
2. **Row Processing** - ScummVM uses `_maxWidth` suggesting row-by-row processing
3. **Data Positions** - cdata/RLE split positions may need recalculation
4. **Cel Format** - May need to examine ScummVM's cel/view handling more closely

---

## Test Images Generated
Location: `/Users/nervous/Library/CloudStorage/Dropbox/Github/scene/extracts/`

Key test files:
- `kq5_pic1_correct_palette.png` - Using palette from offset 266
- `kq5_pic1_codebase_fixed.png` - Via updated codebase
- Various experimental variants (offset, width, RLE variant tests)

---

## Reference Resources

- **ScummVM Source**: `engines/sci/resource/decompressor.cpp` - `reorderPic()`, `decodeRLE()`
- **ScummVM Source**: `engines/sci/graphics/celobj32.cpp` - RLE rendering
- **ScummVM Wiki**: https://wiki.scummvm.org/index.php/SCI/Specifications

---

## Next Steps When Resuming

1. Compare byte-by-byte output with a known working extractor (if available)
2. Examine ScummVM's actual PIC rendering path more closely
3. Check if KQ5 uses a variant format different from standard SCI1
4. Consider if the LZW decompression itself might have subtle bugs
5. Try extracting with ScummVM's built-in debugger to get reference output
