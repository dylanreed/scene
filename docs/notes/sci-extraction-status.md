# SCI Image Extraction - Status & Options

## Current Status

The SCI resource extraction code in `src/sci/` can successfully:
- Parse RESOURCE.MAP files (SCI0 and SCI1 formats)
- Read and decompress resources using LZW (compression methods 1, 3, 4)
- Load VGA palettes from PIC resources

**Blocking Issue:** SCI1 VGA PIC rendering produces garbled output. The LZW-decompressed data requires a complex `reorderPic()` transformation that combines separate RLE control and pixel data streams. Multiple interpretation attempts have not succeeded.

## Files Implemented

- `src/sci/resource_map.py` - RESOURCE.MAP parser
- `src/sci/resource_reader.py` - Resource file reader with LZW decompression
- `src/sci/lzw.py` - LZW decompression (MSB/LSB variants)
- `src/sci/palette.py` - VGA palette handling
- `src/sci/pic_renderer.py` - PIC renderer (incomplete/not working)

## Alternative Approaches

### Option 1: ScummVM Screenshots
ScummVM can run KQ5 and other SCI games. Use the screenshot feature to capture backgrounds:
- Run game in ScummVM
- Use screenshot hotkey (usually F5 menu or configurable)
- Batch process by playing through rooms

### Option 2: Existing Extraction Tools
- **SCI Viewer** - GUI tool for viewing SCI resources
- **SCI Resource Dumper** by Vladimir Gneushev - Extracts from SCI1.1 and SCI32 games
- **SCICompanion** - Full SCI game development tool with resource viewing

### Option 3: ScummVM Debugger
ScummVM has a built-in debugger that can dump resources:
- Launch with `--debugflags=...`
- Use console commands to extract resources

### Option 4: Continue Custom Development
The `reorderPic()` function in ScummVM's `engines/sci/resource/decompressor.cpp` contains the exact algorithm needed. Key insights:
- Decompressed data has header: view_size (2), view_start (2), cdata_size (2), viewdata (7)
- Palette is 1024 bytes at offset 13
- cdata (pixel colors) is at end of data
- RLE control bytes are between palette and cdata
- `decodeRLE()` combines the streams

## References

- [ScummVM SCI Specifications](https://wiki.scummvm.org/index.php/SCI/Specifications)
- [ScummVM Source - decompressor.cpp](https://github.com/scummvm/scummvm/blob/master/engines/sci/resource/decompressor.cpp)
- [VOGONS SCI1 Picture Discussion](https://www.vogons.org/viewtopic.php?t=41256)

## Test Files

Debug/test images saved to `extracts/` during development attempts.
