"""Read and decompress SCI resources from RESOURCE.xxx files."""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .lzw import decompress_lzw, reorder_pic
from .resource_map import ResourceEntry, ResourceMap, TYPE_PIC


# Compression method codes
COMP_NONE = 0
COMP_LZW = 1      # SCI0 LZW
COMP_HUFFMAN = 2  # Huffman
COMP_LZW1 = 3     # SCI1 LZW (view format)
COMP_LZW1_PIC = 4 # SCI1 LZW (pic format, needs reordering)


@dataclass
class ResourceHeader:
    """Header for a resource in a RESOURCE.xxx file."""
    resource_type: int
    resource_number: int
    packed_size: int
    unpacked_size: int
    compression: int


class ResourceReader:
    """Read resources from SCI game files."""

    def __init__(self, game_path: Path):
        self.game_path = Path(game_path)
        self.resource_map = ResourceMap(self.game_path / "RESOURCE.MAP")

    def _get_resource_file(self, file_number: int) -> Path:
        """Get path to a RESOURCE.xxx file."""
        return self.game_path / f"RESOURCE.{file_number:03d}"

    def _read_header_sci0(self, f) -> ResourceHeader:
        """Read SCI0 format resource header (8 bytes)."""
        data = f.read(8)
        if len(data) < 8:
            raise ValueError("Incomplete header")

        res_id, packed_plus_4, unpacked, compression = struct.unpack("<HHHH", data)

        resource_type = (res_id >> 11) & 0x1F
        resource_number = res_id & 0x7FF
        packed_size = packed_plus_4 - 4  # Subtract the 4 bytes for partial header

        return ResourceHeader(
            resource_type=resource_type,
            resource_number=resource_number,
            packed_size=packed_size,
            unpacked_size=unpacked,
            compression=compression,
        )

    def _read_header_sci1(self, f) -> ResourceHeader:
        """Read SCI1 format resource header (9 bytes)."""
        data = f.read(9)
        if len(data) < 9:
            raise ValueError("Incomplete header")

        resource_type = data[0]
        resource_number, packed_plus_4, unpacked, compression = struct.unpack(
            "<HHHH", data[1:9]
        )

        # Normalize type (SCI1 uses 0x80+ for types)
        if resource_type >= 0x80:
            resource_type -= 0x80

        packed_size = packed_plus_4 - 4

        return ResourceHeader(
            resource_type=resource_type,
            resource_number=resource_number,
            packed_size=packed_size,
            unpacked_size=unpacked,
            compression=compression,
        )

    def read_resource(self, entry: ResourceEntry) -> Optional[bytes]:
        """Read and decompress a resource.

        Args:
            entry: ResourceEntry from the resource map

        Returns:
            Decompressed resource data, or None if failed
        """
        resource_file = self._get_resource_file(entry.file_number)
        if not resource_file.exists():
            return None

        with open(resource_file, "rb") as f:
            f.seek(entry.offset)

            # Try SCI0 header first (8 bytes) - works for SCI0 and SCI01 games
            # SCI01 games like KQ5 use SCI1 map format but SCI0 resource headers
            header = self._read_header_sci0(f)

            # Validate header - if it doesn't match expected resource, try SCI1
            if header.resource_type != entry.resource_type or header.compression > 10:
                f.seek(entry.offset)
                header = self._read_header_sci1(f)

            # Verify resource matches what we expect
            if header.resource_type != entry.resource_type:
                print(f"Warning: Type mismatch - expected {entry.resource_type}, "
                      f"got {header.resource_type}")

            # Read compressed data
            if header.packed_size <= 0:
                return None

            compressed_data = f.read(header.packed_size)
            if len(compressed_data) < header.packed_size:
                return None

            # Decompress based on method
            return self._decompress(
                compressed_data,
                header.unpacked_size,
                header.compression,
                entry.resource_type,
            )

    def _decompress(
        self,
        data: bytes,
        unpacked_size: int,
        compression: int,
        resource_type: int,
    ) -> bytes:
        """Decompress resource data."""

        if compression == COMP_NONE:
            return data

        # Determine bit order based on compression method
        # Methods 3 and 4 (LZW1) always use MSB-first regardless of SCI version
        # Method 1 (LZW) uses LSB-first for SCI0
        if compression in (COMP_LZW1, COMP_LZW1_PIC):
            msb_first = True
        elif compression == COMP_LZW:
            msb_first = self.resource_map.sci_version >= 1
        else:
            msb_first = False

        if compression in (COMP_LZW, COMP_LZW1, COMP_LZW1_PIC):
            decompressed = decompress_lzw(data, unpacked_size, msb_first)

            # PICs with method 4 need reordering
            if compression == COMP_LZW1_PIC and resource_type == TYPE_PIC:
                decompressed = reorder_pic(decompressed)

            return decompressed

        elif compression == COMP_HUFFMAN:
            # TODO: Implement Huffman decompression if needed
            raise NotImplementedError(f"Huffman decompression not implemented")

        else:
            raise ValueError(f"Unknown compression method: {compression}")

    def read_pic(self, pic_number: int) -> Optional[bytes]:
        """Read a specific PIC resource by number."""
        pics = self.resource_map.get_pics()
        for entry in pics:
            if entry.resource_number == pic_number:
                return self.read_resource(entry)
        return None

    def read_all_pics(self) -> list[tuple[int, bytes]]:
        """Read all PIC resources.

        Returns:
            List of (pic_number, data) tuples
        """
        results = []
        for entry in self.resource_map.get_pics():
            data = self.read_resource(entry)
            if data:
                results.append((entry.resource_number, data))
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python resource_reader.py <path/to/game>")
        sys.exit(1)

    game_path = Path(sys.argv[1])
    reader = ResourceReader(game_path)

    print(f"SCI version: {reader.resource_map.sci_version}")
    print(f"Total resources: {len(reader.resource_map.entries)}")

    pics = reader.resource_map.get_pics()
    print(f"PIC resources: {len(pics)}")

    if pics:
        # Try to read the first PIC
        entry = pics[0]
        print(f"\nReading PIC #{entry.resource_number}...")
        data = reader.read_resource(entry)
        if data:
            print(f"  Decompressed size: {len(data)} bytes")
            print(f"  First 20 bytes: {data[:20].hex()}")
        else:
            print("  Failed to read resource")
