"""Parse SCI0/SCI1 RESOURCE.MAP files."""

import struct
from dataclasses import dataclass
from pathlib import Path


# Resource type codes (0-based for SCI0, add 0x80 for SCI1)
RESOURCE_TYPES = {
    0: "VIEW",
    1: "PIC",
    2: "SCRIPT",
    3: "TEXT",
    4: "SOUND",
    5: "MEMORY",
    6: "VOCAB",
    7: "FONT",
    8: "CURSOR",
    9: "PATCH",
    10: "BITMAP",
    11: "PALETTE",
    12: "CDAUDIO",
    13: "AUDIO",
    14: "SYNC",
    15: "MESSAGE",
    16: "MAP",
    17: "HEAP",
}

TYPE_VIEW = 0
TYPE_PIC = 1
TYPE_SCRIPT = 2
TYPE_PALETTE = 11


@dataclass
class ResourceEntry:
    """A single resource entry from the map."""
    resource_type: int
    resource_number: int
    file_number: int
    offset: int

    @property
    def type_name(self) -> str:
        return RESOURCE_TYPES.get(self.resource_type, f"UNKNOWN_{self.resource_type}")


class ResourceMap:
    """Parser for SCI0/SCI1 RESOURCE.MAP files."""

    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.entries: list[ResourceEntry] = []
        self.sci_version = None  # 0 for SCI0, 1 for SCI1
        self._parse()

    def _detect_version(self, data: bytes) -> int:
        """Detect whether this is SCI0 or SCI1 format.

        SCI1 maps start with a type header (type bytes 0x80-0x91).
        SCI0 maps start directly with resource entries.
        """
        if len(data) < 3:
            return 0

        first_byte = data[0]
        # SCI1 type headers use 0x80-0x91 for resource types
        if 0x80 <= first_byte <= 0x91:
            return 1
        return 0

    def _parse(self):
        """Parse the RESOURCE.MAP file."""
        with open(self.map_path, "rb") as f:
            data = f.read()

        self.sci_version = self._detect_version(data)

        if self.sci_version == 1:
            self._parse_sci1(data)
        else:
            self._parse_sci0(data)

    def _parse_sci0(self, data: bytes):
        """Parse SCI0/SCI01 format resource map.

        Format: flat list of 6-byte entries, terminated by 6 x 0xFF.
        - Bytes 0-1 (LE 16-bit): type (high 5 bits) + number (low 11 bits)
        - Bytes 2-5 (LE 32-bit): file number (high 6 bits) + offset (low 26 bits)
        """
        pos = 0
        while pos + 6 <= len(data):
            # Check for end marker (6 x 0xFF)
            if data[pos:pos+6] == b'\xff\xff\xff\xff\xff\xff':
                break

            type_and_num = struct.unpack_from("<H", data, pos)[0]
            packed_offset = struct.unpack_from("<I", data, pos + 2)[0]

            # Extract type (high 5 bits) and number (low 11 bits)
            resource_type = (type_and_num >> 11) & 0x1F
            resource_number = type_and_num & 0x7FF

            # SCI0 uses 6 bits for file number, 26 bits for offset
            file_number = (packed_offset >> 26) & 0x3F
            file_offset = packed_offset & 0x03FFFFFF

            self.entries.append(ResourceEntry(
                resource_type=resource_type,
                resource_number=resource_number,
                file_number=file_number,
                offset=file_offset,
            ))
            pos += 6

    def _parse_sci1(self, data: bytes):
        """Parse SCI1 format resource map.

        Format:
        - Type headers: 3 bytes each (type + 2-byte offset), terminated by 0xFF
        - Lookup tables: 6 bytes each (2-byte number + 4-byte packed offset)
        """
        # Read type header table
        type_offsets = {}
        pos = 0

        while pos < len(data):
            resource_type = data[pos]
            if resource_type == 0xFF:
                break

            offset = struct.unpack_from("<H", data, pos + 1)[0]
            type_offsets[resource_type] = offset
            pos += 3

        # Read each type's lookup table
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])

        for i, (resource_type, start_offset) in enumerate(sorted_types):
            if i + 1 < len(sorted_types):
                end_offset = sorted_types[i + 1][1]
            else:
                end_offset = len(data)

            pos = start_offset
            while pos + 6 <= end_offset and pos + 6 <= len(data):
                resource_number = struct.unpack_from("<H", data, pos)[0]

                if resource_number == 0xFFFF:
                    break

                packed = struct.unpack_from("<I", data, pos + 2)[0]

                file_number = (packed >> 28) & 0x0F
                file_offset = packed & 0x0FFFFFFF

                # Convert SCI1 type (0x80+) to 0-based
                norm_type = resource_type - 0x80 if resource_type >= 0x80 else resource_type

                self.entries.append(ResourceEntry(
                    resource_type=norm_type,
                    resource_number=resource_number,
                    file_number=file_number,
                    offset=file_offset,
                ))
                pos += 6

    def get_resources_by_type(self, resource_type: int) -> list[ResourceEntry]:
        """Get all resources of a specific type."""
        return [e for e in self.entries if e.resource_type == resource_type]

    def get_pics(self) -> list[ResourceEntry]:
        """Get all PIC resources."""
        return self.get_resources_by_type(TYPE_PIC)

    def get_palettes(self) -> list[ResourceEntry]:
        """Get all PALETTE resources."""
        return self.get_resources_by_type(TYPE_PALETTE)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resource_map.py <path/to/RESOURCE.MAP>")
        sys.exit(1)

    map_path = Path(sys.argv[1])
    rmap = ResourceMap(map_path)

    print(f"SCI version: {rmap.sci_version}")
    print(f"Found {len(rmap.entries)} resources:")

    # Count by type
    type_counts = {}
    for entry in rmap.entries:
        type_counts[entry.type_name] = type_counts.get(entry.type_name, 0) + 1

    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count}")

    pics = rmap.get_pics()
    print(f"\nPIC resources ({len(pics)}):")
    for pic in pics[:10]:
        print(f"  PIC #{pic.resource_number}: file {pic.file_number}, offset {pic.offset}")
    if len(pics) > 10:
        print(f"  ... and {len(pics) - 10} more")
