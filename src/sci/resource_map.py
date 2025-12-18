"""Parse SCI1 RESOURCE.MAP files."""
import os
import struct
from typing import Dict, List


class ResourceMap:
    """Parser for SCI1 RESOURCE.MAP files."""

    def __init__(self, game_path: str):
        """Load and parse RESOURCE.MAP from the game directory."""
        self.game_path = game_path
        self.map_path = os.path.join(game_path, "RESOURCE.MAP")
        self._type_offsets: Dict[int, int] = {}
        self._resources: Dict[int, List[dict]] = {}
        self._parse()

    def _parse(self):
        """Parse the RESOURCE.MAP file."""
        with open(self.map_path, 'rb') as f:
            data = f.read()

        # Parse type headers (3 bytes each) until we hit 0xFF
        pos = 0
        while pos < len(data):
            resource_type = data[pos]
            if resource_type == 0xFF:
                break
            offset = struct.unpack_from('<H', data, pos + 1)[0]
            self._type_offsets[resource_type] = offset
            pos += 3

        # Now parse lookup tables for each type
        sorted_types = sorted(self._type_offsets.items(), key=lambda x: x[1])

        for i, (resource_type, start_offset) in enumerate(sorted_types):
            # Determine end offset (next type's start or end of file)
            if i + 1 < len(sorted_types):
                end_offset = sorted_types[i + 1][1]
            else:
                end_offset = len(data)

            self._resources[resource_type] = []
            pos = start_offset

            while pos + 6 <= end_offset:
                # 6-byte entry: 2 bytes resource number, 4 bytes packed location
                resource_number = struct.unpack_from('<H', data, pos)[0]
                packed = struct.unpack_from('<I', data, pos + 2)[0]

                # Check for end marker (0xFFFF resource number)
                if resource_number == 0xFFFF:
                    break

                # Extract file number (high 4 bits) and offset (low 28 bits)
                file_number = (packed >> 28) & 0x0F
                offset = packed & 0x0FFFFFFF

                self._resources[resource_type].append({
                    'resource_number': resource_number,
                    'file_number': file_number,
                    'offset': offset
                })

                pos += 6

    def get_resource_types(self) -> List[int]:
        """Return list of resource types found in the map."""
        return list(self._type_offsets.keys())

    def get_resources(self, resource_type: int) -> List[dict]:
        """Return list of resources for the given type."""
        return self._resources.get(resource_type, [])
