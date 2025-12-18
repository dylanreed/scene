"""Parse SCI0/early SCI1 RESOURCE.MAP files.

This handles the older resource map format used by games like:
- King's Quest 5 (1990)
- King's Quest 4
- Space Quest 3
- Other early SCI games

Format differences from SCI1:
- No type offset header table
- Flat list of 6-byte entries
- Resource ID encodes type (5 bits) and number (11 bits)
- Type codes: 0=VIEW, 1=PIC, 2=SCRIPT, 3=TEXT, 4=SOUND, etc.
"""
import os
import struct
from typing import Dict, List


# SCI0 resource type codes (different from SCI1)
SCI0_TYPES = {
    0: 'VIEW',
    1: 'PIC',
    2: 'SCRIPT',
    3: 'TEXT',
    4: 'SOUND',
    5: 'MEMORY',
    6: 'VOCAB',
    7: 'FONT',
    8: 'CURSOR',
    9: 'PATCH',
    10: 'BITMAP',
    11: 'PALETTE',
    12: 'CDAUDIO',
    13: 'AUDIO',
    14: 'SYNC',
    15: 'MESSAGE',
    16: 'MAP',
    17: 'HEAP',
}

# Map SCI0 types to SCI1 types for compatibility
SCI0_TO_SCI1_TYPE = {
    0: 0x80,  # VIEW
    1: 0x81,  # PIC
    2: 0x82,  # SCRIPT
    3: 0x83,  # TEXT
    4: 0x84,  # SOUND
    6: 0x86,  # VOCAB
    7: 0x87,  # FONT
    8: 0x88,  # CURSOR
    9: 0x89,  # PATCH
}


class ResourceMapSCI0:
    """Parser for SCI0/early SCI1 RESOURCE.MAP files."""

    def __init__(self, game_path: str):
        """Load and parse RESOURCE.MAP from the game directory."""
        self.game_path = game_path
        self.map_path = os.path.join(game_path, "RESOURCE.MAP")
        self._resources: Dict[int, List[dict]] = {}
        self._parse()

    def _parse(self):
        """Parse the RESOURCE.MAP file.

        Format: Flat list of 6-byte entries
        - Bytes 0-1: Resource ID (type << 11 | resource_number)
        - Bytes 2-5: Location (file_number << 26 | offset)

        End marker: 0xFFFF (type=31, number=2047)
        """
        with open(self.map_path, 'rb') as f:
            data = f.read()

        num_entries = len(data) // 6

        for i in range(num_entries):
            offset = i * 6
            if offset + 6 > len(data):
                break

            entry = data[offset:offset+6]
            id_word = struct.unpack_from('<H', entry, 0)[0]
            location = struct.unpack_from('<I', entry, 2)[0]

            # Check for end marker
            if id_word == 0xFFFF:
                break

            # Decode resource ID
            res_type = (id_word >> 11) & 0x1F
            res_num = id_word & 0x7FF

            # Decode location
            file_number = (location >> 26) & 0x3F
            file_offset = location & 0x03FFFFFF

            # Convert to SCI1 type code for compatibility
            sci1_type = SCI0_TO_SCI1_TYPE.get(res_type, res_type)

            if sci1_type not in self._resources:
                self._resources[sci1_type] = []

            self._resources[sci1_type].append({
                'resource_number': res_num,
                'file_number': file_number,
                'offset': file_offset,
                'sci0_type': res_type,
            })

    def get_resource_types(self) -> List[int]:
        """Return list of resource types found in the map (SCI1 codes)."""
        return list(self._resources.keys())

    def get_resources(self, resource_type: int) -> List[dict]:
        """Return list of resources for the given type (SCI1 code)."""
        return self._resources.get(resource_type, [])

    def get_resources_by_sci0_type(self, sci0_type: int) -> List[dict]:
        """Return list of resources for the given SCI0 type code."""
        sci1_type = SCI0_TO_SCI1_TYPE.get(sci0_type, sci0_type)
        return self._resources.get(sci1_type, [])
