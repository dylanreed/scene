"""Read resources from SCI0/early SCI1 RESOURCE.xxx files.

SCI0 resource header format (8 bytes):
- Bytes 0-1: Resource ID (type << 11 | number)
- Bytes 2-3: Compressed size - 4 (size of data after header)
- Bytes 4-5: Decompressed size
- Bytes 6-7: Compression method
"""
import os
import struct
from typing import Tuple


class ResourceReaderSCI0:
    """Reader for SCI0/early SCI1 resource files."""

    HEADER_SIZE = 8

    def __init__(self, game_path: str):
        """Initialize with path to game directory."""
        self.game_path = game_path

    def _get_resource_file_path(self, file_number: int) -> str:
        """Get path to RESOURCE.xxx file."""
        return os.path.join(self.game_path, f"RESOURCE.{file_number:03d}")

    def read_header(self, file_number: int, offset: int) -> dict:
        """Read and parse resource header at given location.

        SCI0 Header format (8 bytes):
        - Bytes 0-1: Resource ID (type << 11 | number)
        - Bytes 2-3: Compressed size (actual data size after this header)
        - Bytes 4-5: Decompressed size
        - Bytes 6-7: Compression method
        """
        path = self._get_resource_file_path(file_number)
        with open(path, 'rb') as f:
            f.seek(offset)
            header_data = f.read(self.HEADER_SIZE)

        if len(header_data) < self.HEADER_SIZE:
            raise ValueError(f"Could not read full header at offset {offset}")

        resource_id = struct.unpack_from('<H', header_data, 0)[0]
        compressed_size = struct.unpack_from('<H', header_data, 2)[0]
        decompressed_size = struct.unpack_from('<H', header_data, 4)[0]
        compression_method = struct.unpack_from('<H', header_data, 6)[0]

        # Decode resource ID
        resource_type = (resource_id >> 11) & 0x1F
        resource_number = resource_id & 0x7FF

        return {
            'type': resource_type,
            'resource_number': resource_number,
            'compressed_size': compressed_size,
            'decompressed_size': decompressed_size,
            'compression_method': compression_method,
        }

    def read_resource(self, file_number: int, offset: int) -> Tuple[bytes, dict]:
        """Read resource data and header from given location.

        Returns tuple of (compressed_data, header_dict).
        """
        header = self.read_header(file_number, offset)

        path = self._get_resource_file_path(file_number)
        with open(path, 'rb') as f:
            f.seek(offset + self.HEADER_SIZE)
            data = f.read(header['compressed_size'])

        return data, header
