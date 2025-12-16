"""LZW decompression for SCI resources."""

from typing import Optional


class BitReader:
    """Read variable-width bits from a byte stream."""

    def __init__(self, data: bytes, msb_first: bool = False):
        self.data = data
        self.msb_first = msb_first
        self.pos = 0  # byte position
        self.bit_pos = 0  # bit position within current byte
        # For MSB reading, we accumulate bits differently
        self._bits_buffer = 0
        self._bits_in_buffer = 0

    def read_bits(self, num_bits: int) -> Optional[int]:
        """Read num_bits from the stream."""
        if self.msb_first:
            return self._read_bits_msb(num_bits)
        else:
            return self._read_bits_lsb(num_bits)

    def _read_bits_lsb(self, num_bits: int) -> Optional[int]:
        """Read bits LSB-first (SCI0 style)."""
        result = 0
        for i in range(num_bits):
            if self.pos >= len(self.data):
                return None
            byte = self.data[self.pos]
            bit = (byte >> self.bit_pos) & 1
            result |= (bit << i)
            self.bit_pos += 1
            if self.bit_pos >= 8:
                self.bit_pos = 0
                self.pos += 1
        return result

    def _read_bits_msb(self, num_bits: int) -> Optional[int]:
        """Read bits MSB-first (SCI01/1 style)."""
        # Fill buffer with enough bits
        while self._bits_in_buffer < num_bits:
            if self.pos >= len(self.data):
                return None
            self._bits_buffer = (self._bits_buffer << 8) | self.data[self.pos]
            self._bits_in_buffer += 8
            self.pos += 1

        # Extract the requested bits from the top
        self._bits_in_buffer -= num_bits
        result = (self._bits_buffer >> self._bits_in_buffer) & ((1 << num_bits) - 1)
        return result


# Special LZW codes
RESET_CODE = 256
END_CODE = 257
FIRST_CODE = 258  # First code for dictionary entries


def decompress_lzw(data: bytes, decompressed_size: int, msb_first: bool = False) -> bytes:
    """Decompress LZW-compressed data.

    Args:
        data: Compressed data
        decompressed_size: Expected size of decompressed data
        msb_first: True for SCI01/1 (MSB-first), False for SCI0 (LSB-first)

    Returns:
        Decompressed data
    """
    reader = BitReader(data, msb_first)
    output = bytearray()

    # Dictionary: codes 0-255 are single bytes, 256=reset, 257=end, 258+ are sequences
    # We store only the sequences starting from code 258
    dictionary: list[bytes] = []

    # Start with 9-bit codes
    bits = 9
    # SCI01/1 increases bit width one code earlier
    if msb_first:
        code_limit = (1 << bits) - 1  # 511 for 9 bits
    else:
        code_limit = 1 << bits  # 512 for 9 bits

    prev_string = b""
    table_size = FIRST_CODE  # Next code to assign

    while len(output) < decompressed_size:
        code = reader.read_bits(bits)
        if code is None:
            break

        if code == RESET_CODE:
            # Reset dictionary
            dictionary = []
            bits = 9
            if msb_first:
                code_limit = (1 << bits) - 1
            else:
                code_limit = 1 << bits
            table_size = FIRST_CODE
            prev_string = b""

            # After reset, read next code
            code = reader.read_bits(bits)
            if code is None:
                break

        if code == END_CODE:
            break

        # Get string for this code
        if code < 256:
            # Literal byte
            current_string = bytes([code])
        elif code >= FIRST_CODE:
            dict_index = code - FIRST_CODE
            if dict_index < len(dictionary):
                current_string = dictionary[dict_index]
            elif dict_index == len(dictionary) and prev_string:
                # Special case: code not in dictionary yet (KwKwK case)
                current_string = prev_string + prev_string[0:1]
            else:
                # Invalid code - try to recover
                break
        else:
            # Code 256 or 257 should have been handled above
            break

        output.extend(current_string)

        # Add new dictionary entry
        if prev_string and table_size < 4096:
            new_entry = prev_string + current_string[0:1]
            dictionary.append(new_entry)
            table_size += 1

            # Increase bit width if needed
            if table_size >= code_limit and bits < 12:
                bits += 1
                if msb_first:
                    code_limit = (1 << bits) - 1
                else:
                    code_limit = 1 << bits

        prev_string = current_string

    return bytes(output[:decompressed_size])


def reorder_pic(data: bytes) -> bytes:
    """Reorder PIC data after LZW decompression.

    Note: SCI1 VGA PICs don't need this simple reordering - they have
    a more complex structure with header, palette, and RLE bitmap.
    This function is kept for compatibility but returns data unchanged.
    """
    # For SCI1 VGA games like KQ5, the data is already in the correct order
    # The "reordering" happens during the PIC parsing stage instead
    return data


if __name__ == "__main__":
    # Simple test
    test_data = b"AAAAAABBBBCCCCCC"
    print(f"Test data: {test_data}")
    print(f"Length: {len(test_data)}")
