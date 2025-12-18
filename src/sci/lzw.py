"""SCI1 LZW decompression."""
from typing import List


class BitReaderMSB:
    """MSB-first bit reader matching ScummVM's implementation."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.dwBits = 0  # 32-bit buffer
        self.nBits = 0   # Valid bits in buffer

    def fetch_bits(self):
        """Load more bytes into the bit buffer."""
        while self.nBits <= 24 and self.pos < len(self.data):
            self.dwBits |= self.data[self.pos] << (24 - self.nBits)
            self.nBits += 8
            self.pos += 1

    def get_bits(self, n: int) -> int:
        """Read n bits from buffer, MSB-first."""
        if self.nBits < n:
            self.fetch_bits()
        if self.nBits < n:
            return -1  # End of data

        # Extract top n bits
        result = self.dwBits >> (32 - n)
        self.dwBits <<= n
        self.dwBits &= 0xFFFFFFFF  # Keep 32-bit
        self.nBits -= n
        return result


def decompress_lzw(data: bytes, expected_size: int) -> bytes:
    """Decompress SCI1 LZW-compressed data.

    SCI01/1 LZW uses:
    - MSB-first bit reading (matching ScummVM)
    - Variable code widths (9-12 bits)
    - "Early change" - codeLimit = (1 << nBits) - 1
    - Code 256 = reset dictionary
    - Code 257 = end of stream
    """
    if len(data) == 0:
        return bytes()

    reader = BitReaderMSB(data)

    # Constants
    RESET_CODE = 256
    END_CODE = 257
    MAX_TABLE_SIZE = 4096

    def reset_dictionary() -> List[bytes]:
        # 256 single-byte entries + reset + end placeholders
        return [bytes([i]) for i in range(256)] + [None, None]

    dictionary = reset_dictionary()
    code_width = 9
    table_size = 258  # Next entry to add
    code_limit = (1 << code_width) - 1  # 511 for 9 bits (early change)

    output = bytearray()
    prev_string = None

    while len(output) < expected_size:
        code = reader.get_bits(code_width)

        if code == -1 or code == END_CODE:
            break

        if code == RESET_CODE:
            dictionary = reset_dictionary()
            code_width = 9
            table_size = 258
            code_limit = (1 << code_width) - 1
            prev_string = None
            continue

        # Decode current code
        if code < len(dictionary) and dictionary[code] is not None:
            current_string = dictionary[code]
        elif code == table_size and prev_string is not None:
            # Special case: code not yet in dictionary (KwKwK case)
            current_string = prev_string + bytes([prev_string[0]])
        else:
            # Invalid code
            break

        output.extend(current_string)

        # Add new entry to dictionary
        if prev_string is not None and table_size < MAX_TABLE_SIZE:
            new_entry = prev_string + bytes([current_string[0]])
            if table_size < len(dictionary):
                dictionary[table_size] = new_entry
            else:
                dictionary.append(new_entry)
            table_size += 1

            # Increase width when table_size reaches code_limit
            if table_size == code_limit and code_width < 12:
                code_width += 1
                code_limit = (1 << code_width) - 1

        prev_string = current_string

    return bytes(output[:expected_size])
