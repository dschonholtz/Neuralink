import os
import numpy as np
import zlib
from pathlib import Path

from BetterBaseCompressor import BetterBaseCompressor


class ProbLookUpCompressor(BetterBaseCompressor):
    def __init__(
        self,
        data_dir: str,
        table_size: int = 32,
        zlib: bool = False,
    ):
        super().__init__(data_dir)
        self.table_size = table_size
        self.zlib = zlib

        # self.next_value_prob,
        self.lookup_table, self.index_map = self.build_lookup_table(data_dir)
        self.lookup_dict = {value: idx for idx, value in enumerate(self.lookup_table)}
        self.lookup_hits = 0
        self.total_hits = 0
        self.values_compressed = 0
        self.values_decompressed = 0

        # Initialize categories based on ranges
        self.categories = [1, 5, 33]  # Ends of each category range
        self.category_bits = [0] + [
            self.calculate_bits(n - self.categories[i - 1])
            for i, n in enumerate(self.categories[1:], start=1)
        ]

    def calculate_bits(self, num_values):
        from math import ceil, log2

        return ceil(log2(num_values))

    def get_category(self, index: int) -> int:
        for i, cat_end in enumerate(self.categories):
            if index < cat_end:
                return i
        return len(self.categories) - 1

    def get_bits_for_category(self, category: int, index: int) -> str:
        if category == 0:
            return "00"  # No additional bits needed, only one value
        offset = self.categories[category - 1] if category > 0 else 0
        bits_needed = self.category_bits[category]
        return f"{format(category, '02b')}{index - offset:0{bits_needed}b}"

    def build_lookup_table(self, data_dir: str) -> (np.ndarray, dict):
        lookup_table_file = "lookup_table.npy"
        index_map_file = "index_map.npy"

        if os.path.exists(lookup_table_file) and os.path.exists(index_map_file):
            lookup_table = np.load(lookup_table_file)
            index_map = np.load(index_map_file, allow_pickle=True).item()
            return lookup_table, index_map

        if data_dir is None:
            raise ValueError(
                "data_dir is None and no lookup table files found on disk."
            )

        # Find the audio files in the directory
        audio_files = [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".wav")
        ]
        # Load the audio files
        audio_data = [self.load_audio_file(Path(file)) for file in audio_files]
        # Flatten the audio data and find the most common values
        flat_data = np.concatenate(audio_data)
        unique, counts = np.unique(flat_data, return_counts=True)
        sorted_indices = np.argsort(-counts)
        lookup_table = unique[sorted_indices[: self.table_size - 1]]

        # Add a placeholder for default values at the last index
        lookup_table = np.append(lookup_table, -1)

        # Create the index map for all unique values
        index_map = {value: idx for idx, value in enumerate(unique)}

        # Save the lookup table, next value probability dictionary, and index map
        np.save(lookup_table_file, lookup_table)
        np.save(index_map_file, index_map)

        return lookup_table, index_map

    def compress(self, filename: str):
        compressed_bits = []
        audio = self.load_audio_file(Path(filename))

        # Add the count of int16 values as a 32-bit binary string
        count_bits = f"{len(audio):032b}"
        compressed_bits.append(count_bits)

        for i, value in enumerate(audio):
            if i == 0 or value not in self.lookup_dict:
                if value in self.lookup_dict:
                    index = self.lookup_dict[value]
                    self.lookup_hits += 1
                    category = self.get_category(index)
                    compressed_bits.append(self.get_bits_for_category(category, index))
                else:
                    index = self.index_map[value]
                    compressed_bits.append(f"11{index:010b}")
            else:
                index = self.lookup_dict[value]
                category = self.get_category(index)
                compressed_bits.append(self.get_bits_for_category(category, index))

        bit_string = "".join(compressed_bits)

        # Pad the bit string to make its length a multiple of 8
        padding_length = (8 - len(bit_string) % 8) % 8
        bit_string = bit_string + "0" * padding_length

        # Convert the bit string to bytes
        byte_array = int(bit_string, 2).to_bytes(
            (len(bit_string) + 7) // 8, byteorder="big"
        )

        if self.zlib:
            byte_array = zlib.compress(byte_array, level=zlib.Z_BEST_COMPRESSION)

        self.write_compressed_file(byte_array, Path(filename))

    def decompress(self, input_filename: str, output_filename: str):
        compressed_data = self.load_compressed_file(Path(input_filename))

        if self.zlib:
            compressed_data = zlib.decompress(compressed_data)

        bit_string = "".join(f"{byte:08b}" for byte in compressed_data)

        # Read the count of int16 values from the first 32 bits
        count = int(bit_string[:32], 2)
        bit_string = bit_string[32:]

        decompressed_data = []
        i = 0
        while len(decompressed_data) < count:
            category_bits = bit_string[i : i + 2]
            i += 2
            if category_bits == "00":
                index = 0
                decompressed_data.append(self.lookup_table[0])
            elif category_bits == "01":
                index = (
                    int(bit_string[i : i + self.category_bits[1]], 2)
                    + self.categories[0]
                )
                i += self.category_bits[1]
                decompressed_data.append(self.lookup_table[index])
            elif category_bits == "10":
                index = (
                    int(bit_string[i : i + self.category_bits[2]], 2)
                    + self.categories[1]
                )
                i += self.category_bits[2]
                decompressed_data.append(self.lookup_table[index])
            else:  # "11"
                index = int(bit_string[i : i + 10], 2)
                i += 10
                value = list(self.index_map.keys())[
                    list(self.index_map.values()).index(index)
                ]
                decompressed_data.append(value)

        # Convert the decompressed data back to the original audio data format
        audio_data = np.array(decompressed_data, dtype=np.int16)
        self.write_audio_file(audio_data, Path(output_filename))


if __name__ == "__main__":
    compressor = ProbLookUpCompressor(data_dir="data")
    compressor.compress("data/a05767ff-af49-4cae-852e-c463a09a85c8.wav")
    compressor.decompress(
        "data/a05767ff-af49-4cae-852e-c463a09a85c8.wav.brainwire",
        "data/a05767ff-af49-4cae-852e-c463a09a85c8.wav.copy",
    )
