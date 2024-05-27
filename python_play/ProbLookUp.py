import os
import numpy as np
import zlib
from pathlib import Path

from BetterBaseCompressor import BetterBaseCompressor


class ProbLookUpCompressor(BetterBaseCompressor):
    def __init__(
        self,
        data_dir: str,
        table_size: int = 128,
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
        self.categories = [
            1,
            4 + 1,
            123 + 4 + 1,
        ]
        if self.categories[3] != self.table_size:
            raise ValueError("categories[3] != table_size")

    def get_category(self, index: int) -> int:
        if index < self.categories[0]:
            return 0
        elif index < self.categories[1]:
            return 1
        elif index < self.categories[2]:
            return 2
        else:
            return 3

    def get_bits_for_category(self, category: int, index: int) -> str:
        if category == 0:
            return f"00"
        elif category == 1:
            return f"01{index - self.categories[0]:02b}"
        elif category == 2:
            return f"10{index - self.categories[1]:07b}"
        else:
            return f"11{index:010b}"

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
        # print("compressing: ", filename)
        compressed_bits = []
        audio = self.load_audio_file(Path(filename))
        # print("audio length: ", len(audio))
        # print("first 5 vals: ", audio[:5])
        # print("last 5 vals: ", audio[-5:])

        # Add the count of int16 values as a 32-bit binary string
        count_bits = f"{len(audio):032b}"
        compressed_bits.append(count_bits)

        # Buffer to store the last 5 values
        last_five_values = []
        first_index = None
        last_index = None

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
            if first_index is None:
                first_index = index
            last_index = index

            # Update the last five values buffer
            last_five_values.append(value)
            if len(last_five_values) > 5:
                last_five_values.pop(0)

        bit_string = "".join(compressed_bits)
        # print("raw bit string length: ", len(bit_string))
        # print("first index: ", first_index)
        # print("last index: ", last_index)
        # print("lookuptable 31:65", self.lookup_table[31:65])
        # print the dict values and their indices between 31 and 65
        # for value in self.lookup_table[31:65]:
        #     print(f"{value}: {self.lookup_dict[value]}")

        # Pad the bit string to make its length a multiple of 8
        padding_length = (8 - len(bit_string) % 8) % 8
        bit_string = bit_string + "0" * padding_length
        # print("padded bit string length: ", len(bit_string))

        # Convert the bit string to bytes
        byte_array = int(bit_string, 2).to_bytes(
            (len(bit_string) + 7) // 8, byteorder="big"
        )
        # print("byte array length: ", len(byte_array))

        if self.zlib:
            byte_array = zlib.compress(byte_array, level=zlib.Z_BEST_COMPRESSION)

        self.write_compressed_file(byte_array, Path(filename))

        # Print the last five values
        # print("Last 5 values compressed:", last_five_values)

    def decompress(self, input_filename: str, output_filename: str):
        # print("decompressing: ", input_filename)
        compressed_data = self.load_compressed_file(Path(input_filename))
        # print("compressed data length: ", len(compressed_data))

        if self.zlib:
            compressed_data = zlib.decompress(compressed_data)

        bit_string = "".join(f"{byte:08b}" for byte in compressed_data)
        # print("bit string length: ", len(bit_string))

        # Read the count of int16 values from the first 32 bits
        count = int(bit_string[:32], 2)
        bit_string = bit_string[32:]

        decompressed_data = []
        last_decoded = None
        first_index = None
        last_index = None
        i = 0
        while len(decompressed_data) < count:
            category_bits = bit_string[i : i + 2]
            i += 2
            # print(f"Category bits: {category_bits}")
            if category_bits == "00":
                index = 0
                decompressed_data.append(self.lookup_table[0])
            elif category_bits == "01":
                index = int(bit_string[i : i + 2], 2) + self.categories[0]
                i += 2
                decompressed_data.append(self.lookup_table[index])
            elif category_bits == "10":
                index = int(bit_string[i : i + 7], 2) + self.categories[1]
                i += 7
                decompressed_data.append(self.lookup_table[index])
            else:  # "11"
                try:
                    index = int(bit_string[i : i + 10], 2)
                except:
                    print("bit string: ", bit_string[i : i + 10])
                    print("index out of bounds")
                    print("i: ", i)
                    print("index: ", index)
                    print("last_decoded: ", last_decoded)
                    print("len(self.lookup_table): ", len(self.lookup_table))
                    print("count: ", count)
                    print("len(decompressed_data): ", len(decompressed_data))
                    print("first 5 vals: ", decompressed_data[:5])
                    print("last 5 vals: ", decompressed_data[-5:])
                    print("first index: ", first_index)
                    print("last index: ", last_index)
                    print("lookuptable 31:65", self.lookup_table[31:65])
                    raise
                i += 10
                value = list(self.index_map.keys())[
                    list(self.index_map.values()).index(index)
                ]
                decompressed_data.append(value)
            last_decoded = decompressed_data[-1]
            if first_index is None:
                first_index = index
            last_index = index

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
