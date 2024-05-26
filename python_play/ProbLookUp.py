"""
This is based off of the neural network checked compression idea combined with the look up table.

The thought is simple. We can do error correction and check if the value is in the look up table.
If the value is very common in the look up table for the next value, the next value for instance 
we just have to signify what category of data we are in.

So we will always send two category bits. 
We need to use two bits because we need to be able to signify more than just if the item is in the look up table.
The four categories will correspond with sections of the look up table and be determined at initialization via a 
passed in array.
[1, 4 + 1, 32 + 4 + 1]
This would suggest that the four categories are the first element, the second through 5th element, the fourth through 37 element
and then all elements after that will be referred to by their real data.

This so the first element in the look up table would be referenced simply by:
00

The 5th element in the look up table would be referenced by:
01 11
As this represents category 1 (zero indexed) and the last element in that category

10 11111
This would represent the last element in category 3 (1 indexed) as you need 5 bits to represent 32 and the first two bits represent 

In this dataset, we only see 1023 unique values. We are going to take a risk and assume that this continues to other datasets.
So the all values, that aren't captured by their look up table will be captured by 10 bits instead of by the full 16 they normally would.
I think this will be ok given how often the least common values are seen.
Value: -24568, Count: 6
Value: 20467, Count: 6
Value: 26617, Count: 7
Value: 28667, Count: 8
Value: 16367, Count: 14
Value: -20468, Count: 21
Value: -28668, Count: 27
Value: 30717, Count: 36
Value: -26618, Count: 49
Value: 18417, Count: 51 

Therefore to represent a value in the last category we would do something like:
11 00000 00000
Two starter bits, then 10 bits to give us a value 0 through 1023

If you believe the jupyter notebook calculations (I have reason not to)
This should theoretically give us a compression ratio of 3.2 45% better than the current zlib compression.
"""

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
        use_category_bits: bool = True,
    ):
        super().__init__(data_dir)
        self.table_size = table_size
        self.zlib = zlib
        self.use_category_bits = use_category_bits
        self.lookup_table, self.next_value_prob, self.index_map = (
            self.build_lookup_table(data_dir)
        )
        self.lookup_dict = {value: idx for idx, value in enumerate(self.lookup_table)}
        self.lookup_hits = 0
        self.total_hits = 0
        self.categories = [
            1,
            4 + 1,
            32 + 4 + 1,
        ]  # Define categories as per the docstring

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
            return f"10{index - self.categories[1]:05b}"
        else:
            return f"11{index:010b}"

    def build_lookup_table(self, data_dir: str) -> (np.ndarray, dict, dict):
        lookup_table_file = "lookup_table.npy"
        next_value_prob_file = "next_value_prob.npy"
        index_map_file = "index_map.npy"

        if (
            os.path.exists(lookup_table_file)
            and os.path.exists(next_value_prob_file)
            and os.path.exists(index_map_file)
        ):
            lookup_table = np.load(lookup_table_file)
            next_value_prob = np.load(next_value_prob_file, allow_pickle=True).item()
            index_map = np.load(index_map_file, allow_pickle=True).item()
            return lookup_table, next_value_prob, index_map

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

        # Initialize the next value probability dictionary
        next_value_prob = {value: {} for value in unique}

        # Calculate the next value probabilities
        for audio in audio_data[:4]:
            for i in range(len(audio) - 1):
                current_value = audio[i]
                next_value = audio[i + 1]
                if next_value in next_value_prob[current_value]:
                    next_value_prob[current_value][next_value] += 1
                else:
                    next_value_prob[current_value][next_value] = 1

        # Normalize the probabilities
        for current_value, next_values in next_value_prob.items():
            total_count = sum(next_values.values())
            for next_value in next_values:
                next_value_prob[current_value][next_value] /= total_count

        # Create the index map for all unique values
        index_map = {value: idx for idx, value in enumerate(unique)}

        # Save the lookup table, next value probability dictionary, and index map
        np.save(lookup_table_file, lookup_table)
        np.save(next_value_prob_file, next_value_prob)
        np.save(index_map_file, index_map)

        return lookup_table, next_value_prob, index_map

    def compress(self, filename: str):
        """
        This iterates through the list of int16.
        If no previous value or the previous value is not in the look up table.
        Then we attempt to return the index of the value of the most common values in the look up table.
        If the value is not in the look up table we add a 1 and then the full int16 value. to the output.
        If the value is in the look up table we add a 0 and then the index of the value in the look up table. to the output.
        The binary representation of that is the length of the indicator bit + the minimum binary representation of table_size.
        So if table_size is 64 we will have a 1 + 6 bits for the index.
        Finally, when we are done with the whole array, we pad with zeros.
        """
        compressed_bits = []
        index_bits = (
            self.table_size - 1
        ).bit_length()  # Calculate the number of bits needed for indices
        audio = self.load_audio_file(Path(filename))

        # Add the count of int16 values as a 32-bit binary string
        count_bits = f"{len(audio):032b}"
        compressed_bits.append(count_bits)

        for i, value in enumerate(audio):
            self.total_hits += 1
            if i == 0 or value not in self.lookup_dict:
                if value in self.lookup_dict:
                    index = self.lookup_dict[value]
                    self.lookup_hits += 1
                    if self.use_category_bits:
                        category = self.get_category(index)
                        compressed_bits.append(
                            self.get_bits_for_category(category, index)
                        )
                    else:
                        compressed_bits.append(
                            f"0{index:0{index_bits}b}"
                        )  # 0 bit for flag + index_bits for index
                else:
                    # Use 10 bits for values not in the lookup table
                    index = self.index_map[value]
                    compressed_bits.append(
                        f"1{index:010b}"
                    )  # 1 bit for flag + 10 bits for index
            else:
                index = self.lookup_dict[value]
                if self.use_category_bits:
                    category = self.get_category(index)
                    compressed_bits.append(self.get_bits_for_category(category, index))
                else:
                    compressed_bits.append(
                        f"0{index:0{index_bits}b}"
                    )  # 0 bit for flag + index_bits for index
                self.lookup_hits += 1

        # Join all bits into a single string
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

        index_bits = (
            self.table_size - 1
        ).bit_length()  # Calculate the number of bits needed for indices
        bit_string = "".join(f"{byte:08b}" for byte in compressed_data)

        # Read the count of int16 values from the first 32 bits
        count = int(bit_string[:32], 2)
        bit_string = bit_string[32:]

        decompressed_data = []
        i = 0
        while len(decompressed_data) < count:
            if self.use_category_bits:
                category_bits = bit_string[i : i + 2]
                i += 2
                if category_bits == "00":
                    index = 0
                    decompressed_data.append(self.lookup_table[index])
                elif category_bits == "01":
                    index = int(bit_string[i : i + 2], 2) + self.categories[0]
                    i += 2
                    decompressed_data.append(self.lookup_table[index])
                elif category_bits == "10":
                    index = int(bit_string[i : i + 5], 2) + self.categories[1]
                    i += 5
                    decompressed_data.append(self.lookup_table[index])
                else:  # "11"
                    index = int(bit_string[i : i + 10], 2)
                    i += 10
                    value = list(self.index_map.keys())[
                        list(self.index_map.values()).index(index)
                    ]
                    decompressed_data.append(value)
            else:
                if bit_string[i] == "0":
                    index = int(bit_string[i + 1 : i + 1 + index_bits], 2)
                    if index < len(self.lookup_table) - 1:
                        decompressed_data.append(self.lookup_table[index])
                    else:
                        decompressed_data.append(
                            self.lookup_table[-1]
                        )  # Use the default value
                    i += 1 + index_bits  # 1 bit for flag + index_bits for index
                else:
                    index = int(bit_string[i + 1 : i + 11], 2)
                    value = list(self.index_map.keys())[
                        list(self.index_map.values()).index(index)
                    ]
                    decompressed_data.append(value)
                    i += 11  # 1 bit for flag + 10 bits for index

        # Convert the decompressed data back to the original audio data format
        audio_data = np.array(decompressed_data, dtype=np.int16)
        self.write_audio_file(audio_data, Path(output_filename))
