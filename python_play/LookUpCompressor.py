import os
import numpy as np
from typing import List
from abc import ABC
import soundfile as sf
from pathlib import Path
import zlib
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import os


class BetterBaseCompressor(ABC):

    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir

    def load_audio_file(self, filepath: Path) -> np.ndarray:
        audio, _ = sf.read(filepath, dtype="int16")
        return audio

    def write_audio_file(self, audio_data: np.ndarray, output_path: Path):
        # make the parent path if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_data.T, 19531, format="WAV")

    def write_compressed_file(
        self,
        compressed_data: bytes,
        prefix: str,
    ):
        file_path = os.path.join(f"{prefix}.brainwire")
        with open(file_path, "wb") as f:
            f.write(compressed_data)

    def load_compressed_file(self, filepath: Path) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()

    def compress(self, filename: str):
        pass

    def decompress(self, input_filename: str, output_filename: str):
        pass


class MP3Compressor(BetterBaseCompressor):
    def compress(self, filename: str):
        audio = self.load_audio_file(Path(filename))
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=19531,  # Updated frame_rate
            sample_width=2,
            channels=1,
        )
        buffer = BytesIO()
        audio_segment.export(buffer, format="mp3")
        buffer.seek(0)  # Reset buffer position to the beginning
        self.write_compressed_file(buffer.getvalue(), Path(filename))

    def decompress(self, input_filename: str, output_filename: str):
        compressed_data = self.load_compressed_file(Path(input_filename))
        buffer = BytesIO(compressed_data)
        audio_segment = AudioSegment.from_file(buffer, format="mp3")
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
        self.write_audio_file(audio_array, Path(output_filename))


class FLACCompressor(BetterBaseCompressor):
    def compress(self, filename: str):
        audio = self.load_audio_file(Path(filename))
        buffer = BytesIO()
        sf.write(buffer, audio.T, 19531, format="FLAC")  # Updated frame_rate

        self.write_compressed_file(buffer.getvalue(), Path(filename))

    def decompress(self, input_filename: str, output_filename: str):
        compressed_data = self.load_compressed_file(Path(input_filename))
        buffer = BytesIO(compressed_data)
        audio_data, _ = sf.read(buffer, dtype="int16")
        self.write_audio_file(audio_data, Path(output_filename))


import zlib


class ZIPCompressor(BetterBaseCompressor):
    def compress(self, filename: str):
        audio = self.load_audio_file(Path(filename))
        compressed_data = zlib.compress(audio.tobytes())
        self.write_compressed_file(compressed_data, Path(filename))

    def decompress(self, input_filename, output_filename):
        compressed_data = self.load_compressed_file(Path(input_filename))
        audio = np.frombuffer(zlib.decompress(compressed_data), dtype=np.int16)
        self.write_audio_file(audio, Path(output_filename))


class LookUpCompressor(BetterBaseCompressor):
    def __init__(self, data_dir: str, table_size: int = 128, zlib: bool = True):
        super().__init__(data_dir)
        self.table_size = table_size
        self.zlib = zlib
        self.lookup_table, self.next_value_prob = self.build_lookup_table(data_dir)
        self.lookup_dict = {value: idx for idx, value in enumerate(self.lookup_table)}

    def build_lookup_table(self, data_dir: str) -> (np.ndarray, dict):
        lookup_table_file = "lookup_table.npy"
        next_value_prob_file = "next_value_prob.npy"

        if os.path.exists(lookup_table_file) and os.path.exists(next_value_prob_file):
            lookup_table = np.load(lookup_table_file)
            next_value_prob = np.load(next_value_prob_file, allow_pickle=True).item()
            return lookup_table, next_value_prob

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

        # Save the lookup table and next value probability dictionary
        np.save(lookup_table_file, lookup_table)
        np.save(next_value_prob_file, next_value_prob)

        return lookup_table, next_value_prob

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
            if i == 0 or value not in self.lookup_dict:
                if value in self.lookup_dict:
                    index = self.lookup_dict[value]
                    compressed_bits.append(
                        f"0{index:0{index_bits}b}"
                    )  # 0 bit for flag + index_bits for index
                else:
                    compressed_bits.append(
                        f"1{value & 0xFFFF:016b}"
                    )  # 1 bit for flag + 16 bits for value upscaled to be uint16
            else:
                index = self.lookup_dict[value]
                compressed_bits.append(
                    f"0{index:0{index_bits}b}"
                )  # 1 bit for flag + index_bits for index

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
                value = int(bit_string[i + 1 : i + 17], 2)
                decompressed_data.append(value)
                i += 17  # 1 bit for flag + 16 bits for value

        # Convert the decompressed data back to the original audio data format
        audio_data = np.array(decompressed_data, dtype=np.int16)
        self.write_audio_file(audio_data, Path(output_filename))
