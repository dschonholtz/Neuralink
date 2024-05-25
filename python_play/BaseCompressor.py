from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import os


class BaseCompressor(ABC):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.audio_data: List[np.ndarray] = self.load_audio_files()

    def load_audio_files(self) -> List[np.ndarray]:
        audio_files = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.data_dir, file_name)
                audio, _ = sf.read(file_path, dtype="int16")
                audio_files.append(audio)
        return audio_files

    def write_audio_files(self, audio_data: List[np.ndarray], output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, audio in enumerate(audio_data):
            output_path = os.path.join(output_dir, f"audio_{i}.wav")
            sf.write(output_path, audio.T, 19531)  # Assuming a sample rate of 19531

    def write_compressed_files(
        self,
        compressed_data: List[bytes],
        prefix: str,
        subfolder: str = "compressed_files",
    ) -> List[str]:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        file_paths = []
        for i, data in enumerate(compressed_data):
            file_path = os.path.join(subfolder, f"{prefix}_{i}.brainwire")
            with open(file_path, "wb") as f:
                f.write(data)
            file_paths.append(file_path)
        return file_paths

    @abstractmethod
    def compress(self) -> List[bytes]:
        pass

    @abstractmethod
    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        pass


class MP3Compressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        for i, audio in enumerate(self.audio_data):
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=19531,  # Updated frame_rate
                sample_width=2,
                channels=1,
            )
            buffer = BytesIO()
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)  # Reset buffer position to the beginning
            compressed_data.append(buffer.getvalue())
        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []
        for data in compressed_data:
            buffer = BytesIO(data)
            audio_segment = AudioSegment.from_file(buffer, format="mp3")
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            decompressed_data.append(
                audio_array.reshape((-1, audio_segment.channels)).T
            )
        return decompressed_data


class FLACCompressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        for audio in self.audio_data:
            buffer = BytesIO()
            sf.write(buffer, audio.T, 19531, format="FLAC")  # Updated frame_rate
            compressed_data.append(buffer.getvalue())
        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []
        for data in compressed_data:
            buffer = BytesIO(data)
            audio_data, _ = sf.read(buffer, dtype="int16")
            decompressed_data.append(audio_data.T)
        return decompressed_data


import zlib


class ZIPCompressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        for audio in self.audio_data:
            compressed_data.append(zlib.compress(audio.tobytes()))
        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []
        for i, data in enumerate(compressed_data):
            decompressed_data.append(
                np.frombuffer(zlib.decompress(data), dtype=np.int16).reshape(
                    self.audio_data[i].shape
                )
            )
        return decompressed_data


import heapq
from collections import defaultdict, Counter


class HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def build_huffman_tree(frequencies):
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.frequency + node2.frequency)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def build_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.value is not None:
        codebook[node.value] = prefix
    else:
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)

    return codebook


class HuffmanCompressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        for audio in self.audio_data:
            flat_audio = audio.flatten()
            frequencies = Counter(flat_audio)
            huffman_tree = build_huffman_tree(frequencies)
            huffman_codes = build_huffman_codes(huffman_tree)

            encoded_audio = "".join(huffman_codes[sample] for sample in flat_audio)
            padded_encoded_audio = encoded_audio + "0" * (
                (8 - len(encoded_audio) % 8) % 8
            )
            byte_array = bytearray(
                int(padded_encoded_audio[i : i + 8], 2)
                for i in range(0, len(padded_encoded_audio), 8)
            )
            compressed_data.append(bytes(byte_array))

        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []
        for i, data in enumerate(compressed_data):
            byte_string = "".join(f"{byte:08b}" for byte in data)
            frequencies = Counter(self.audio_data[i].flatten())
            huffman_tree = build_huffman_tree(frequencies)
            current_node = huffman_tree
            decoded_audio = []

            for bit in byte_string:
                current_node = current_node.left if bit == "0" else current_node.right
                if current_node.value is not None:
                    decoded_audio.append(current_node.value)
                    current_node = huffman_tree

            # Remove padding bits
            decoded_audio = decoded_audio[: len(self.audio_data[i].flatten())]

            decompressed_data.append(
                np.array(decoded_audio, dtype=np.int16).reshape(
                    self.audio_data[i].shape
                )
            )

        return decompressed_data


class QuantizedCompressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        self.original_min_max = (
            []
        )  # Store original min and max values for decompression
        print("audio type: ", self.audio_data[0].dtype)
        i = 0
        # assert all types are int16 and throw an error otherwise
        for audio in self.audio_data:
            i += 1
            if audio.dtype != np.int16:
                raise ValueError(f"Audio data {i} must be int16")
            # Store original min and max values
            min_val = audio.min()
            max_val = audio.max()
            self.original_min_max.append((min_val, max_val))

            # Normalize audio to range [0, 1]
            normalized_audio = (audio - min_val) / (max_val - min_val)

            # Scale to range [0, 65535] and convert to uint16
            quantized_audio = (normalized_audio * 256).astype(np.uint8)

            # Convert to bytes and store
            compressed_data.append(quantized_audio.tobytes())
            if i == 1:
                # print the min val max val shape of the quantized audio, normalized audio and original audio
                print("min val: ", min_val, "max val: ", max_val)
                print("quantized audio shape: ", quantized_audio.shape)
                print("normalized audio shape: ", normalized_audio.shape)
                print("original audio shape: ", audio.shape)
                print("uncompressed data size in bytes: ", len(audio.tobytes()))
                print("compressed data size in bytes: ", len(compressed_data[0]))

        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []

        for i, data in enumerate(compressed_data):
            # Convert bytes back to uint16 array
            quantized_audio = np.frombuffer(data, dtype=np.uint8)

            # Retrieve original min and max values
            min_val, max_val = self.original_min_max[i]
            # Normalize back to range [0, 1]
            normalized_audio = quantized_audio / 256.0

            # Rescale to original range
            audio = normalized_audio * (max_val - min_val) + min_val

            # Convert back to int16
            audio = audio.astype(np.int16)

            # Reshape to original shape
            decompressed_data.append(audio.reshape(self.audio_data[i].shape))

        return decompressed_data


class RLECompressor(BaseCompressor):
    def compress(self) -> List[bytes]:
        compressed_data = []
        for audio in self.audio_data:
            flat_audio = audio.flatten()
            compressed_audio = []
            duplicates_info = []
            i = 0
            while i < len(flat_audio):
                count = 1
                while (
                    i + count < len(flat_audio)
                    and flat_audio[i] == flat_audio[i + count]
                ):
                    count += 1
                if count > 3:
                    duplicates_info.append((i, count))
                    compressed_audio.append(flat_audio[i])
                    i += count
                else:
                    compressed_audio.extend(flat_audio[i : i + count])
                    i += count

            # Build the compressed data
            compressed_bytes = bytearray()
            compressed_bytes.append(
                len(duplicates_info)
            )  # Count of duplicate sequences
            for loc, cnt in duplicates_info:
                compressed_bytes.extend(loc.to_bytes(4, "big"))  # Location (4 bytes)
                compressed_bytes.extend(cnt.to_bytes(4, "big"))  # Count (4 bytes)
            compressed_bytes.extend(
                np.array(compressed_audio, dtype=np.int16).tobytes()
            )  # Raw values

            compressed_data.append(bytes(compressed_bytes))
        return compressed_data

    def decompress(self, compressed_data: List[bytes]) -> List[np.ndarray]:
        decompressed_data = []
        for i, data in enumerate(compressed_data):
            byte_stream = BytesIO(data)
            count_of_duplicates = int.from_bytes(byte_stream.read(1), "big")
            duplicates_info = []
            for _ in range(count_of_duplicates):
                loc = int.from_bytes(byte_stream.read(4), "big")
                cnt = int.from_bytes(byte_stream.read(4), "big")
                duplicates_info.append((loc, cnt))

            compressed_audio = np.frombuffer(byte_stream.read(), dtype=np.int16)
            decompressed_audio = []
            j = 0
            for loc, cnt in duplicates_info:
                decompressed_audio.extend(compressed_audio[j:loc])
                decompressed_audio.extend([compressed_audio[loc]] * cnt)
                j = loc + 1
            decompressed_audio.extend(compressed_audio[j:])

            decompressed_data.append(
                np.array(decompressed_audio, dtype=np.int16).reshape(
                    self.audio_data[i].shape
                )
            )
        return decompressed_data
