import click
import numpy as np
import os
import glob
from BaseCompressor import (
    MP3Compressor,
    FLACCompressor,
    ZIPCompressor,
    HuffmanCompressor,
    QuantizedCompressor,
    RLECompressor,  # Import the RLECompressor
)


def verify_lossless(original: np.ndarray, decompressed: np.ndarray) -> bool:
    return np.array_equal(original, decompressed)


def print_compression_results(name: str, original_size: int, compressed_size: int):
    compression_ratio = original_size / compressed_size
    print(f"{name} Compression:")
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print()


def calculate_file_sizes(file_paths):
    file_sizes = [os.path.getsize(file) for file in file_paths]
    print(file_sizes[0:5])
    return sum(file_sizes)


def process_compression(compressor, method, original_size):
    compressed_data = compressor.compress()
    compressed_files = compressor.write_compressed_files(compressed_data, method)
    compressed_size = calculate_file_sizes(compressed_files)
    print_compression_results(method, original_size, compressed_size)
    decompressed_data = compressor.decompress(compressed_data)
    lossless = all(
        verify_lossless(original, decompressed)
        for original, decompressed in zip(compressor.audio_data, decompressed_data)
    )
    if lossless:
        print(f"{method} compression is lossless!")
    else:
        print(f"{method} COMPRESSION IS NOT LOSSLESS!")


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--method",
    type=click.Choice(["mp3", "flac", "zip", "huffman", "quantized", "rle", "all"]),
    default="all",
    help="Compression method to use",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default="data/",
    help="Directory containing audio files",
)
def compress(method, data_dir):
    print("using method: ", method)
    compressors = {
        "mp3": MP3Compressor(data_dir),
        "flac": FLACCompressor(data_dir),
        "zip": ZIPCompressor(data_dir),
        "huffman": HuffmanCompressor(data_dir),
        "quantized": QuantizedCompressor(data_dir),
        "rle": RLECompressor(data_dir),  # Add RLECompressor to the dictionary
    }

    original_size = calculate_file_sizes(
        [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".wav")
        ]
    )

    if method == "all":
        for name, compressor in compressors.items():
            process_compression(compressor, name, original_size)
    else:
        process_compression(compressors[method], method, original_size)


@cli.command()
def cleanup():
    brainwire_files = glob.glob("compressed_files/*.brainwire")
    for file in brainwire_files:
        os.remove(file)
    print(f"Deleted {len(brainwire_files)} .brainwire files.")


if __name__ == "__main__":
    cli()
