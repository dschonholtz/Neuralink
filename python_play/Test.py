import click
import numpy as np
import os
import glob

# from BaseCompressor import (
#     MP3Compressor,
#     FLACCompressor,
#     ZIPCompressor,
#     HuffmanCompressor,
#     QuantizedCompressor,
#     RLECompressor,
# )
from LookUpCompressor import (
    LookUpCompressor,
    ZIPCompressor,
    MP3Compressor,
    FLACCompressor,
)


def print_compression_results(name: str, original_size: int, compressed_size: int):
    compression_ratio = original_size / compressed_size
    print(f"{name} Compression:")
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print()


def calculate_file_size(file_path):
    file_size = os.path.getsize(file_path)
    # print(file_size)
    return file_size


def process_compression(compressor, method, data_dir):
    # get the paths of all of the wav files in data_dir
    wav_files = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if file.endswith(".wav")
    ]
    lossless = True
    original_sizes = []
    compressed_sizes = []
    for wav_file in wav_files:
        c = compressor(data_dir)
        c.compress(wav_file)
        compressed_size = calculate_file_size(wav_file + ".brainwire")
        original_size = calculate_file_size(wav_file)
        original_sizes.append(original_size)
        compressed_sizes.append(compressed_size)
        # print_compression_results(method, original_size, compressed_size)
        c2 = compressor(None)
        c2.decompress(wav_file + ".brainwire", wav_file + ".copy")
        if lossless:
            try:
                lossless = (
                    c2.load_audio_file(wav_file)
                    == c2.load_audio_file(wav_file + ".copy")
                ).all()
            except Exception as e:
                print(
                    "Error when trying to compare wavs. Setting lossless to false ", e
                )
                lossless = False
    print(f"{method} original size: {sum(original_sizes) / len(original_sizes)}")
    print(f"{method} compressed size: {sum(compressed_sizes) / len(compressed_sizes)}")
    print(
        f"{method} compression ratio: {(sum(original_sizes) / len(original_sizes)) / (sum(compressed_sizes) / len(compressed_sizes))}"
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
    type=click.Choice(
        [
            "mp3",
            "flac",
            "zip",
            #  "huffman", "quantized", "rle",
            "lookup",
            "all",
        ]
    ),
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
        "mp3": MP3Compressor,
        "flac": FLACCompressor,
        "zip": ZIPCompressor,
        # "huffman": HuffmanCompressor(data_dir),
        # "quantized": QuantizedCompressor(data_dir),
        # "rle": RLECompressor(data_dir),
        "lookup": LookUpCompressor,  # Add LookUpCompressor to the dictionary
    }

    if method == "all":
        for name, compressor in compressors.items():
            process_compression(compressor, name, data_dir)
    else:
        process_compression(compressors[method], method, data_dir)


@cli.command()
def cleanup():
    brainwire_files = glob.glob("compressed_files/*.brainwire")
    for file in brainwire_files:
        os.remove(file)
    print(f"Deleted {len(brainwire_files)} .brainwire files.")


if __name__ == "__main__":
    cli()
