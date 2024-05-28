import click
import numpy as np
import os
import pickle
import zlib  # Changed from lzma to zlib
import lzma
from Tokenizer import (
    load_audio_data,
    load_lookup_tables,
    value_to_binary_string,
    load_lookup_tables,
)
from ReadAudioFiles import save_audio_data, load_all_audio_data, load_audio_data


def build_bigram_model(data: np.ndarray, lookup_file_path: str) -> np.ndarray:
    # Initialize a 2D array to count occurrences of bigrams
    bigram_counts = np.zeros((1024, 1024), dtype=int)
    val_to_idx, _, idx_to_val, _ = load_lookup_tables(lookup_file_path)

    # Count occurrences of bigrams
    for i in range(len(data) - 1):
        current_value = data[i]
        next_value = data[i + 1]
        if current_value in val_to_idx and next_value in val_to_idx:
            current_index = val_to_idx[current_value]
            next_index = val_to_idx[next_value]
            bigram_counts[current_index, next_index] += 1
        else:
            raise ValueError("Value not found in lookup tables")

    # Sort the next tokens by frequency for each token in descending order
    sorted_next_tokens = np.argsort(-bigram_counts, axis=1)

    return sorted_next_tokens


def save_bigram_model(bigram_model: np.ndarray, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(bigram_model, f)


def load_bigram_model(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as f:
        bigram_model = pickle.load(f)
    return bigram_model


def compress_data(
    data: np.ndarray, bigram_model: np.ndarray, val_to_idx: np.ndarray
) -> str:
    compressed_data = ""

    # Add the first value as the full lookup value, prepended by 11
    first_value = data[0]
    if first_value in val_to_idx:
        # The first bit is to indicate this uses the full look up.
        # that is already encoded in the two leading bits here.
        compressed_data += (
            f"11{value_to_binary_string(first_value, val_to_idx, {})[1:]}"
        )
    else:
        raise ValueError("First value not found in lookup tables")

    cat_1 = 0
    cat_2 = 0
    cat_3 = 0
    cat_4 = 0
    for i in range(len(data) - 1):
        current_value = data[i]
        next_value = data[i + 1]
        if current_value in val_to_idx and next_value in val_to_idx:
            current_index = val_to_idx[current_value]
            next_index = val_to_idx[next_value]
            sorted_next_tokens = bigram_model[current_index]

            if next_index == sorted_next_tokens[0]:
                compressed_data += "00"
                cat_1 += 1
            else:
                position = np.where(sorted_next_tokens == next_index)[0][0]
                if position < 4:
                    compressed_data += f"01{position:02b}"
                    cat_2 += 1
                elif position < 32:
                    compressed_data += f"10{position:05b}"
                    cat_3 += 1
                else:
                    # The first bit is to indicate this uses the full look up.
                    # that is already encoded in the two leading bits here.
                    compressed_data += (
                        f"11{value_to_binary_string(next_value, val_to_idx, {})[1:]}"
                    )
                    cat_4 += 1
        else:
            raise ValueError("Value not found in lookup tables")
    # print(f"Cat 1: {cat_1}, Cat 2: {cat_2}, Cat 3: {cat_3}, Cat 4: {cat_4}")
    return compressed_data


def bitstring_to_bytearray(bitstring: str) -> bytearray:
    # Pad the bitstring to make its length a multiple of 8
    padding_length = (8 - len(bitstring) % 8) % 8
    bitstring = bitstring + "0" * padding_length

    # Convert the bitstring to a bytearray
    byte_array = bytearray(
        int(bitstring[i : i + 8], 2) for i in range(0, len(bitstring), 8)
    )
    # print(f"Bitstring length: {len(bitstring)}")
    # print(f"Byte array length: {len(byte_array)}")

    return byte_array


def byte_array_to_bitstring(byte_array: bytearray) -> str:
    bitstring = "".join(f"{byte:08b}" for byte in byte_array)
    return bitstring


def save_compressed_data(compressed_data: bytearray, original_file_path: str):
    brainwire_file_path = original_file_path + ".brainwire"
    with open(brainwire_file_path, "wb") as f:
        f.write(compressed_data)


def load_compressed_data(brainwire_file_path: str) -> str:
    with open(brainwire_file_path, "rb") as f:
        compressed_data = f.read()
    return compressed_data


def _compress(wav_file_path: str, lookup_file_path: str, bigram_model_path: str):
    # Load audio data
    audio_data = load_audio_data(wav_file_path)

    # Load lookup tables from disk
    val_to_idx, _, idx_to_val, _ = load_lookup_tables(lookup_file_path)

    # Load bigram model from file
    bigram_model = load_bigram_model(bigram_model_path)

    # Compress data
    compressed_data = compress_data(audio_data, bigram_model, val_to_idx)

    # Add the total number of values as a 32-bit integer at the beginning
    total_values = len(audio_data)
    total_values_binary = f"{total_values:032b}"
    compressed_data = total_values_binary + compressed_data

    byte_array = bitstring_to_bytearray(compressed_data)

    # Apply zlib compression
    # compressed_data = zlib.compress(byte_array)  # Changed from lzma.compress
    compressed_data = lzma.compress(byte_array)

    # Save compressed data to a file
    save_compressed_data(compressed_data, wav_file_path)  # Changed variable name

    # print(f"Compressed data saved to {wav_file_path}.brainwire")


def _decompress(
    brainwire_file_path: str,
    lookup_file_path: str,
    bigram_model_path: str,
    output_file_path: str,
):
    # Load compressed data
    compressed_bytes = load_compressed_data(brainwire_file_path)

    # Apply zlib decompression
    # decompressed_bytes = zlib.decompress(
    #     compressed_bytes
    # )  # Changed from lzma.decompress
    decompressed_bytes = lzma.decompress(compressed_bytes)

    compressed_data = byte_array_to_bitstring(decompressed_bytes)

    # Extract the total number of values from the first 32 bits
    total_values = int(compressed_data[:32], 2)
    compressed_data = compressed_data[32:]

    # Load lookup tables from disk
    val_to_idx, _, idx_to_val, _ = load_lookup_tables(lookup_file_path)

    # Load bigram model from file
    bigram_model = load_bigram_model(bigram_model_path)

    # Decompress data
    decompressed_data = []
    i = 0
    prev_idx = None
    while i < len(compressed_data) and len(decompressed_data) < total_values:
        # if i > len(compressed_data) - 30:
        #     print("last 50 decompressed: ", decompressed_data[-30:])
        if compressed_data[i : i + 2] == "11":
            i += 2
            value = int(compressed_data[i : i + 10], 2)
            prev_idx = value
            decompressed_data.append(idx_to_val[value])
            i += 10
        elif compressed_data[i : i + 2] == "00":
            prev_idx = bigram_model[prev_idx][0]
            decompressed_data.append(idx_to_val[prev_idx])
            i += 2
        elif compressed_data[i : i + 2] == "01":
            i += 2
            index = int(compressed_data[i : i + 2], 2)
            prev_idx = bigram_model[prev_idx][index]
            decompressed_data.append(idx_to_val[prev_idx])
            i += 2
        elif compressed_data[i : i + 2] == "10":
            i += 2
            index = int(compressed_data[i : i + 5], 2)
            prev_idx = bigram_model[prev_idx][index]
            decompressed_data.append(idx_to_val[prev_idx])
            i += 5

    # print("last 50 decompressed: ", decompressed_data[-50:])
    # Convert decompressed data to numpy array
    decompressed_data = np.array(decompressed_data)

    # Save decompressed data to a new .wav file
    save_audio_data(output_file_path, decompressed_data)

    # print(f"Decompressed data saved to {output_file_path}")


@click.group()
def cli():
    pass


@cli.command()
def test():
    wav_file_path = "./data/ffb6837e-be2b-474f-bdd0-3c9cd631f39d.wav"
    lookup_file_path = "lookup_tables.pkl"
    bigram_model_path = "bigram_model.pkl"
    all_audio_data = load_all_audio_data("./data")

    # bigram_model = build_bigram_model(all_audio_data, lookup_file_path)
    # save_bigram_model(bigram_model, bigram_model_path)

    _compress(wav_file_path, lookup_file_path, bigram_model_path)

    brainwire_file_path = "./data/ffb6837e-be2b-474f-bdd0-3c9cd631f39d.wav.brainwire"
    output_file_path = "./data/ffb6837e-be2b-474f-bdd0-3c9cd631f39d.wav.copy"
    _decompress(
        brainwire_file_path, lookup_file_path, bigram_model_path, output_file_path
    )

    original_audio_data = load_audio_data(wav_file_path)
    decompressed_audio_data = load_audio_data(output_file_path)
    original_size = os.path.getsize(wav_file_path)
    compressed_size = os.path.getsize(brainwire_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}")

    if np.array_equal(original_audio_data, decompressed_audio_data):
        print("The original and decompressed files are equal.")
    else:
        print("The original and decompressed files are NOT equal.")


@cli.command()
@click.argument("input_file")
@click.argument("lookup_file")
@click.argument("bigram_model_file")
@click.argument("output_file")
def compress(input_file, lookup_file, bigram_model_file, output_file):
    _compress(input_file, lookup_file, bigram_model_file)


@cli.command()
@click.argument("input_file")
@click.argument("lookup_file")
@click.argument("bigram_model_file")
@click.argument("output_file")
def decompress(input_file, lookup_file, bigram_model_file, output_file):
    _decompress(input_file, lookup_file, bigram_model_file, output_file)


if __name__ == "__main__":
    cli()
