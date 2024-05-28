import soundfile as sf
import numpy as np
import pickle
import glob
from ReadAudioFiles import load_all_audio_data, load_audio_data


def create_lookup_tables(data: np.ndarray):
    unique_values, counts = np.unique(data, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_values = unique_values[sorted_indices]

    val_to_idx = {}
    val_to_idx_small = {}
    idx_to_val = {}
    idx_to_val_small = {}

    for idx, value in enumerate(sorted_values):
        val_to_idx[value] = idx
        idx_to_val[idx] = value
        if idx < 128:
            val_to_idx_small[value] = idx
            idx_to_val_small[idx] = value

    return val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small


def save_lookup_tables(
    val_to_idx: dict,
    val_to_idx_small: dict,
    idx_to_val: dict,
    idx_to_val_small: dict,
    file_path: str,
):
    with open(file_path, "wb") as f:
        pickle.dump((val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small), f)


def load_lookup_tables(file_path: str):
    with open(file_path, "rb") as f:
        val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small = pickle.load(f)
    return val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small


def value_to_binary_string(value: int, val_to_idx: dict, val_to_idx_small: dict) -> str:
    if value in val_to_idx_small:
        index = val_to_idx_small[value]
        return f"0{index:07b}"  # 0 followed by 7-bit binary
    elif value in val_to_idx:
        index = val_to_idx[value]
        return f"1{index:010b}"  # 1 followed by 10-bit binary
    else:
        raise ValueError("Value not found in lookup tables")


def tokenize(data: np.ndarray, val_to_idx: dict, val_to_idx_small: dict) -> list:
    return [
        value_to_binary_string(value, val_to_idx, val_to_idx_small) for value in data
    ]


# Example usage
if __name__ == "__main__":
    # file_path = "path_to_audio_file.wav"
    lookup_file_path = "lookup_tables.pkl"

    # Load audio data
    audio_data = load_all_audio_data("./data/")

    # Create lookup tables
    val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small = create_lookup_tables(
        audio_data
    )

    # Save lookup tables to disk
    save_lookup_tables(
        val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small, lookup_file_path
    )

    # Load lookup tables from disk
    val_to_idx, val_to_idx_small, idx_to_val, idx_to_val_small = load_lookup_tables(
        lookup_file_path
    )

    # Tokenize data
    # load first audio file and tokenize it
    audio_files = glob.glob("./data/*.wav")
    audio_data = load_audio_data(audio_files[0])
    tokenized_data = tokenize(audio_data, val_to_idx, val_to_idx_small)
    # print(tokenized_data)
