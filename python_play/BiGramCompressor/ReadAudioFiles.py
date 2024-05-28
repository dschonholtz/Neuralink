import glob
import numpy as np
import soundfile as sf


def load_all_audio_data(data_dir: str) -> np.ndarray:
    # Load all of the .wav files in the dir
    audio_data = []
    for file_name in glob.glob(data_dir + "/*.wav"):
        audio_data.append(load_audio_data(file_name))
    return np.concatenate(audio_data)


def load_audio_data(file_path: str) -> np.ndarray:
    audio_data, _ = sf.read(
        file_path,
        dtype="int16",
    )
    return audio_data


def save_audio_data(file_path: str, audio_data: np.ndarray):
    sf.write(file_path, audio_data, 19531, format="WAV")
