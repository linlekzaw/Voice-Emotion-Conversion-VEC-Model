import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt

DATASET_PATH = "CREMA-D/AudioWAV"
SAVE_PATH = "CREMA-D/Spectrograms"

# Create output directory
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def process_audio(file_path):
    """Converts a .wav file to a Mel-Spectrogram."""
    y, sr = librosa.load(file_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Process all files
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        file_path = os.path.join(DATASET_PATH, file)
        spec = process_audio(file_path)

        # Save as NumPy array
        np.save(os.path.join(SAVE_PATH, file.replace(".wav", ".npy")), spec)

print("✅ Audio Preprocessing Complete! Spectrograms Saved.")
