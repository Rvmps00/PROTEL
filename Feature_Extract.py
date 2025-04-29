import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def extract_features(file_path, sr=22050, n_mfcc=13, save_spectrogram=False, spectrogram_dir="Spectrograms"):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_centroid_mean = np.mean(spec_centroid)
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        
        features = np.hstack((mfccs_mean, mfccs_std, chroma_mean, spec_centroid_mean, zcr_mean))

        # Save spectrogram if required
        if save_spectrogram:
            save_spectrogram_image(file_path, audio, sr, spectrogram_dir)

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_spectrogram_image(file_path, audio, sr, base_output_dir):
    """ Generate and save spectrogram image in a structured folder """
    try:
        # Compute STFT
        stft = librosa.stft(audio)
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # Extract category (child/qari) from file path
        category = os.path.basename(os.path.dirname(file_path))
        output_dir = os.path.join(base_output_dir, category)

        # Ensure save directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate spectrogram and save it
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title(f"Spectrogram of {os.path.basename(file_path)}")

        output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.wav', '.png'))
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Error saving spectrogram for {file_path}: {e}")
