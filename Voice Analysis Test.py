import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Memuat file audio anak dan qari
audio_anak, sr = librosa.load('Hijaiyah.wav', sr=None)
audio_qari, _ = librosa.load('Hijaiyah.wav', sr=sr)  # pastikan sama sr

# Menghitung STFT
stft_anak = librosa.stft(audio_anak)
stft_qari = librosa.stft(audio_qari)

# Mengonversi ke dB
db_anak = librosa.amplitude_to_db(np.abs(stft_anak), ref=np.max)
db_qari = librosa.amplitude_to_db(np.abs(stft_qari), ref=np.max)

# Menampilkan spectrogram
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(db_anak, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title('Spectrogram Anak')

plt.subplot(2, 1, 2)
librosa.display.specshow(db_qari, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title('Spectrogram Qari')

plt.tight_layout()
plt.show()
