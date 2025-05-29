import os
import numpy as np
from Feature_Extract import extract_features  # Ensure correct import

# Define paths
dataset_path = "dataset"
feature_output_folder = "Feature_Extracted"
spectrogram_output_folder = "Spectrograms"  # Use existing folder

os.makedirs(feature_output_folder, exist_ok=True)  # Create Feature_Extracted if missing
os.makedirs(spectrogram_output_folder, exist_ok=True)  # Ensure Spectrograms folder exists

labels = {"child": 0, "qari": 1}

X = []
y = []

# Loop through dataset folders
for label_name, label in labels.items():
    folder = os.path.join(dataset_path, label_name)
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder, filename)
            features = extract_features(file_path, save_spectrogram=True, spectrogram_dir=spectrogram_output_folder)
            if features is not None:
                X.append(features)
                y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save extracted features and labels
np.save(os.path.join(feature_output_folder, "features.npy"), X)
np.save(os.path.join(feature_output_folder, "labels.npy"), y)

print("Dataset created and saved in 'Feature_Extracted' folder!")
print(f"Number of samples: {X.shape[0]}")
print(f"Feature vector size: {X.shape[1]}")
print(f"Spectrograms saved in {spectrogram_output_folder}/child and {spectrogram_output_folder}/qari")
