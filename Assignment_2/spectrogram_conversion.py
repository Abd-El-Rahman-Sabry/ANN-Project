import glob
import os

import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
from PIL import Image

audio_file = 'sample.wav'

def convert_to_img(sample ,name, saveAt=''):
    y, sr = librosa.load(sample)

    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert power to decibels
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(6, 5) ,)
    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # Save the spectrogram as a PNG file
    plt.savefig(os.path.join(saveAt,name + '.jpg'))

# Show the spectrogram



if __name__ == '__main__':


    # Train
    for i in range(10):
        files = glob.glob(f'S2S/Train/{i}/*.jpg')
        for idx , file in enumerate(files):
            img = Image.open(file)
            img = img.resize((32,32))

            img.save(file)
            print(f" current Number {i} in the {idx} image")

    # Test
    for i in range(10):
        files = glob.glob(f'S2S/Reduced Testing data/{i}/*.jpg')
        for idx, file in enumerate(files):
            img = Image.open(file)
            img = img.resize([32,32])
            img.save(file)
            print(f" current Number {i} in the {idx} image")

    
