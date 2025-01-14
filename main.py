import librosa
import numpy as np
from tensorflow import keras
import math # Import the math module

# ... (rest of your code) ...
# Load the saved model
model = keras.models.load_model("MusicGenre_CNN_79.73.h5")

# Function to preprocess the audio file
def process_input(audio_file, track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    # Extract MFCCs for the first segment
    start = samples_per_segment * 0  
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)

    mfcc = mfcc.T

    return mfcc

# Dictionary to map predicted index to genre
genre_dict = {
    0: "disco",
    1: "pop",
    2: "classical",
    3: "metal",
    4: "rock",
    5: "blues",
    6: "hiphop",
    7: "reggae",
    8: "country",
    9: "jazz",
}

# Path to your audio file
audio_file_path = ""  

# Preprocess the audio
new_input_mfcc = process_input(audio_file_path, 30)

# Reshape the input for the model
X_to_predict = new_input_mfcc[np.newaxis, ..., np.newaxis]

# Make the prediction
prediction = model.predict(X_to_predict)

# Get the predicted genre
predicted_index = np.argmax(prediction, axis=1)
predicted_genre = genre_dict[int(predicted_index)]

print("Predicted Genre:", predicted_genre)