import math  # For mathematical calculations
import librosa  # For audio processing
import numpy as np  # For numerical operations
from tensorflow import keras  # For loading the trained model

# Load the pre-trained model for genre classification
model = keras.models.load_model("");

# Function to preprocess the audio file
def process_input(audio_file, track_duration=30):
    SAMPLE_RATE = 22050  # Sample rate for audio processing
    NUM_MFCC = 13  # Number of MFCCs to extract
    N_FTT = 2048  # Number of samples per FFT
    HOP_LENGTH = 512  # Number of samples between successive frames
    TRACK_DURATION = track_duration  # Duration of the audio to process (in seconds)
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION  # Total samples in the track
    NUM_SEGMENTS = 10  # Number of segments to split the audio into

    # Calculate samples and MFCC vectors per segment
    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    # Load the audio file and resample it
    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    # Extract MFCCs for the first segment
    start = samples_per_segment * 0  # Start index of the first segment
    finish = start + samples_per_segment  # End index of the first segment
    mfcc = librosa.feature.mfcc(
        y=signal[start:finish],
        sr=sample_rate,
        n_mfcc=NUM_MFCC,
        n_fft=N_FTT,
        hop_length=HOP_LENGTH,
    )

    # Transpose the MFCC array to make it suitable for the model
    mfcc = mfcc.T
    return mfcc

# Dictionary to map the model's predicted index to a genre
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

# Function to predict the genre of the audio file
def predict_genre(audio_file):
    # Preprocess the uploaded audio file
    new_input_mfcc = process_input(audio_file, 30)
    # Reshape the input to match the model's expected input shape
    X_to_predict = new_input_mfcc[np.newaxis, ..., np.newaxis]
    # Make the prediction using the loaded model
    prediction = model.predict(X_to_predict)
    # Get the genre index with the highest probability
    predicted_index = np.argmax(prediction, axis=1)
    # Map the predicted index to the genre name
    predicted_genre = genre_dict[int(predicted_index)]
    return f"Predicted Genre: {predicted_genre}"

#interface
import gradio as gr  # Import Gradio for building the UI
# Create the Gradio interface
interface = gr.Interface(
    fn=predict_genre,  # Function to be executed when user uploads a file
    inputs=gr.Audio(type="filepath"),  # Accept uploaded audio files
    outputs="text",  # Display the prediction as text
    title="Music Genre Classifier",  # Title of the app
    description="Upload an audio file to predict its genre.",  # Short description
)

# Launch the Gradio app
interface.launch()
