import gradio as gr
import joblib
import numpy as np
import librosa
import tensorflow.keras as keras

# Load models
cnn_model = keras.models.load_model("MusicGenre_CNN.h5")
svm_model = joblib.load("MusicGenre_SVM.joblib")

# Genre dictionary
genre_dict = {0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock", 
              5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"}

def process_input(audio_file, track_duration=30):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512
    SAMPLES_PER_TRACK = SAMPLE_RATE * track_duration
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    mfccs = []
    for d in range(NUM_SEGMENTS):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfccs.append(mfcc.T)

    return np.array(mfccs)

def predict_genre_with_cnn(audio_file):
    mfcc = process_input(audio_file)  # shape: (10, 130, 13)
    X_to_predict = mfcc[0][np.newaxis, ..., np.newaxis]  # shape: (1, 130, 13, 1)
    probs = cnn_model.predict(X_to_predict)[0]
    predicted_index = np.argmax(probs)
    return genre_dict[int(predicted_index)]

def predict_genre_with_svm(audio_file):
    mfcc = process_input(audio_file)
    mfcc_flat = mfcc.flatten().reshape(1, -1)
    expected_shape = svm_model.n_features_in_
    
    if mfcc_flat.shape[1] < expected_shape:
        pad_width = expected_shape - mfcc_flat.shape[1]
        mfcc_flat = np.pad(mfcc_flat, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc_flat.shape[1] > expected_shape:
        mfcc_flat = mfcc_flat[:, :expected_shape]
    
    pred = svm_model.predict(mfcc_flat)
    return genre_dict[int(pred[0])]

def classify_genre(audio_file, model_type):
    if model_type == "CNN":
        return predict_genre_with_cnn(audio_file)
    else:
        return predict_genre_with_svm(audio_file)

# Gradio interface
iface = gr.Interface(
    fn=classify_genre,
    inputs=[
        gr.Audio(type="filepath"),  # Remove source argument
        gr.Radio(["CNN", "SVM"], label="Select Model")
    ],
    outputs="text",
    title="Music Genre Classification",
    description="Upload an audio file and select a model (CNN or SVM) to classify the music genre."
)


iface.launch()