import os
import streamlit as st
import librosa
import soundfile
import numpy as np
import pickle
import sounddevice as sd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

# Load the trained MLP model from a pickle file
model_file = "C:/Users/ashim/Desktop/SERProject/saved_model10000.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Load the scaler from a pickle file
scaler_file = "C:/Users/ashim/Desktop/SERProject/scaler.pkl"
with open(scaler_file, "rb") as file:
    scaler = pickle.load(file)

# Emotions in the RAVDESS dataset
emotions = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

# Emotions in the TESS dataset
tess_emotions = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fearful',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'surprised',
    'sad': 'sad'
}



# Observed emotions in three datasets
observed_emotions = set(list(emotions.values()) + list(tess_emotions.values()))


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(
                librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, mel))
            
    
    # Preprocess the features using StandardScaler
    scaled_features = scaler.transform([result])
    return scaled_features

# Function to predict the emotion of an audio file
def predict_emotion(file_name):
    # Extract features from the audio file
    features = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
    
    # Check if the audio is silent
    if np.isnan(features).any():
        st.write("<h1 style='text-align:center; color:red;'>The recorded audio is silent</h1>", unsafe_allow_html=True)
        return
    
    # Reshape the features to match the expected input shape of the model
    features = features.reshape(1, -1)

    # Predict the emotion of the audio file
    emotion = model.predict(features)[0]

    # Load the audio file and extract the mfcc, mel and chroma features
    y, sr = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    # Plot the mfcc, mel and chroma graphs
    fig, axs = plt.subplots(3, 1, figsize=(10, 17), gridspec_kw={'height_ratios': [1, 1, 1]})

    axs[0].set(title='MFCC')
    librosa.display.specshow(mfccs, x_axis='time', ax=axs[0])
    axs[0].set(ylabel='MFCC')

    axs[1].set(title='Mel Spectrogram')
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time', ax=axs[1])
    axs[1].set(ylabel='Mel frequency')

    axs[2].set(title='Chromagram')
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=axs[2])
    axs[2].set(ylabel='Chromagram')

    # Display the predicted emotion
    st.write(f"<h1 style='text-align:center; color:black;'>The predicted emotion is: {emotion}</h1>", unsafe_allow_html=True)

    # Display the mfcc, mel and chromagraphs
    st.pyplot(fig)


 
# Create a main function
def main():
    
    # Set page title and favicon
    st.set_page_config(page_title="Emotion Recognition", page_icon=":microphone:")

    # Set page header
    st.write("<h1 style='text-align:center; color:black; font-size:75px;'>Speech Emotion Recognition</h1>", unsafe_allow_html=True)
    
    # Add a sidebar with options
    st.sidebar.title("Features")
    option = st.sidebar.selectbox("Select an option", ("Upload audio file", "Record your audio"))

    # Record audio
    if option == "Record your audio":
        # Get the recording duration from the user
        recording_duration = st.sidebar.text_input("Recording duration in seconds", value="3")

        # Convert the recording duration to a float
        recording_duration = float(recording_duration)

        # Add a record button
        record_button = st.sidebar.button("Record audio", key="record_button")
        if record_button:
            st.write("<h4 style='color:black;'> Recording your audio </h4>", unsafe_allow_html=True)
            recording = sd.rec(int(recording_duration * 44100), 44100, channels=1)
            sd.wait()

            # Save the recorded audio as a WAV file
            file_name = "recording.wav"
            soundfile.write(file_name, recording, 44100)

            # Display the audio player
            audio_player = st.audio(file_name)

            # Predict the emotion of the audio file
            emotion = predict_emotion(file_name)

        
    # Upload audio file
    elif option == "Upload audio file":
        # Get the audio file from the user
        audio_file = st.sidebar.file_uploader("Choose an audio file", type="wav")

        if audio_file is not None:
            file_name = audio_file.name

            # Save the audio file
            with open(file_name, "wb") as f:
                f.write(audio_file.getbuffer())

            # Display the audio player
            audio_player = st.audio(file_name)

            # Predict the emotion of the audio file
            emotion = predict_emotion(file_name)


    # Show instructions if no option is selected
    else:
        st.write("Select an option to begin")

if __name__ == "__main__":
    main()

#Background Image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""<style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover}}</style>""",unsafe_allow_html=True)
add_bg_from_local('comic-face.png')  

# Load the image to be added on top of the background image
footer_logo = Image.open('footer-logo.png')
container = st.container()
col1, col2, col3 = st.columns([1, 1.5, 1])
col2.image(footer_logo, use_column_width=True)
container.markdown("<p style='padding: 180px'></p>", unsafe_allow_html=True)
