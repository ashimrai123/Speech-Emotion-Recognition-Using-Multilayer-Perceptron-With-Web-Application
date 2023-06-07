import os
import numpy as np
import librosa
import random
import soundfile as sf


input_directory = "C:/Users/ashim/Desktop/SERProject/TESS"
output_directory = "C:/Users/ashim/Desktop/SERProject/TESS10"

emotions = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

def augment_data(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".wav"):
                try:
                    # Load audio file
                    audio, sr = librosa.load(file_path, sr=None)

                    # Randomly change pitch
                    pitch_shift = random.uniform(-1, 1)
                    audio_pitch_shift = librosa.effects.pitch_shift(audio, sr, n_steps=pitch_shift)

                    # Randomly change speed
                    speed_change = random.uniform(0.8, 1.2)
                    audio_speed_change = librosa.effects.time_stretch(audio_pitch_shift, speed_change)

                    # Randomly change volume
                    volume_change = random.uniform(0.5, 1.5)
                    audio_volume_change = audio_speed_change * volume_change

                    # Add noise to the audio
                    noise_level = random.uniform(0.0, 0.005)  # reduce noise level
                    noise = np.random.normal(0, noise_level, len(audio_volume_change))
                    audio_with_noise = audio_volume_change + noise

                    # Get emotion label from parent directory name
                    emotion = os.path.basename(os.path.dirname(file_path))

                    # Create output directory if it doesn't exist
                    output_emotion_directory = os.path.join(output_directory, emotion)
                    if not os.path.exists(output_emotion_directory):
                        os.makedirs(output_emotion_directory)

                    # Save augmented audio file
                    output_file_path = os.path.join(output_emotion_directory, f"{file[:-4]}_augmented10.wav")
                    sf.write(output_file_path, audio_with_noise, sr)
                except:
                    print(f"Error occurred while processing file: {file_path}")

augment_data(input_directory, output_directory)
