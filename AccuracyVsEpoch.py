import os
Root = "C:/Users/ashim/Desktop/SERProject/Speech"
os.chdir(Root)


import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import itertools

# Define the scaler object
scaler = StandardScaler()

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
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


# Observed emotions in both datasets
observed_emotions = set(list(emotions.values()) + list(tess_emotions.values()))

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    
    # Load RAVDESS dataset
    for file in glob.glob("C:/Users/ashim/Desktop/SERProject/Speech/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    
    # Load TESS dataset
    for file in glob.glob("C:/Users/ashim/Desktop/SERProject/TESS/*/*.wav"):
        file_name = os.path.basename(file)
        emotion = tess_emotions[file_name.split("_")[2].split(".")[0]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        


    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), y, test_size=test_size, random_state=9)

    # Normalize the features using StandardScaler
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# Initialize the Multi Layer Perceptron Classifier with modified parameters
model = MLPClassifier(alpha=0.01, batch_size=16, hidden_layer_sizes=(512,), learning_rate='adaptive', learning_rate_init=0.01, max_iter=1, warm_start=True)

# Initialize empty lists to store training and validation accuracies for each epoch
train_acc_list = []
val_acc_list = []

# Set the number of epochs
num_epochs = 100

# Train the model for each epoch and store training and validation accuracies
for epoch in range(num_epochs):
    model.partial_fit(x_train, y_train, classes=np.unique(y_train))
    train_acc = accuracy_score(y_train, model.predict(x_train))
    val_acc = accuracy_score(y_test, model.predict(x_test))
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    print(f"Epoch {epoch+1}: Train Accuracy = {train_acc:.3f}, Validation Accuracy = {val_acc:.3f}")

# Plot the training and validation accuracies vs epochs
plt.plot(range(1,num_epochs+1), train_acc_list, label="Train Accuracy")
plt.plot(range(1,num_epochs+1), val_acc_list, label="Validation Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
