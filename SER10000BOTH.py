import os
Root = "C:/Users/ashim/Desktop/SERProject"
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
        
     # Load augmented RAVDESS dataset
    for emotion in observed_emotions:
        for file in glob.glob(f"C:/Users/ashim/Desktop/SERProject/Speech2/{emotion}/*.wav"):
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
        
    # Load augmented TESS dataset
    for emotion in observed_emotions:
        for file in glob.glob(f"C:/Users/ashim/Desktop/SERProject/TESS2/{emotion}/*.wav"):
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
x_train, x_test, y_train, y_test = load_data(test_size=0.20)

# Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier with modified parameters
model = MLPClassifier(alpha=0.01, batch_size=32, hidden_layer_sizes=(512,), learning_rate='constant',learning_rate_init=0.01, max_iter=300)

# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy and f1-score of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
f1score = f1_score(y_test, y_pred, average=None)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head(20))

# Print the accuracy and f1-score
print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1-score:", f1score)

# Save the model in pkl format
filename = 'C:/Users/ashim/Desktop/SERProject/saved_model10000.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
    
# Save the StandardScaler object in pkl format
filename = 'C:/Users/ashim/Desktop/SERProject/scaler.pkl'
with open(filename, 'wb') as file:
    pickle.dump(scaler, file)

    
    
#Confusion matrix to save in a file 
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Write the confusion matrix and accuracy to a file
with open("C:/Users/ashim/Desktop/SERProject/results.txt", "w") as f:
    f.write("Confusion matrix:\n")
    f.write(np.array2string(conf_matrix, separator=",") + "\n")
    f.write("Accuracy: " + str(accuracy))


# Compute confusion matrix to display in plot 
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(observed_emotions))
plt.xticks(tick_marks, observed_emotions, rotation=45)
plt.yticks(tick_marks, observed_emotions)

# Format the plot
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
