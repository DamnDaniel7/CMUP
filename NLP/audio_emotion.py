import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#dataset from https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio

emotion_labels = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

focused_emotion_labels = ['happy','calm','neutral', 'disgust','angry']

def audio_features(file_title, mfcc, chroma, mel):
    with sf.SoundFile(file_title) as audio_recording:
        aud = audio_recording.read(dtype="float32")
        audio= lb.to_mono(aud)
        sample_rate = audio_recording.samplerate
        if chroma:
            stft=np.abs(lb.stft(audio,n_fft=8192))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=8192).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate,n_fft=8192).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate,n_fft=8192).T,axis=0)
            result=np.hstack((result, mel))
        return result

def loading_audio_data():
    x = []
    y = []
    for file in glob.glob("ravdess//Actor_*//*.wav"):
        file_path=os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]
        if emotion not in focused_emotion_labels:
             continue
        feature = audio_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=0)
    return final_dataset

X_train, X_test, y_train, y_test = loading_audio_data()

model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy of the Recognizer is: {:.1f}%".format(accuracy*100))

#saving the model 
if not os.path.isdir("model"): 
   os.mkdir("model") 
pickle.dump(model, open("model/mlp_classifier1.model", "wb"))