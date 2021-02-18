# Import library
import IPython.display as ipd
import glob
from scipy.io import wavfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import struct
from scipy.io import wavfile as wav
import os   
from datetime import datetime 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint 

# Librosa Feature
def extract_features(file_name):
    nmfcc=40
    try:
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        audio, _ = librosa.effects.trim(y)
        hop_length = 1024
        n_fft = 2048
        n_mels = 128
        S = librosa.feature.mfcc(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=nmfcc, fmin=0)
        S_DB = librosa.power_to_db(S, ref=np.max)
        mfccsscaled = np.mean(S_DB.T,axis=0)   
    except Exception:
        print("Error encountered while parsing file: ", file_name)
        return None 
    return mfccsscaled

## Get data
#path="IN1_Audio_Samples/"
#files=os.listdir(path)
#for filename in glob.glob(os.path.join(path, '*.wav')):
# samplerate,data=wavfile.read(filename)
#data=librosa.core.load(file_path, sr=11025)

# Audio filepath
fulldatasetpath = 'IN7_Audio_Samples/'

# Label filepath
metadata = pd.read_csv('Audio_Y.csv')
features = []

# Check for audio in labels and get features
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath),str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_features(file_name)
    features.append([data, class_label])
# Convert to pandas dataframe
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
print(featuresdf)

#Converting Train Data
# Getting size of data
X = np.array(featuresdf.feature.tolist())
print (X.shape)

# Getting label size
y = np.array(featuresdf.class_label.tolist())
print(y.shape)

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
print(yy)

# Split the dataset 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2,shuffle=False)
num_labels = yy.shape[1]
print(num_labels)
filter_size = 2

# Construct model 
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#Setting parameters
num_epochs = 200
num_batch_size = 50

#Timenote
start = datetime.now()
duration = datetime.now() - start
print("Training completed: ", duration)

#Fitting
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# Basic Training
score = model.evaluate(x_train, y_train, verbose=0)
print('train accuracy: {}'.format(score))
# experiment.log_metric("train_acc", score)

# Basic Testing
score = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy: {}'.format(score))
# experiment.log_metric("val_acc", score)