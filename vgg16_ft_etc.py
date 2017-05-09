'''
This code is combine prevlous code to do the freature extraction
i cut the sample to the same length 30000
win_len = 1024,over_lap = 25%,per sample obtain 40 frames and per frame obtain 187 dims features 
'''
"""
Created on 5/8/2017

@author: panzhanpeng
"""
import glob
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from skimage import transform

# matplotlib inline
try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle

win_len = 2048  # window length of per sample
hop_len = win_len / 2  # overlap = 50%
sr = 50000  # sample rate
pca = PCA(n_components=30)

def extract_feature(file_name, sr, win_len, hop_len):
    input_image = np.empty([30, 30, 3], np.uint8)
    X, sample_rate = librosa.load(file_name, sr)  # load the wav file
    X = X[0:30000]  # cut the X to the same length(30000)
    stft = np.abs(librosa.stft(X, n_fft=win_len, hop_length=hop_len, win_length=win_len)).T
    pcastft = pca.fit_transform(stft)
    scalestft = preprocessing.minmax_scale(pcastft, [0, 255])
    intstft = np.uint8(scalestft)
    input_image[:, :, 0] = intstft
    input_image[:, :, 1] = intstft
    input_image[:, :, 2] = intstft
    output = transform.resize(input_image, (224, 224))
    return output


def parse_audio_files(parent_dir, sub_dirs, sr, win_len, hop_len, file_ext='*.wav'):
    labels,image =  np.empty(0,np.uint8), np.empty([0,224*224*3],np.uint8)
    i = 0
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):

            stft = extract_feature(fn, sr, win_len, hop_len)
            stft = np.reshape(stft, [1, 224*224*3])
            image = np.vstack([image, stft])
            i = i + 1
            labels = np.append(labels, label)
    output = np.reshape(image,[i,224,224,3])
    return np.array(output), np.array(labels, dtype=np.uint8)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


parent_dir = '/home/pzp/data/Saarbruecken Voice Database/2000data'
tr_sub_dirs = ['Pathology/Train', 'Normal/Train']
ts_sub_dirs = ['Pathology/Test', 'Normal/Test']
tr_features, tr_labels = parse_audio_files(parent_dir, tr_sub_dirs, sr, win_len, hop_len, file_ext='*.wav')
ts_features, ts_labels = parse_audio_files(parent_dir, ts_sub_dirs, sr, win_len, hop_len, file_ext='*.wav')
tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)
file_temp1 = open('vgg16_pac_feature', 'w')
pickle.dump(tr_features, file_temp1)
pickle.dump(tr_labels, file_temp1)
pickle.dump(ts_features, file_temp1)
pickle.dump(ts_labels, file_temp1)
file_temp1.close()
