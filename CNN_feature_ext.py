'''
This code is combine prevlous code to do the freature extraction
i cut the sample to the same length 30000
win_len = 1024,over_lap = 25%,per sample obtain 40 frames and per frame obtain 187 dims features 
'''
"""
Created on Mon Nov 14 13:43:51 2016

@author: panzhanpeng
"""
import glob
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA 
from sklearn import preprocessing
#matplotlib inline
try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle

win_len = 2048                  #window length of per sample
hop_len = win_len/2                 #overlap = 50%
sr = 50000                      #sample rate
def vstack(tup):
    return _nx.concatenate([np.atleast_3d(_m) for _m in tup], 0)
    
def extract_feature(file_name,sr,win_len,hop_len):
    X, sample_rate = librosa.load(file_name,sr)   #load the wav file
    X = X[0:30000]                    #cut the X to the same length(30000)
    stft = np.abs(librosa.stft(X,n_fft = win_len,hop_length=hop_len,win_length=win_len)).T
#    mel_s = librosa.logamplitude(librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=win_len,hop_length=hop_len))
#    mfccs = librosa.feature.mfcc(S=mel_s, sr=sample_rate, n_mfcc=40).T
#    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T
#    mel = librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=win_len,hop_length=hop_len).T
#    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T
  #  tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T
    
    return stft

def parse_audio_files(parent_dir,sub_dirs,sr,win_len,hop_len,file_ext='*.wav'):
    features, labels = np.empty((0,30*1024)), np.empty(0)
    #n_num = 0
    #features, labels = np.empty((0,100000)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            stft = extract_feature(fn,sr,win_len,hop_len)
           # ext_features = np.hstack([mfccs,chroma,mel,contrast])
#            ext_features = preprocessing.scale(stft,axis=1)     #normalization
#            pca=PCA()
#            pca_tr=pca.fit_transform(ext_features)
         #   print len(ext_features)
            stft_1024 = np.delete(stft,1024,axis=1)
            ext_features = np.reshape(stft_1024,[1,30*1024])
            
            #ext_features = ext_features[np.newaxis,:,:]
            features = np.vstack([features,ext_features])
  #          labels = np.append(labels, fn.split('/')[2].split('-')[1])
            labels = np.append(labels,label)
            #n_num = n_num +1
  #          print labels
    #featuress = np.reshape(features,[n_num,59,193])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = '/home/pzp/data/Saarbruecken Voice Database/2000data'
tr_sub_dirs = ['Pathology/Train','Normal/Train']
ts_sub_dirs = ['Pathology/Test','Normal/Test']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs,sr,win_len,hop_len,file_ext='*.wav')
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs,sr,win_len,hop_len,file_ext='*.wav')
tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)
#pca_tr = sklearn.decomposition.PCA()
#tr_features = np.log10(tr_features)
#ts_features = np.log10(ts_features)
tr_features = preprocessing.minmax_scale(tr_features)
ts_features = preprocessing.minmax_scale(ts_features)
file_temp1 = open('CNN_stft_2000_norm', 'w')
pickle.dump(tr_features, file_temp1)
pickle.dump(tr_labels, file_temp1)
pickle.dump(ts_features, file_temp1)
pickle.dump(ts_labels, file_temp1)
file_temp1.close()
