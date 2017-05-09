"""
this code is transfrom the stft feature map to a image with shape :[244,244,3]
@pzp creat in 5/8/2017

"""

import glob
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import preprocessing
from skimage import transform,io

#matplotlib inline
try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle
##the superparameter of WAV load ang stft
win_len = 2048                  #window length of per sample
hop_len = win_len/2                 #overlap = 50%
sr = 50000                      #sample rate
parent_dir = '/home/pzp/data/Saarbruecken Voice Database/2000data'   #the parent direction
tr_sub_dirs = ['Pathology/Train', 'Normal/Train']
ts_sub_dirs = ['Pathology/Test', 'Normal/Test']

##load the file
X, sample_rate = librosa.load('123.wav',sr)
X = X[len(X)-30000:len(X)+1]

##compute the stft,shape:[30,1025]
stft = np.abs(librosa.stft(X,n_fft = win_len,hop_length=hop_len,win_length=win_len)).T
pca = PCA(n_components=30)
pcastft = pca.fit_transform(stft)
# logstft = np.log(pcastft)
scalestft = preprocessing.minmax_scale(pcastft,[0, 255])
intstft = np.uint8(scalestft)
##transfrom to [1025,1025,3]
input_image = np.empty([30,30,3],np.uint8)
input_image[:,:,0] = intstft
input_image[:,:,1]=intstft
input_image[:,:,2]= intstft
io.imshow(input_image)
output = transform.resize(input_image,(224,224))
io.imshow(output)
a=1