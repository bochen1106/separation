#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:36:25 2018

@author: bochen
"""
import sys
import numpy as np
import librosa
import h5py
import matplotlib.pyplot as plt
import glob
import os

import util
import cPickle as pickle
from util.config import Config

config = Config('../config.json')


instr_1 = "vn"
instr_2 = "cl"
data_type = "test"
instr_mix = instr_1 +"-"+ instr_2 
path_h5 = os.path.join("../data/feat/", data_type, instr_mix)
path_mix = os.path.join("data/segments/mix/", data_type, instr_mix)
path_phase = os.path.join("data/segments/phase/", data_type, instr_mix)

filenames = glob.glob(path_h5 + "/*.h5")

#%%
i = 0

filename = filenames[i]
name = os.path.basename(filename)
name = os.path.splitext(name)[0]


f = h5py.File(filename)
X = np.array(f['X_mag'])  # F x T
Y = np.array(f['Y'])  # 3 x F x T
X_pha = np.array(f['X_pha'])
f.close()

Y = Y[0,:,:]
#%%
wav = util.restore_waveform(config, X, X_pha, Y)
librosa.output.write_wav("test.wav", wav, 8000)



#%%
import librosa
import matplotlib.pyplot as plt

import util
from util.config import Config
config = Config('../config.json')
filename = "../data/audio/mix/valid/vn-cl/vn0001-cl0015.wav"
filename1 = "../data/audio/single/valid/vn/vn0001@trk92&dur0005-0010.wav"
filename2 = "../data/audio/single/valid/cl/cl0015@trk82&dur0090-0095.wav"

data_mix, fs = librosa.load(filename, sr=8000)
data1, fs = librosa.load(filename1, sr=8000)
data2, fs = librosa.load(filename2, sr=8000)
X_mag0, Y0, X_pha0 = util.extract_feat(config, data_mix, data1, data2)


wav = util.restore_waveform(config, X_mag0, X_pha0, Y0[0,:,:])
librosa.output.write_wav("test.wav", wav, 8000)

#%%
import numpy as np
import h5py
f = h5py.File("../data/feat/valid/vn-cl/vn0001-cl0015.h5")
X_mag = np.array(f['X_mag'])
X_pha = np.array(f['X_pha'])
f.close()

#%%

f = h5py.File("test.h5")
f['X_mag0'] = X_mag0
f['X_pha0'] = X_pha0
f['Y0'] = Y0
f.close()

#%%
f = h5py.File("test.h5")
X_mag1 = np.array(f["X_mag0"])
X_pha1 = np.array(f["X_pha0"])
            
            

