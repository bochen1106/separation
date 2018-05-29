#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:28:00 2018

@author: bochen
"""

import librosa
import numpy as np

filename = "../data/audio/mix/valid/vn-cl/vn0001-cl0015.wav"

data, fs = librosa.load(filename, sr=8000)

stft = librosa.core.stft(y=data, n_fft=256,
                              hop_length=int(np.floor(0.008 * fs)),
                              win_length=int(np.floor(0.032 * fs)),
                              window=np.sqrt(np.hanning(256)), center='True', pad_mode='constant')
X_mag, X_pha = librosa.core.magphase(stft)

X_mag = librosa.core.power_to_db(S=X_mag, ref=1, amin=1e-10, top_db=None)
#%%