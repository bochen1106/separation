

#import sys
#sys.path.append('/home/lurui_i/audio-visual/codes/deep_cluster_keras')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import h5py
import numpy as np
import pickle
import scipy.io as sio
import os
import os.path as osp
import librosa

from sklearn.cluster import KMeans
import scipy.signal
import util
import glob
from util.config import Config


#%%
if __name__ == "__main__":
    
    
    config = Config('../model/config_03.json')
    fs = config.get("fs")
    
    data_type = "test"
    instr_mix = config.get("instr_mix")
    path_feat = config.get("path_feat")
    path_model = config.get("path_model")
    path_result = config.get("path_result")
    
    path_h5_aggr = os.path.join(path_feat, data_type)
    path_feat = os.path.join(path_feat, data_type, instr_mix)
    path_result = os.path.join(path_result, os.path.basename(path_model))
    
    path_result = path_result + "_ibm"
    
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    """ Load the data mean and std
    """
    file_h5 = os.path.join(path_h5_aggr, instr_mix) + ".h5"
    f = h5py.File(file_h5)
    data_mean = np.array( f['global_mean'] )
    data_std = np.array( f['global_std'] )
    f.close()
    
    
    filenames = glob.glob(path_feat + "/*.h5")

    for i in range(len(filenames)):
#    for i in range(1):
    
        filename = filenames[i]
        name = os.path.basename(filename).split(".")[0]
    
        f = h5py.File(filename)
        X_mag = np.array(f['X_mag'])
        X_pha = np.array(f['X_pha'])
        Y = np.asarray(np.swapaxes(f['Y'][...], 0, 2), dtype=np.int)  # T x d x 2
        f.close()
    


        X_mag_norm = (np.swapaxes(X_mag[...], 0, 1)  - data_mean) / data_std
        pred = Y
        
        for j in range(2):
            mask = pred.T[j,:,:]
            wav = util.restore_waveform(config, X_mag, X_pha, mask)
            filename_out = os.path.join(path_result, name) + "_" + str(j+1) + ".wav"
            librosa.output.write_wav(filename_out, wav, fs)
        
