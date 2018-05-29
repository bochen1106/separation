

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
import scipy
from util.config import Config


#%%
def get_mask_total(model, data, label):
    """ Get the mask of an audio by single run.
        We only use 'label' to mask out silent parts, this process has not
        used any clean-data information.

    :param audio_model : the trained model
    :param data        : T x input_size
    :param label       : T x input_size x num_spk
    :return: T x input_size x 2
    """
    data = np.asarray(data[None, ...], dtype=np.float32)

    # V: 1 x T x input_size x D
    V = model.predict_on_batch(x=[data])

    # Cluster part
    T = V.shape[1]
    input_size = V.shape[2]
    D = V.shape[-1]

    V = np.reshape(V, newshape=(-1, D))

    kmean = KMeans(n_clusters=2, random_state=0).fit(V)

    # 'pred_label_total' : T x input_size x num_spk
    pred_label_total = np.concatenate(((1 - kmean.labels_)[:, None], kmean.labels_[:, None]), axis=1)
    pred_label_total = np.reshape(pred_label_total, newshape=(T, input_size, -1))

    return pred_label_total * np.sum(label, axis=-1)[..., None]
#%%
def switch_mask(Y, pred):
    
    n_frame = Y.shape[0]
    for fnum in range(n_frame):
        y1 = Y[fnum,:,0].copy()
        y2 = Y[fnum,:,1].copy()
        p1 = pred[fnum,:,0].copy()
        p2 = pred[fnum,:,1].copy()
        m = np.zeros((2,2))
        m[0,0] = np.sum((y1-p1)**2)
        m[0,1] = np.sum((y1-p2)**2)
        m[1,0] = np.sum((y2-p1)**2)
        m[1,1] = np.sum((y2-p2)**2)
        if m[0,0]*m[1,1] > m[0,1]*m[1,0]:
            pred[fnum,:,0] = p2.copy()
            pred[fnum,:,1] = p1.copy()
    return pred
#%%
def switch_mask_new(Y, pred, size_med=0):
    
    mask = np.sum(Y, axis=-1)[..., None]
    p_pos = np.sum(np.sum(pred * mask * Y, axis=-1), axis=-1)
    p_neg = np.sum(np.sum((1 - pred) * mask * Y, axis=-1), axis=-1)
    t_order = np.asarray(p_pos < p_neg, dtype=np.float32)
    
    if size_med > 0:
        t_order = scipy.signal.medfilt(t_order, 3)
    
    pred_acc = np.copy(pred)
    pred_acc[np.asarray(t_order, dtype=np.bool)] = 1 - pred_acc[np.asarray(t_order, dtype=np.bool)]
    pred_acc = pred_acc * mask
    
    
    
    return pred_acc
    
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
    
    path_result = path_result + "_adjust_med55"
    
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    """ Load the model
    """
    from my_model import MyModel
    m = MyModel(config)
    m.build_model( path_weights=osp.join(path_model, 'model.best.h5') )
    model = m.my_model
    
    """ Load the data mean and std
    """
    file_h5 = os.path.join(path_h5_aggr, instr_mix) + ".h5"
    f = h5py.File(file_h5)
    data_mean = np.array( f['global_mean'] )
    data_std = np.array( f['global_std'] )
    f.close()
    
    
    filenames = glob.glob(path_feat + "/*.h5")

    for i in range(len(filenames)):
#    for i in range(5,6):
    
        filename = filenames[i]
        print filename
        name = os.path.basename(filename).split(".")[0]
    
        f = h5py.File(filename)
        X_mag = np.array(f['X_mag'])
        np.savetxt('../result/tmp/X_mag.txt', X_mag, fmt="%d")
        X_pha = np.array(f['X_pha'])
        Y = np.asarray(np.swapaxes(f['Y'][...], 0, 2), dtype=np.int)  # T x d x 2
        f.close()
    


        X_mag_norm = (np.swapaxes(X_mag[...], 0, 1)  - data_mean) / data_std
        pred = get_mask_total(model=model, data=X_mag_norm, label=Y)
        
        np.savetxt('../result/tmp/pred1.txt', pred.T[0,:,:], fmt="%d")
        np.savetxt('../result/tmp/pred2.txt', pred.T[1,:,:], fmt="%d")
        np.savetxt('../result/tmp/y1.txt', Y.T[0,:,:], fmt="%d")
        np.savetxt('../result/tmp/y2.txt', Y.T[1,:,:], fmt="%d")
        
        #%%
        pred = switch_mask_new(Y, pred, size_med = 55)
        np.savetxt('../result/tmp/pred_aj_1.txt', pred.T[0,:,:], fmt="%d")
        np.savetxt('../result/tmp/pred_aj_2.txt', pred.T[1,:,:], fmt="%d")
        
        for j in range(2):
            mask = pred.T[j,:,:]
            wav = util.restore_waveform(config, X_mag, X_pha, mask)
            filename_out = os.path.join(path_result, name) + "_" + str(j+1) + ".wav"
            librosa.output.write_wav(filename_out, wav, fs)
        
