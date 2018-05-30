#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:18:27 2018

@author: bochen
"""
import os
import numpy as np
import librosa
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anm

#%%
def extract_feat(config, y_mix, y1, y2):
    
    fs = config.get("fs", log=False)
    frame_len = config.get("frame_len", log=False)
    frame_hop = config.get("frame_hop", log=False)
    n_fft = config.get("n_fft", log=False)
    active_threshold = config.get("active_threshold", log=False)
    
    stft1 = librosa.core.stft(y=y1, n_fft=n_fft,
                              hop_length=int(np.floor(frame_hop * fs)),
                              win_length=int(np.floor(frame_len * fs)),
                              window=np.sqrt(np.hanning(n_fft)), center='True', pad_mode='constant')

    stft2 = librosa.core.stft(y=y2, n_fft=n_fft,
                              hop_length=int(np.floor(frame_hop * fs)),
                              win_length=int(np.floor(frame_len * fs)),
                              window=np.sqrt(np.hanning(n_fft)), center='True', pad_mode='constant')

    stft_mix = librosa.core.stft(y=y_mix, n_fft=n_fft,
                                 hop_length=int(np.floor(frame_hop * fs)),
                                 win_length=int(np.floor(frame_len * fs)),
                                 window=np.sqrt(np.hanning(n_fft)), center='True', pad_mode='constant')

    # For reconstruction
    mag_mix, phase_mix = librosa.core.magphase(stft_mix)
    mag1, phase1 = librosa.core.magphase(stft1)
    mag2, phase2 = librosa.core.magphase(stft2)

    """ Step 2: Compute the log magnitude
    """
    # We don't use 'top_db' here since we will
    # mask out the silent parts later
    log_stft1 = librosa.core.power_to_db(S=mag1, ref=1, amin=1e-10, top_db=None)
    log_stft2 = librosa.core.power_to_db(S=mag2, ref=1, amin=1e-10, top_db=None)
    log_stft_mix = librosa.core.power_to_db(S=mag_mix, ref=1, amin=1e-10, top_db=None)

    """ Step 3: Get the mask to eliminate silent parts
    """
    log_stft_mix_active = (log_stft_mix >= (log_stft_mix.max() - active_threshold)).astype(np.int)

    """ Step 4: Get the label
    """
    Y = np.array([(log_stft1 >= log_stft2), (log_stft1 < log_stft2)]).astype(np.int)
    Y = log_stft_mix_active[None, ...] * Y

    Y = Y > 0
    return log_stft_mix, Y, phase_mix

#%%
def restore_waveform(config, X_mag, X_pha, mask):
    
    fs = config.get("fs", log=False)
    frame_len = config.get("frame_len", log=False)
    frame_hop = config.get("frame_hop", log=False)
    n_fft = config.get("n_fft", log=False)
    
    X_mag = librosa.core.db_to_power(X_mag, ref=1.0)
    X_sep = X_mag * mask
    
    wav = librosa.core.istft(X_sep * X_pha, 
                              hop_length=int(np.floor(frame_hop * fs)), 
                              win_length=int(np.floor(frame_len * fs)), 
                              window=np.sqrt(np.hanning(n_fft)), center='True')
    return wav





#%%
def smo_pose(pose, win_size=0):
    
    
    # fill zero entries
    for i in range(pose.shape[1]):

          tmp = pose[:,i]
          idx_nz = np.where(tmp!=0)[0]
          tmp[:min(idx_nz)] = tmp[min(idx_nz)]
          idx_z = np.where(tmp==0)[0]

          for k in range(len(idx_z)):
              tmp[idx_z[k]] = tmp[idx_z[k]-1]
          
          if win_size > 1:
              p = int(win_size/2)
              tmp[p:-p] = np.convolve(tmp, np.ones(win_size)/win_size, 'valid')
          
          pose[:,i] = tmp
                    
    return pose


#%%
def norm_pose(pose):
    
    m_x = np.mean(pose[:,:10])
    m_y = np.mean(pose[:,10:])
    pose[:,:10] -= m_x
    pose[:,10:] -= m_y
    
    std = np.std(pose)
    pose /= std
    
    return pose
    
#%%

def add_audio(file_vid, file_audio, file_out):
    #%%
    command = ''
    command += 'ffmpeg '
    command += '-i '
    command += '"' + file_audio + '" '
    command += '-i '
    command += '"' + file_vid + '" '
    command += '-c:v libx264 '
    command += '-c:a aac '
    command += '"' + file_out + '" '
    #%%
    command += '-y '
    #%%
    os.system(command)
    
    os.remove(file_vid)
    
    
def write_video(pose, filename_vid, filename_aud):
    '''
    pose: n_frame * 10 * 2
    '''
   
    filename_vid_silent = filename_vid[:-4] + "_silent" + ".mp4"
    FFMpegWriter = anm.writers['ffmpeg']
    writer = FFMpegWriter(fps=125)
    fig, ax = plt.subplots(1, 1)
    l_edge, = ax.plot([], [], color=[0.7,0.7,0.7])
    l_part1, = ax.plot([], [], 'ro')
    l_part2, = ax.plot([], [], 'bo')
    l_part3, = ax.plot([], [], 'go')
    
    lim = 3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.xaxis.grid()
    ax.yaxis.grid()
    
    n_frame = pose.shape[0]
    with writer.saving(fig, filename_vid_silent, 100):
        for fnum in range(n_frame):
#            if fnum % 100 == 0:
#                print ("frame %d of %d" % (fnum, n_frame))
            pose_fnum = pose[fnum,:,:]
            pose_fnum[:,1] = -pose_fnum[:,1]
            l_part1.set_data(pose_fnum[np.array([0,1,8,9]),0], pose_fnum[np.array([0,1,8,9]),1])
            l_part2.set_data(pose_fnum[2:5,0], pose_fnum[2:5,1])
            l_part3.set_data(pose_fnum[5:8,0], pose_fnum[5:8,1])
            l_edge.set_data(pose_fnum[np.array([4,3,2,1,0,1,8,1,9,1,5,6,7]),0], pose_fnum[np.array([4,3,2,1,0,1,8,1,9,1,5,6,7]),1])
            writer.grab_frame()
    
    
    add_audio(filename_vid_silent, filename_aud, filename_vid)





















    
    




