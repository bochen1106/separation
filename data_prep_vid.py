#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:24:41 2018

@author: bochen
"""

import numpy as np
import os
import glob
import random
import h5py
import re
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import util

#import sys
#sys.exit(0)

class DataPrepVid(object):
    
    def __init__(self, config=None, logger=None):
        
        logger.log("##########################################")
        logger.log("Initilize the data preparer")
        logger.log("##########################################")
        self.config = config
        self.logger = logger
        
        self.path_dataset_pose = config.get("path_dataset_pose")
        self.path_audio_single = config.get("path_audio_single")
        self.path_pose_single = config.get("path_pose_single")

        self.list_piece = config.get("list_piece")
        self.idx_tracks = config.get("idx_tracks")
        self.files_track = config.get("files_track")
        
        self.fps = 29.97
        frame_len = config.get("frame_len")
        frame_hop = config.get("frame_hop")
        self.tcen_aud = np.array(range(626)) * frame_hop
        #%%
        
    def split_player(self):
        '''
        Split the original pose data, then each text file is for one player
        This function is only used once
        '''
        
        path_OpenPose = "../../URMP/Dataset_OpenPose/_body_keras"
        path_dataset_pose = self.path_dataset_pose
        list_piece = self.list_piece
        
        for piece_name in list_piece:
            filename = os.path.join(path_OpenPose, piece_name) + ".txt"
            data = np.loadtxt(filename)
            data = util.smo_pose(data, 5)
            instr_list = piece_name.split("_")[2:]
            piece_name_pure = piece_name.split("_")[0] + "_" + piece_name.split("_")[1]
            n_player = len( instr_list )

            for p in range(n_player):
                name = "pose_" + str(p+1) + "_" + instr_list[p] + "_" + piece_name_pure + ".txt"
                filename_out = os.path.join(path_dataset_pose, piece_name, name)
                if not os.path.exists(os.path.dirname(filename_out)):
                    os.makedirs(os.path.dirname(filename_out))
                data_player = data[:, p*20 : p*20+20].astype('d')
                np.savetxt(filename_out, data_player, fmt='%.2f\t')
                
    def get_segment(self, instr_list, data_type):
        
        path_dataset_pose = self.path_dataset_pose
        path_audio_single = self.path_audio_single
        path_pose_single = self.path_pose_single

        files_track = self.files_track
        fps = self.fps
        tcen_aud = self.tcen_aud
        
        for instr in instr_list:

            path_audio_single = os.path.join(path_audio_single, data_type, instr)
            filenames = glob.glob(path_audio_single + "/*.wav")
            path_pose_single = os.path.join(path_pose_single, data_type, instr)
            if not os.path.exists(path_pose_single):
                os.makedirs(path_pose_single)
            #%%
            for filename in filenames:
#            filename = filenames[0]
                
                name = os.path.basename(filename).split(".")[0]
                expr = r'(\w+)(\d{4})@trk(\d{2,4})&dur(\d{4})-(\d{4})'
                m = re.match(expr, name)
                idx_track = int(m.group(3))-1
                t_seg = ( int(m.group(4)), int(m.group(5)) )
                file_track = files_track[idx_track]
                file_track = file_track.split(".")[0]
                file_track = file_track.replace("AuSep", "pose")
                filename_pose = os.path.join(path_dataset_pose, file_track) + ".txt"
                #%%
                data = np.loadtxt(filename_pose)
                n_frame = data.shape[0]
                tcen_vid = (np.asarray(range(n_frame))+1) / fps
                idx1 = np.where(tcen_vid > t_seg[0])[0][0]
                idx2 = np.where(tcen_vid < t_seg[1])[0][-1]
                data_seg = data[idx1:idx2+1, :]
                tcen_vid = tcen_vid[idx1 : idx2+1] - t_seg[0]
                
                data_seg_interp = np.zeros((len(tcen_aud), 20))
                for k in range(data_seg.shape[1]):  # range(20)
                    tmp = data_seg[:,k]
                    tmp = np.interp(tcen_aud, tcen_vid, tmp)
                    data_seg_interp[:,k] = tmp
                
                filename_pose = os.path.join(path_pose_single, name) + ".txt"
                np.savetxt(filename_pose, data_seg_interp, fmt='%.2f\t')
                
    def get_pose_visualization(self, instr_list, data_type):
        

        
        instr = instr_list[0]
        
        path_audio_single = os.path.join(self.path_audio_single, data_type, instr)
        path_pose_single = os.path.join(self.path_pose_single, data_type, instr)
        path_pose_visualization = path_pose_single.replace("pose", "pose_visualization")
        if not os.path.exists(path_pose_visualization):
            os.makedirs(path_pose_visualization)
        
        filenames = glob.glob(path_pose_single + "/*.txt")
        
        for filename in filenames:

            name = os.path.basename(filename).split(".")[0]
        
            filename_aud = os.path.join(path_audio_single, name) + ".wav"
            filename_pose = os.path.join(path_pose_single, name) + ".txt"
            filename_vid = os.path.join(path_pose_visualization, name) + ".mp4"
            
            
            
            pose = np.loadtxt(filename_pose)
            pose = util.norm_pose(pose)
            pose = np.reshape(pose, (-1,10,2), 'F')    # n_frame * 10 * 2
            
            util.write_video(pose, filename_vid, filename_aud)
          
#%%
        
if __name__ == '__main__':

    from util.config import Config
    from util.logger import Logger

    config = Config("../model/config_001.json")
    config_idx = config.get("config_idx")
    path_set = "../data/set_" + config_idx

    path_pose_single = config.get("path_pose_single")
    log_filename = os.path.join(path_pose_single, "log.txt")
    logger = Logger(log_filename)
           
    d = DataPrepVid(config, logger)
#    d.split_player()
    d.get_segment(["vn"], "train")
    d.get_segment(["vn"], "valid")
    d.get_segment(["vn"], "test")
    # d.get_pose_visualization(["vn"], "train")
    #%%
#    pose = d.pose

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    