#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:24:41 2018

@author: bochen
"""

import numpy as np
import librosa
import os
import glob
import random
import h5py
import cPickle as pickle
import util

class DataPrep(object):
    
    def __init__(self, config=None, logger=None):
        
        logger.log("##########################################")
        logger.log("Initilize the data preparer")
        logger.log("##########################################")
        
        self.config = config
        self.logger = logger
        self.fs = config.get("fs")
        self.path_dataset = config.get("path_dataset")
        self.files_track = config.get("files_track", log=False)
        self.path_audio = config.get("path_audio")
        self.path_feat = config.get("path_feat")
        
        

        
        
    def get_single(self, dur_seg=5, split_ratio=0.7, instr_list=None):
        '''
        generate the single track audio excerpts for the given instruments
        
        - input
            - dur_seg:      the duration (in sec) of each audio excerpt
            - split_ratio:  the ratio to split the data between train and valid/test set
                            default: 0.5 (train:valid:test = 0.5:0.25:0.25)
            - instr_list:   a list of instruments to generate the audio excerpts
            
        - output
            there is no function output parameter
            it outputs the single audio excerpts in the folder path_single
            
        sample usage:
            obj.get_single(instr_list=["vn", "va", "vc", "vd", "fl", "cl"])
        
        note:
            - this function may only need to run once
            - each call generates all the train/valid/test
        '''
        
        config = self.config
        logger = self.logger
        logger.log("++++++++++++++++++++++++++++++++++++++++++")
        logger.log("generate audio excerpts for single tracks")
                   
        path_dataset = self.path_dataset
        files_track = self.files_track
        if instr_list is None:
            raise Exception("No instrument given")
            return
        n_instr = len(instr_list)
        idx_instr_list = []
        for i in range(n_instr):
            idx_instr_list.append(config.get("idx_"+instr_list[i]))
        
        for i in range(n_instr):  # the loop of each instrument
            instr = instr_list[i]
            idx_instr = idx_instr_list[i]
            n_track = len(idx_instr_list[i])
            logger.log("__________________________________________")
            logger.log("instrument: [%s], number of tracks: %d" % (instr, n_track) )
#%%
            bound1 = max(1, int(n_track*split_ratio))
            bound2 = max( bound1+1, int(n_track*(0.5+split_ratio/2)) )
            idx_train = range(0, bound1)
            idx_valid = range(bound1, bound2)
            idx_test = range(min(n_track-1, bound2), n_track)
            #%%
            logger.log("idx_train:%s, idx_valid:%s, idx_test:%s" % 
                       (str(idx_train),str(idx_valid),str(idx_test)) )

            data_type = "train"
            path_single = os.path.join(self.path_audio, "single", data_type, instr)
            if not os.path.exists(path_single):
                os.makedirs(path_single)
            count = 0
            for j in idx_train:
                filename = os.path.join( path_dataset, files_track[ idx_instr[j]-1 ] )
                wav, fs = librosa.load(filename, sr=self.fs)
                duration = librosa.get_duration(wav, fs)
                num_seg = int(duration / dur_seg)
                for k in range(num_seg):
                    t_seg = (dur_seg*k, dur_seg*(k+1))
                    data = wav[t_seg[0]*fs : t_seg[1]*fs]
                    e = np.sqrt(np.mean(data**2))
                    if e < 0.01:
                        continue
                    count += 1
                    name = instr + "%04d"%count + "@trk%04d"%idx_instr[j] + "&dur%04d-%04d"%(t_seg[0], t_seg[1])
                    filename_out = os.path.join(path_single, name) + ".wav"
                    librosa.output.write_wav(filename_out, data, fs)
            logger.log("finished generating %d samples for train set" % count)
            
            data_type = "valid"
            path_single = os.path.join(self.path_audio, "single", data_type, instr)
            if not os.path.exists(path_single):
                os.makedirs(path_single)
            count = 0
            for j in idx_valid:
                filename = os.path.join( path_dataset, files_track[ idx_instr[j]-1 ] )
                wav, fs = librosa.load(filename, sr=self.fs)
                duration = librosa.get_duration(wav, fs)
                num_seg = int(duration / dur_seg)
                for k in range(num_seg):
                    t_seg = (dur_seg*k, dur_seg*(k+1))
                    data = wav[t_seg[0]*fs : t_seg[1]*fs]
                    e = np.sqrt(np.mean(data**2))
                    if e < 0.01:
                        continue
                    count += 1
                    name = instr + "%04d"%count + "@trk%04d"%idx_instr[j] + "&dur%04d-%04d"%(t_seg[0], t_seg[1])
                    filename_out = os.path.join(path_single, name) + ".wav"
                    librosa.output.write_wav(filename_out, data, fs)
            logger.log("finished generating %d samples for valid set" % count)
            
            data_type = "test"
            path_single = os.path.join(self.path_audio, "single", data_type, instr)
            if not os.path.exists(path_single):
                os.makedirs(path_single)
            count = 0
            for j in idx_test:
                filename = os.path.join( path_dataset, files_track[ idx_instr[j]-1 ] )
                wav, fs = librosa.load(filename, sr=self.fs)
                duration = librosa.get_duration(wav, fs)
                num_seg = int(duration / dur_seg)
                for k in range(num_seg):
                    t_seg = (dur_seg*k, dur_seg*(k+1))
                    data = wav[t_seg[0]*fs : t_seg[1]*fs]
                    e = np.sqrt(np.mean(data**2))
                    if e < 0.01:
                        continue
                    count += 1
                    name = instr + "%04d"%count + "@trk%04d"%idx_instr[j] + "&dur%04d-%04d"%(t_seg[0], t_seg[1])
                    filename_out = os.path.join(path_single, name) + ".wav"
                    librosa.output.write_wav(filename_out, data, fs)
            logger.log("finished generating %d samples for test set" % count)
           
            
            
    def get_mix(self, data_type=None, n_sample=0):
        
        '''
        generate the 2-instrument mix audio excerpts for the given instruments
        It random get the single track samples for each instrument until reach the desired number of mix samples
        
        - input:
            - instr_list:   a list of two given instruments to generate the mix
            - data_type:    either "train", or "valid", or "test"
            - n_samples:    the desired number of mix samples
        
        - output:
            there is no function output parameter
            it outputs the mix audio excerpts in the folder path_mix
            
        sample usage:
            obj.get_mix(instr_list=["vn", "cl"], data_type="train", n_sample=100)
        '''
        
        logger = self.logger
        logger.log("++++++++++++++++++++++++++++++++++++++++++")
        logger.log("generate audio excerpts for mix tracks")
        
        instr_mix = self.instr_mix
        [instr1, instr2] = instr_mix.split("-")
        
        path_mix = os.path.join(self.path_audio, "mix", data_type, instr_mix)
        if not os.path.exists(path_mix):
            os.makedirs(path_mix)
        path_single_1 = os.path.join(self.path_audio, "single", data_type, instr1)
        path_single_2 = os.path.join(self.path_audio, "single", data_type, instr2)
        filenames_1 = glob.glob(path_single_1 + "/*.wav")
        filenames_2 = glob.glob(path_single_2 + "/*.wav")
        n_sample_1 = len(filenames_1)
        n_sample_2 = len(filenames_2)
        logger.log("mix [%s] (%d samples) and [%s] (%d samples)" % 
                   (instr1, n_sample_1, instr2, n_sample_2))
        
        if n_sample > n_sample_1*n_sample_2 * 0.8:
            raise Exception("too much desired number of mix samples")
            return
        
        count = 0
        idx_all = []
        while count < n_sample:
            idx1 = random.sample(range(n_sample_1), 1)[0]
            idx2 = random.sample(range(n_sample_2), 1)[0]
            if (idx1, idx2) in idx_all:
                continue
            if instr1==instr2 and idx1==idx2:
                continue
            
            count += 1
            idx_all.append((idx1, idx2))
            
            filename1 = filenames_1[idx1]
            filename2 = filenames_2[idx2]
            expr = r'(\w+)(\d{4})@'
            m1 = re.match(expr, os.path.basename(filename1))
            m2 = re.match(expr, os.path.basename(filename2))
            name_out = instr1+m1.group(2) + "-" + instr2+m2.group(2)
            filename_out = os.path.join(path_mix, name_out) + ".wav"
            
            wav1, fs = librosa.load(filename1, sr=self.fs)
            wav2, fs = librosa.load(filename2, sr=self.fs)
            wav = wav1 + wav2
            librosa.output.write_wav(filename_out, wav, fs)
        logger.log("finish %d mix samples for [%s]" % (n_sample, instr_mix))
    
    
    def get_feat(self, data_type=None):
        '''
        feature extraction
        
        - input:
            - instr_list:   a list of two given instruments to generate the mix
            - data_type:    either "train", or "valid", or "test"
            
        - output:
            there is no function output parameter
            it outputs the extracted features as h5 files in the folder path_feat
            each h5 file contains
            - spec magnitude of the audio mixture (F*T)
            - spec phase of the audio mixture (F*T)
            - separation mask (F*T*2)
        '''
        config = self.config
        logger = self.logger
        logger.log("++++++++++++++++++++++++++++++++++++++++++")
        logger.log("extract audio features")
        
        instr_mix = self.instr_mix
        [instr1, instr2] = instr_mix.split("-")

        path_single_1 = os.path.join(self.path_audio, "single", data_type, instr1)
        path_single_2 = os.path.join(self.path_audio, "single", data_type, instr2)
        path_mix = os.path.join(self.path_audio, "mix", data_type, instr_mix)
        path_feat = os.path.join(self.path_feat, data_type, instr_mix)

        if not os.path.exists(path_feat):
            os.makedirs(path_feat)
        
        filenames = glob.glob(path_mix + "/*.wav")
        
        for filename in filenames:
            name = os.path.basename(filename).split(".")[0]
            name1, name2 = name.split("-")
            filename1 = glob.glob(os.path.join(path_single_1, name1) + "*.wav")[0]
            filename2 = glob.glob(os.path.join(path_single_2, name2) + "*.wav")[0]

            data_mix, fs = librosa.load(filename, sr=self.fs)
            data1, fs = librosa.load(filename1, sr=self.fs)
            data2, fs = librosa.load(filename2, sr=self.fs)
            X_mag, Y, X_pha = util.extract_feat(config, data_mix, data1, data2)
            
            filename_h5 = os.path.join(path_feat, name) + ".h5"
            f = h5py.File(filename_h5)
            f['X_mag'] = X_mag
            f['X_pha'] = X_pha
            f['Y'] = Y
            f.close()
            
    def get_h5_aggr(self, data_type=None):
        '''
        aggregate the exgtracted features as one h5 file
        
        - input:
            - instr_list:   a list of two given instruments to generate the mix
            - data_type:    either "train", or "valid", or "test"
            
        - output:
            there is no function output parameter
            it outputs the aggregrated h5 files and the info pickle files in the folder path_h5
        '''
        logger = self.logger
        logger.log("++++++++++++++++++++++++++++++++++++++++++")
        logger.log("aggregrate h5 files")
        
        instr_mix = self.instr_mix
        [instr1, instr2] = instr_mix.split("-")
        
        path_feat = os.path.join(self.path_feat, data_type, instr_mix)
        path_h5_aggr = os.path.join(self.path_feat, data_type)
        
        filenames = glob.glob(path_feat + "/*.h5")\
        
        X_total = []
        Y_total = []
        info = {}
        idx_start = 0
        for i in range(len(filenames)):
            f = h5py.File(filenames[i])
            name = os.path.basename(filenames[i])
            X = np.array(f['X_mag'])  # F x T
            X = np.swapaxes(a=X, axis1=0, axis2=1)  # T x F
            Y = np.array(f['Y'])  # 3 x F x T
            Y = np.swapaxes(a=Y, axis1=0, axis2=2)  # T x F x 3
            f.close()
            
            # Append to the total array
            X_total.append(X)
            Y_total.append(Y)
        
            # Update the info
            info[name] = [idx_start, X.shape[0]]
        
            # Update the start index
            idx_start += X.shape[0]
        
        X_total = np.concatenate(tuple(X_total), axis=0)
        Y_total = np.concatenate(tuple(Y_total), axis=0)
        freq_mean = np.mean(X_total, axis=0)
        freq_std = np.std(X_total, axis=0)
        global_mean = np.mean(X_total)
        global_std = np.std(X_total)
        
        # save the data
        filename_h5 = os.path.join(path_h5_aggr, instr_mix) + ".h5"
        f = h5py.File(filename_h5)
        f['X'] = X_total
        f['Y'] = Y_total
        f['freq_mean'] = freq_mean
        f['freq_std'] = freq_std
        f['global_mean'] = global_mean
        f['global_std'] = global_std
        f.close()
        
        filename_info = os.path.join(path_h5_aggr, instr_mix) + ".cpickle"
        pickle.dump({'info': info}, open(filename_info, 'w'))
        
    def get_instr_info(self, instr_mix=None):
        logger = self.logger
        logger.log("++++++++++++++++++++++++++++++++++++++++++")
        logger.log("get the instrument info")
        
        if instr_mix is None:
            raise Exception("Please specify meaningful input parameters")
            return
        if len(instr_mix.split("-")) != 2:
            raise Exception("Please input exact two instruments")
            return
        self.instr_mix = instr_mix
        
        
#%%
        
if __name__ == '__main__':
        
    from util.config import Config
    from util.logger import Logger
    config = Config("../config.json")
    logger = Logger(config)
    config.set_logger(logger)            
    d = DataPrep(config, logger)
    
#    d.get_single(instr_list=["va", "tba"])
    
    instr_mix = config.get("instr_mix")
    n_sample_train = config.get("n_sample_train")
    n_sample_valid = config.get("n_sample_valid")
    n_sample_test = config.get("n_sample_test")
    
    d.get_instr_info(instr_mix)
    
    d.get_mix(data_type='train', n_sample=n_sample_train)
    d.get_feat(data_type='train')
    d.get_h5_aggr(data_type='train')
    
    d.get_mix(data_type='valid', n_sample=n_sample_valid)
    d.get_feat(data_type='valid')
    d.get_h5_aggr(data_type='valid')
    
    d.get_mix(data_type='test', n_sample=n_sample_test)
    d.get_feat(data_type='test')
    d.get_h5_aggr(data_type='test')





