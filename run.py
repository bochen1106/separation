#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:39:24 2018

@author: bochen
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


from util.config import Config
from util.logger import Logger
from data_prep import DataPrep
from trainer import Trainer
#%%
if __name__ == '__main__':
    
    config = Config("../config.json")
    logger = Logger(config)
    config.set_logger(logger)
    #%%
#    d = DataPrep(config, logger)
#    #d.get_single(instr_list=["va", "tba"])
#    d.get_instr_info(["ob", "hn"])
#    d.get_mix(data_type='train', n_sample=10)
#    d.get_feat(data_type='train')
#    d.get_h5_aggr(data_type='train')

#%%
    t = Trainer(config, logger, "ob-hn")