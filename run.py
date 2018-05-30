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
from trainer import Trainer
#%%
if __name__ == '__main__':
    config = Config("../model/config_001.json")
    config_idx = config.get("config_idx")
    filename_log = "../model/model_" + config_idx + "/log.txt"
    logger = Logger(filename_log)
    config.set_logger(logger)

    t = Trainer(config, logger)
    t.run()
