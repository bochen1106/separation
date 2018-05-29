#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:39:24 2018

@author: bochen
"""

from util.config import Config
from util.logger import Logger
from data_prep import DataPrep
#%%
if __name__ == '__main__':
    
    config = Config("../model/config_01.json")
    # config = Config("../config.json")
    logger = Logger(config)
    config.set_logger(logger)
    #%%
    d = DataPrep(config, logger)
    # instr_list = ["vn","va","vc","db","fl","ob","cl","bn","sax","tpt","hn","tbn","tba"]
    # d.get_single(instr_list=instr_list)
    
    instr_mix = config.get("instr_mix")
    n_sample_train = config.get("n_sample_train")
    n_sample_valid = config.get("n_sample_valid")
    n_sample_test = config.get("n_sample_test")
    
    d.get_instr_info(instr_mix)
    
     #    d.get_mix(data_type='train', n_sample=n_sample_train)
     #    d.get_feat(data_type='train')
     #    d.get_h5_aggr(data_type='train')
     #    
     #    d.get_mix(data_type='valid', n_sample=n_sample_valid)
     #    d.get_feat(data_type='valid')
     #    d.get_h5_aggr(data_type='valid')
    
    # d.get_mix(data_type='test', n_sample=n_sample_test)
    d.get_feat(data_type='test')
    d.get_h5_aggr(data_type='test')

