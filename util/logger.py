#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:43:52 2017

@author: user
"""

import os
import time
import datetime


class Logger(object):
    def __init__(self, config):
        self.config = config
        self.file = None

        log_filename = config.get('log', None)
        if log_filename is not None:
            log_dir = os.path.split(log_filename)[0]
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            self.file = open(log_filename, 'wt')

        self.verbose_level = config.get('verbose_level', 0)

    def log(self, contents, level=1):
        # print level below verbose level
        st = datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S]')
        if level <= self.verbose_level:
            line = contents
            if isinstance(contents, list):
                line = '\n'.join(line)
                line = st + '\n' + line
            else:
                line = st + ': ' + line

            print(line)

            line = line + '\n'
            if self.file is not None:
                self.file.writelines(line)
                self.file.flush()


    def add_chainer_log(self, file_name):
        #%%
        fid = open(file_name, "r") 
        lines = fid.readlines()
        fid.close()
        #%%
        import re
        self.file.writelines("epoch \ttime_elap \tloss_main \tleft \tright \tloss_validation \tleft \tright\n")
        loss_vali_all = 0
        loss_vali_l = 0
        loss_vali_r = 0
        loss_main_all = 0
        loss_main_l = 0
        loss_main_r = 0
        for l in lines:

            if l.find("validation") >= 0:
                # Validation
                if l.find("loss_all") >= 0:
                    loss_vali_all = float( re.findall(r'\d+.\d+', l)[0] )
                if l.find("loss_left") >= 0:
                    loss_vali_l = float( re.findall(r'\d+.\d+', l)[0] )
                if l.find("loss_right") >= 0:
                    loss_vali_r = float( re.findall(r'\d+.\d+', l)[0] )
            elif l.find("main")>=0:
                # for the main 
                if l.find("loss_all") >= 0:
                    loss_main_all = float( re.findall(r'\d+.\d+', l)[0] )
                if l.find("loss_left") >= 0:
                    loss_main_l = float( re.findall(r'\d+.\d+', l)[0] )
                if l.find("loss_right") >= 0:
                    loss_main_r = float( re.findall(r'\d+.\d+', l)[0] )
                pass

            if "elapsed_time" in l:
                time = float( re.findall(r'\d+.\d+', l)[0] )
            if "epoch" in l:
                epoch = float( re.findall(r'\d+', l)[0] )
                
            if "}" in l:
                content = "%d \t%.1f \t\t%.4f \t\t%.4f \t%.4f \t%.4f \t\t%.4f \t%.4f\n" % (
                        epoch, time, loss_main_all, loss_main_l, loss_main_r, loss_vali_all, loss_vali_l, loss_vali_r)
                self.file.writelines(content)
