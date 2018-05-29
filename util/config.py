#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:39:02 2017

@author: user
"""

#!/usr/bin/python

import sys
import json


class Config(object):
    def __init__(self, filename_json=None, strict=True):
        self.contents = dict()
        self.logger = None
        self.strict = strict
        self.filename_json = filename_json

        if filename_json is not None:
            if isinstance(filename_json, str) or isinstance(filename_json, unicode):
                self.filename_json = filename_json
                with open(filename_json, 'rb') as f:
                    self.contents = json.load(f)
            elif isinstance(filename_json, dict):
                self.contents = filename_json
            else:
                print('cannot load proper json')
                sys.exit(-1)


    def get(self, key, default_value=None, log=True):
        if key in self.contents:
            if self.logger is not None and log is True:
                self.logger.log('getting %s as %s' % (key, str(self.contents[key])), 0)
            return self.contents[key]
        if self.strict is True:
            if self.logger is not None:
                self.logger.log("cannot find configuration %s from %s" % (key, self.filename_json), 0)
            sys.exit(-1)
        # load it to json to dump it out
        self.contents[key] = default_value

        if self.logger is not None:
            self.logger.log('getting %s as %s' % (key, str(default_value)), 0)

        return default_value

    def set(self, key, value):
        self.contents[key] = value
        
        if self.logger is not None:
            self.logger.log('setting %s as %s' % (key, str(value)), 0)
        return value

    def set_logger(self, logger):
        self.logger = logger

    # dump all the contents to json including the default values
    def dump(self, filename_out):
        with open(filename_out, 'wb') as f:
            json.dump(self.contents, f, indent=4)

    def copy(self, config_obj):
        for key in self.contents.keys():
            config_obj.get(key, self.contents[key])

