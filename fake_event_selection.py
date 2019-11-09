#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:32:02 2019

@author: misiak
"""
import sys
import os
import json
import matplotlib.pyplot as plt
plt.close('all')

from _fake_data import data_generation


# getting arguments from command line
if len(sys.argv) != 2:
    raise Exception('Expecting 1 argument: model')
    
model, = sys.argv[1:]


# hard coding the DATA and SAVE directories
LIB_DIR_LOCAL = '/home/misiak/Analysis/pulse_fitting/event_library_test'
LIB_DIR_CC = '/pbs/home/d/dmisiak/Analysis/pulse_fitting/event_library'

# priority to local path, then CC, then raise exception of paths not found.
if os.path.isdir(LIB_DIR_LOCAL):
    LIB_DIR = LIB_DIR_LOCAL
elif os.path.isdir(LIB_DIR_CC):
    LIB_DIR = LIB_DIR_CC
else:
    raise Exception(
            (
                    'The directories {} could not be found. LIB_DIR cannot be assigned.'
            ).format(LIB_DIR_LOCAL, LIB_DIR_CC)
    )


label = '_'.join((model, 'fake'))
save_dir = '/'.join((LIB_DIR, label))

# super mkdir for the save directories
path_list = (save_dir,)
for path in path_list:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# data generation from fake_data.py
data_path = '/'.join((save_dir, 'fake_data.npz'))
#data_selection(stream, detector, DATA_DIR, save_dir)
data_generation(model, data_path)


# creating the config_file for the mcmc
config = dict()
config['Data'] = {
        'true_data': False,
        'model': model      
}
config['Selection'] = {
        'directory': LIB_DIR,
}

configpath = '/'.join((save_dir, 'config.json'))
with open(configpath, 'w') as cfg:
    json.dump(config, cfg, indent=4)
        