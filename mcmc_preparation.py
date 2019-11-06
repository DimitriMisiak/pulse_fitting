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

from true_data import data_selection
from analysis_config import proto_fitting


# getting arguments from command line
if len(sys.argv) != 4:
    raise Exception('Expecting 3 arguments: stream, detector and model')
    
stream, detector, model = sys.argv[1:]

# hard coding the DATA and SAVE directories
DATA_DIR_LOCAL = '/home/misiak/Data/data_run59'
ARC_DIR_LOCAL = '/home/misiak/projects/pulse_fitting/archive'
DATA_DIR_CC = '/sps/edelweis/CRYO_IPNL/BatchOUTPUT'
ARC_DIR_CC = '/pbs/home/d/dmisiak/mcmc_output'

# priority to local path, then CC, then raise exception of paths not found.
if os.path.isdir(ARC_DIR_LOCAL):
    ARC_DIR = ARC_DIR_LOCAL
elif os.path.isdir(ARC_DIR_CC):
    ARC_DIR = ARC_DIR_CC
else:
    raise Exception(
            (
                    'The directories {} could not be found. ARC_DIR cannot be assigned.'
            ).format(ARC_DIR_LOCAL, ARC_DIR_CC)
    )

# priority to local path, then CC, then raise exception of paths not found.
if os.path.isdir(DATA_DIR_LOCAL):
    DATA_DIR = DATA_DIR_LOCAL
elif os.path.isdir(DATA_DIR_CC):
    DATA_DIR = DATA_DIR_CC
else:
    raise Exception(
            (
                    'The directories {} could not be found. DATA_DIR cannot be assigned'
            ).format(DATA_DIR_LOCAL, DATA_DIR_CC)
    )


label = '_'.join((stream, detector, model))
save_dir = '/'.join((ARC_DIR, label))

# super mkdir for the save directories
path_list = (save_dir,)
for path in path_list:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# data selection from true_data.py
data_selection(stream, detector, DATA_DIR, save_dir)


# manual fitting from analysis_config.py
data_path = '/'.join((save_dir, 'true_data.npz'))
pf = proto_fitting(data_path, model, save_dir)
param_opt = pf.get_param()

ndim = len(param_opt)


# creating the config_file for the mcmc
config = dict()
config['Data'] = {
        'true_data': True,
        'directory': DATA_DIR,
        'stream': stream,
        'detector': detector,         
}

config['Parameters'] = {
        'label': ['p{}'.format(i) for i in range(ndim)],
        'pinit': list(param_opt),
}

# by default, normal distribution centered in pinit
# and with a relative sigma of 0.1
config['Prior'] = {
        'distribution': ['norm',]*ndim,
        'arg1' : list(param_opt),
        'arg2' : [abs(0.1*p) for p in param_opt],
}

config['Model'] = model

config['MCMC'] = {
        'walkers_per_dim': 10,
        'max_iterations': int(1e5),
        'tau_rtol': 0.01,
}

configpath = '/'.join((save_dir, 'mcmc_config.json'))
with open(configpath, 'w') as cfg:
    json.dump(config, cfg, indent=4)
        