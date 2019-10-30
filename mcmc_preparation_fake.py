#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:32:02 2019

@author: misiak
"""
import sys
import os
import json

from fake_data import data_generation
from analysis_config import proto_fitting

# getting arguments from command line
if len(sys.argv) != 2:
    raise Exception('Expecting 1 argument: model')
    
model, = sys.argv[1:]

# hard coding the DATA and SAVE directories
ARC_DIR = '/pbs/home/d/dmisiak/mcmc_output'

label = '_'.join((model, 'fake'))
save_dir = '/'.join((ARC_DIR, label))

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

# manual fitting from analysis_config.py
pf = proto_fitting(data_path, model, save_dir)
param_opt = pf.get_param()

ndim = len(param_opt)


# creating the config_file for the mcmc
config = dict()
config['Data'] = {
        'true_data': False,    
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
        
