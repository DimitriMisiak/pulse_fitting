#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: misiak
"""

import sys
import os
import json

from run_mcmc_function import run_mcmc

# getting arguments from command line
if len(sys.argv) != 2:
    raise Exception('Expecting 1 argument: model')
    
model, = sys.argv[1:]

# hard coding the DATA and SAVE directories
ARC_DIR = '/home/misiak/projects/pulse_fitting/archive'

label = '_'.join((model, 'fake'))
save_dir = '/'.join((ARC_DIR, label))

assert os.path.isdir(save_dir)

data_path = '/'.join((save_dir, 'fake_data.npz'))
h5_path = '/'.join((save_dir, 'mcmc_output.h5'))
config_path = '/'.join((save_dir, 'mcmc_config.json'))
autocorr_save = '/'.join((save_dir, 'autocorr_time.txt'))

with open(config_path, 'r') as cfg:
    config = json.load(cfg)
    
pinit = config['Parameters']['pinit']
prior_dist = config['Prior']['distribution']
prior_arg1 = config['Prior']['arg1']
prior_arg2 = config['Prior']['arg2']

walkers_per_dim = config['MCMC']['walkers_per_dim']
max_n = config['MCMC']['max_iterations']
tau_rtol = config['MCMC']['tau_rtol']

run_mcmc(
        data_path, h5_path, model, pinit,
        prior_dist, prior_arg1, prior_arg2,
        walkers_per_dim,
        max_n,
        tau_rtol,
        autocorr_save
)
