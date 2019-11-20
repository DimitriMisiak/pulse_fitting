#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: misiak
"""

import sys
import os
import json

from _mcmc_function import mcmc_from_atelier, Atelier

# getting arguments from command line
if len(sys.argv) != 4:
    raise Exception('Expecting 3 arguments: stream, detector and model')
    
stream, detector, model = sys.argv[1:]

# hard coding the LIBrary and OUTput directories
LIB_DIR_LOCAL = '/home/misiak/Analysis/pulse_fitting/event_library_test'
LIB_DIR_CC = '/sps/edelweis/dmisiak/Analysis/pulse_fitting/event_library'
OUT_DIR_LOCAL = '/home/misiak/Analysis/pulse_fitting/mcmc_output_test'
OUT_DIR_CC = '/sps/edelweis/dmisiak/Analysis/pulse_fitting/mcmc_output'

# priority to local path, then CC, then raise exception of paths not found.
if os.path.isdir(LIB_DIR_LOCAL):
    LIB_DIR = LIB_DIR_LOCAL
elif os.path.isdir(LIB_DIR_CC):
    LIB_DIR = LIB_DIR_CC
else:
    raise Exception(
            (
                    'The directories {} could not be found.'
            ).format(LIB_DIR_LOCAL, LIB_DIR_CC)
    )

# priority to local path, then CC, then raise exception of paths not found.
if os.path.isdir(OUT_DIR_LOCAL):
    OUT_DIR = OUT_DIR_LOCAL
elif os.path.isdir(OUT_DIR_CC):
    OUT_DIR = OUT_DIR_CC
else:
    raise Exception(
            (
                    'The directories {} could not be found.'
            ).format(OUT_DIR_LOCAL, OUT_DIR_CC)
    )

lib_label = '_'.join((stream, detector))
lib_dir = '/'.join((LIB_DIR, lib_label))
assert os.path.isdir(lib_dir)

out_label = '_'.join((stream, detector, model))
out_dir = '/'.join((OUT_DIR, out_label))

# super mkdir for the save directories
path_list = (out_dir,)
for path in path_list:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

if detector == 'fake':
    npz_path = '/'.join((lib_dir, 'fake_data.npz'))
else:
    npz_path = '/'.join((lib_dir, 'true_data.npz'))


### Parameters from the MCMC
ato = Atelier(npz_path, model)
WALKERS_PER_DIM = 16
MAX_ITER = int(1e5)

### creating the config_file for the mcmc
config = dict()
config['Selection'] = {
        'stream': stream,
        'detector': detector,
        'directory': LIB_DIR,         
}
config['MCMC'] = {
        'model': model,
        'walkers_per_dim': WALKERS_PER_DIM,
        'max_iter': MAX_ITER,
        'directory': OUT_DIR,
        'success': False,
}

configpath = '/'.join((out_dir, 'config.json'))
with open(configpath, 'w') as cfg:
    json.dump(config, cfg, indent=4)


### Finally launching the mcmc
mcmc_from_atelier(
        ato,
        walkers_per_dim=WALKERS_PER_DIM,
        max_iter=MAX_ITER,
        output_dir=out_dir,
        progress=True
)

### if mcmc is successful
config['MCMC']['success'] = True
with open(configpath, 'w') as cfg:
    json.dump(config, cfg, indent=4)

  