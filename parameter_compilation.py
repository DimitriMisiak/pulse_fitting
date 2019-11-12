#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:25:22 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.close('all')

res_dir = '/home/misiak/Analysis/pulse_fitting/results'
log_dir = '/home/misiak/Analysis/pulse_fitting/stream_logs'

stream_57, temp_57, polar_57,_= np.loadtxt(
        '/'.join((log_dir, 'run57_log.csv')),
        dtype=str,
        delimiter=',',
        skiprows=1,
        unpack=True
)

stream_59, temp_59, polar_59, _, _ = np.loadtxt(
        '/'.join((log_dir, 'run59_log.csv')),
        dtype=str,
        delimiter=',',
        skiprows=1,
        unpack=True
)

##### RUN57
xopt_57 = list()
sinf_57 = list()
ssup_57 = list()
for stream in stream_57:
    dirname = '_'.join( (stream, 'RED70', '2exp') )
    xopt_path = '/'.join( (res_dir, dirname, 'raw_xopt.txt') )
    try:
        xopt, sinf, ssup = np.loadtxt(xopt_path, unpack=True)
    except:
        print(dirname + ' has not succeeded D:')
        xopt = [None,]*4
        sinf = ssup = [0,]*4
    xopt_57.append(xopt)
    sinf_57.append(sinf)
    ssup_57.append(ssup)

cut_ind = (np.array(xopt_57)[:,0] != None)
xopt_57 = np.array(xopt_57)[cut_ind]
sinf_57 = np.array(sinf_57)[cut_ind]
ssup_57 = np.array(ssup_57)[cut_ind]

temp_57a = temp_57.astype(float)[cut_ind]
polar_57a = polar_57.astype(float)[cut_ind]


#### RUN 59
xopt_59 = list()
sinf_59 = list()
ssup_59 = list()
for stream in stream_59:
    dirname = '_'.join( (stream, 'RED71', '2exp') )
    xopt_path = '/'.join( (res_dir, dirname, 'raw_xopt.txt') )
    try:
        xopt, sinf, ssup = np.loadtxt(xopt_path, unpack=True)
    except:
        print(dirname + ' has not succeeded D:')
        xopt = [None,]*4
        sinf = ssup = [0,]*4
    xopt_59.append(xopt)
    sinf_59.append(sinf)
    ssup_59.append(ssup)

cut_ind = (np.array(xopt_59)[:,0] != None)
xopt_59 = np.array(xopt_59)[cut_ind]
sinf_59 = np.array(sinf_59)[cut_ind]
ssup_59 = np.array(ssup_59)[cut_ind]

temp_59a = temp_59.astype(float)[cut_ind]
polar_59a = polar_59.astype(float)[cut_ind]

# =============================================================================
# EPS PLOT
# =============================================================================

temp_color = ('skyblue', 'limegreen', 'coral', 'k')
title_list = ('eps plot', 'tau1 plot', 'tau2 plot', 'tau th plot')

fig, axes = plt.subplots(nrows=2, ncols=2)
axes = axes.flatten()

for j,ax in enumerate(axes):
    ax.set_title(title_list[j])
    
    cut_14 = (temp_57a == 14)
    cut_16 = (temp_57a == 16)
    cut_18 = (temp_57a == 18)
    cut_20 = (temp_57a == 20)
    cut_list = (cut_14, cut_16, cut_18, cut_20)
    yerr = np.stack((sinf_57[:,j], ssup_57[:,0]))
    for i in range(4):
        cut = cut_list[i]
        col = temp_color[i]
        
        ax.errorbar(polar_57a[cut], xopt_57[cut,j], yerr[:,cut],
                     ls='none', marker='s', color=col)
    
    cut_14 = (temp_59a == 14)
    cut_16 = (temp_59a == 16)
    cut_18 = (temp_59a == 18)
    cut_20 = (temp_59a == 20)
    cut_list = (cut_14, cut_16, cut_18, cut_20)
    yerr = np.stack((sinf_59[:,j], ssup_59[:,0]))
    for i in range(4):
        cut = cut_list[i]
        col = temp_color[i]
        
        ax.errorbar(polar_59a[cut], xopt_59[cut,j], yerr[:,cut],
                     ls='none', marker='o', color=col)
    
    ax.set_xscale('log')

for ax in axes[1:]:
#    ax.set_yscale('log')
    pass