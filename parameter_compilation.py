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
plt.rcParams['text.usetex']=True

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


## raw to fine
#xopt_57_bis = 10**xopt_57
#xinf_57 = 10**(xopt_57-sinf_57)
#xsup_57 = 10**(ssup_57 + xopt_57)
#sinf_57 = xopt_57_bis - xinf_57
#ssup_57 = xsup_57 - xopt_57_bis
#xopt_57 = xopt_57_bis

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

## raw to fine
#xopt_59_bis = 10**xopt_59
#xinf_59 = 10**(xopt_59 - sinf_59)
#xsup_59 = 10**(ssup_59 + xopt_59)
#sinf_59 = xopt_59_bis - xinf_59
#ssup_59 = xsup_59 - xopt_59_bis
#xopt_59 = xopt_59_bis


# =============================================================================
# EPS PLOT
# =============================================================================

temp_color = ('b', 'green', 'orange', 'red')
title_list = ('eps plot', 'tau1 plot', 'tau2 plot', 'tau th plot')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
axes = axes.flatten()

for j,ax in enumerate(axes):
#    ax.set_title(title_list[j])
    
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
                     ls='none', marker='s', color=col, alpha=0.7)
#        ax.plot(polar_57a[cut], xopt_57[cut,j],
#                     ls='none', marker='s', color=col, alpha=0.3)
    
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
                     ls='none', marker='o', color=col, mec='k', alpha=0.7)
#        ax.plot(polar_59a[cut], xopt_59[cut,j],
#                     ls='none', marker='o', color=col, mec='k',alpha=0.3)    
    ax.set_xscale('log')

for ax in axes:
    ax.set_xlabel('Bias Current [nA]')
    ax.grid()
    
axes[0].set_ylabel(r'$\epsilon$')
axes[1].set_ylabel(r'$log_{10}(\tau_1)$')
axes[2].set_ylabel(r'$log_{10}(\tau_2)$')
axes[3].set_ylabel(r'$log_{10}(\tau_3)$')

fig.tight_layout()