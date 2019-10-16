#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:53:05 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt

import red_magic as rmc

plt.close('all')
# =============================================================================
# CREATING FAKE DATA
# =============================================================================

npulse = 100
nnoise = 50

fs = 1e3
wlen = 0.5
time_array = np.arange(0, wlen, fs**-1)

model_pulse = rmc.Model_pulse(model='2exp')
model_noise = rmc.Model_white_noise(level=1e-4)

t0_data = np.random.uniform(low=0.24, high=0.26, size=npulse)

# pulse windows
param_exp = [0.2, 5e-2, 25e-2,  5e-3]
pulse_data = list()
for t0 in t0_data:
    param = param_exp + [t0,]
    amp = np.random.uniform(low=5, high=10)
    dc = np.random.uniform(low=-5, high=5)
    
    pulse = amp * model_pulse.function(param, time_array) + dc
    
    noise = model_noise.sample(fs, wlen)
    
    pulse_data.append(pulse+noise)

# noise windows
noise_data = list()
for i in range(nnoise):
    noise = model_noise.sample(fs, wlen)
    noise_data.append(noise)
    

# =============================================================================
# REPR DATA
# =============================================================================
fig = plt.figure('DATA')
axes = fig.subplots(nrows=2)

for t0, pulse in zip(t0_data, pulse_data):
    line, = axes[0].plot(time_array, pulse, alpha=0.1)
    axes[0].axvline(t0, color=line.get_c())

for noise in noise_data:
    axes[1].plot(time_array, noise, alpha=0.1)
    
axes[0].legend(title='pulse data')
axes[1].legend(title='noise data')
for ax in axes:
    ax.legend()
    ax.grid()
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time [s]')
fig.tight_layout()

# =============================================================================
# # SAVING FAKE DATA
# =============================================================================

np.savez(
        'fake_data',
        pulse_data=pulse_data,
        noise_data=noise_data,
        t0_data=t0_data,
        time_array=time_array
)

