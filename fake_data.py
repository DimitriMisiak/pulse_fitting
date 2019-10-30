#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:53:05 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt

import red_magic as rmc

def data_generation(model, npz_path,
                    fs=1e3, wlen=1, param_exp=None, npulse=10, nnoise=10):

    time_array = np.arange(0, wlen, fs**-1)
    
    if model == '2exp':
        model_pulse = rmc.Model_pulse(model='2exp')
    else:
        raise Exception('Model "{}" is not implemented yet.'.format(model))
        
    model_noise = rmc.Model_white_noise(level=1e-4)
    
    t0_data = np.random.uniform(low=0.4*wlen, high=0.6*wlen, size=npulse)
    
    # pulse windows
    if param_exp is None:
        param_exp = model_pulse.parameters_0[:-1]
    
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
    
    # saving the data
    np.savez(
            npz_path,
            pulse_data=pulse_data,
            noise_data=noise_data,
            t0_data=t0_data,
            time_array=time_array
    )


if __name__ == '__main__':
    
    model = '2exp'
    npz_path = '/home/misiak/projects/pulse_fitting/archive/fake_data.npz'
    
    data_generation(model, npz_path, wlen=10)
    
    data = np.load(npz_path)
    t0_data = data['t0_data']
    time_array = data['time_array']
    pulse_data = data['pulse_data']
    noise_data = data['noise_data']
    
    plt.close('all')
    fig = plt.figure('DATA')
    axes = fig.subplots(nrows=2)
    
    for t0, pulse in zip(t0_data, pulse_data):
        line, = axes[0].plot(time_array, pulse, alpha=0.3)
        axes[0].axvline(t0, color=line.get_c())
    
    for noise in noise_data:
        axes[1].plot(time_array, noise, alpha=0.3)
        
    #axes[0].legend(title='pulse data')
    #axes[1].legend(title='noise data')
    for ax in axes:
    #    ax.legend()
        ax.grid()
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
    fig.tight_layout()

