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


from comparator_class import model_function, Comparator

true_data = np.load('true_data.npz')
fake_data = np.load('fake_data.npz')
    
com = Comparator(true_data, model_function)
# =============================================================================
# FITTING
# =============================================================================
psd_data = [rmc.psd(fft, com.fs)[1] for fft in com.fft_data]

### PLOT FOR MANUAL FITTING
fig, axes = plt.subplots(nrows=2)

pulse_lines = list()
psd_lines = list()
for t0, pulse, psd in zip(com.t0_data, com.pulse_data, psd_data):
    line0, = axes[0].plot(com.time_array-t0, pulse, alpha=0.1)
    line1, = axes[1].loglog(com.freq_array, psd, alpha=0.1)
    pulse_lines.append(line0)
    psd_lines.append(line1)

axes[0].set_ylabel('Pulse')
axes[0].set_xlabel('Time [s]')
axes[1].set_ylabel('PSD Pulse')
axes[1].set_xlabel('Frequency [Hz]')
for ax in axes:
    ax.grid()

### preparing for manual fitting
pinit = [0.1, -2, -1,  1e-3]

def manual_fitting_funk(theta):
    pulse_template = model_function(list(theta)+[0.25,], com.time_array)
    psd_template = rmc.psd(np.fft.fft(pulse_template), com.fs)[1]
    return [pulse_template - np.mean(pulse_template), psd_template]

pulse_template, psd_template = manual_fitting_funk(pinit)
line0, = axes[0].plot(com.time_array-0.25, pulse_template, color='k')
line1, = axes[1].plot(com.freq_array, psd_template, color='k')

dc_pulse = [np.mean(pulse) for pulse in com.pulse_data]
def callback_fun(theta):
    chi2_list, amp_list = com.chi2_amp_fun(theta)
    
    # updataing chi2 value
    chi2 = np.sum(chi2_list)
    fig.suptitle('$\chi^2$={:.3e}'.format(chi2))
    
    # updating experimental curves
    for line, pulse, dc, amp in zip(pulse_lines, com.pulse_data, dc_pulse, amp_list):
        line.set_ydata( (pulse-dc)/amp )
#
    for line, psd, amp in zip(psd_lines, psd_data, amp_list):
        line.set_ydata( psd/amp**2 )
              

### actual manual fitting
manual_fitting = rmc.Manual_fitting(
        [line0, line1],
        manual_fitting_funk, 
        pinit,
        callback=callback_fun,
        chi2_fun=com.chi2_fun
)

def bonus_fun(event):
    
    rmc.mcmc_config('mcmc_config.json', manual_fitting.get_param())
    print('Opt param saved!')

manual_fitting.bonus_button('Save popt', bonus_fun)


plt.show()

