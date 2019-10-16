#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:24:11 2019

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np

import red_magic as rmc

model_pulse = rmc.Model_pulse('2exp')

def model_function(param, time_array):
    eps, logt1, logt2, tautherm, t0 = param
    t1 = 10**logt1
    t2 = 10**logt2
    return model_pulse.function([eps, t1, t2, tautherm, t0], time_array)


class Comparator(object):
    
    def __init__(self, data, model_fun):
        
        self.data = data
        self.model_fun = model_fun
        
        attr_lab = ('pulse_data', 'noise_data', 't0_data', 'time_array')
        for lab in attr_lab:
            setattr(self, lab, self.data[lab])
       
        self.fs = (self.time_array[1]-self.time_array[0])**-1
        self.freq_array = rmc.psd_freq(self.time_array)
        self.fft_data = [np.fft.fft(pulse) for pulse in self.pulse_data]

        self.noise_psd = [rmc.psd(np.fft.fft(noise), self.fs)[1] for noise in self.noise_data]
        self.err_array = np.mean(self.noise_psd, axis=0)
    
    def chi2_amp_fun(self, theta):
        amp_list = list()
        chi2_list = list()
        for fft_pulse, t0 in zip(self.fft_data, self.t0_data):
            param = list(theta) + [t0,]
            mod_pulse = self.model_fun(param, self.time_array)
            fft_template = np.fft.fft(mod_pulse)
            
            amp = rmc.opt_chi2_amp(
                    fft_pulse,
                    fft_template,
                    self.err_array,
                    self.fs
            )
            chi2 = rmc.chi2_freq(
                    fft_pulse,
                    amp * fft_template,
                    self.err_array,
                    self.fs
            )
            
            amp_list.append(amp)
            chi2_list.append(chi2)
            
        return chi2_list, amp_list
    
    def chi2_fun(self, theta):
        chi2_list,_ = self.chi2_amp_fun(theta)
        chi2_tot = np.sum(chi2_list)
        return chi2_tot
   
    
if __name__ == '__main__':
    
    fake_data = np.load('fake_data.npz')

    compa = Comparator(fake_data, model_function)
    
    p0 = [0.1, -2, -1, 5e-3]
    
    chi2_amp_list = compa.chi2_amp_fun(p0)
    
    chi2 = compa.chi2_fun(p0)
    
    
    
    