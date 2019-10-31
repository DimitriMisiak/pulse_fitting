#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:24:11 2019

@author: misiak
"""

import numpy as np
import math as m
import red_magic as rmc

# 2exp functions
model_pulse_2exp = rmc.Model_pulse('2exp')

def var_change_2exp(param):
    eps, logt1, logt2, tautherm, t0 = param
    t1 = 10**logt1
    t2 = 10**logt2
    return eps, t1, t2, tautherm, t0

def inv_var_change_2exp(param):
    eps, t1, t2, tautherm, t0 = param
    logt1 = m.log10(t1)
    logt2 = m.log10(t2)
    return eps, logt1, logt2, tautherm, t0

def model_function_2exp(param, time_array):
    param_good = var_change_2exp(param)
    return model_pulse_2exp.function(param_good, time_array)

# 3exp function
model_pulse_3exp = rmc.Model_pulse('3exp')

def var_change_3exp(param):
    eps, ups, logt1, logt2, logt3, tautherm, t0 = param
    t1 = 10**logt1
    t2 = 10**logt2
    t3 = 10**logt3
    return eps, ups, t1, t2, t3, tautherm, t0

def inv_var_change_3exp(param):
    eps, ups, t1, t2, t3, tautherm, t0 = param
    logt1 = m.log10(t1)
    logt2 = m.log10(t2)
    logt3 = m.log10(t3)
    return eps, ups, logt1, logt2, logt3, tautherm, t0

def model_function_3exp(param, time_array):
    param_good = var_change_3exp(param)
    return model_pulse_3exp.function(param_good, time_array)


class Comparator(object):
    
    def __init__(self, data, model):
        
        self.data = data
        
        if model == '2exp':
            self.model_instance = model_pulse_2exp
            self.model_fun = model_function_2exp
            self.parameters_0 = inv_var_change_2exp(model_pulse_2exp.parameters_0)[:-1]
        elif model == '3exp':
            self.model_instance = model_pulse_3exp
            self.model_fun = model_function_3exp
            self.parameters_0 = inv_var_change_3exp(model_pulse_3exp.parameters_0)[:-1]
        else:
            raise Exception('Model "{}" is not implemented yet.'.format(model))
        
        attr_lab = ('pulse_data', 'noise_data', 't0_data', 'time_array')
        for lab in attr_lab:
            setattr(self, lab, self.data[lab])
       
        self.fs = (self.time_array[1]-self.time_array[0])**-1
        self.nsamples = self.time_array.size
        self.wlen = self.nsamples / self.fs
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
   
# for debug
if __name__ == '__main__':
    
#    data_path = 'archive/2exp_fake/fake_data.npz'
    data_path = 'archive/ti04l001_RED71_2exp/true_data.npz'    
    
    data = np.load(data_path)

    compa = Comparator(data, '3exp')
    
    p0 = compa.parameters_0
    
    chi2_amp_list = compa.chi2_amp_fun(p0)
    
    chi2 = compa.chi2_fun(p0)
    
