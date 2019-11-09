#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:53:05 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt

import red_magic as rmc


def data_selection(runname, detector, run_dir, save_dir):

    save_path_trigger = '/'.join((save_dir, 'cut_trigger.txt'))
    save_path_noise = '/'.join((save_dir, 'cut_noise.txt'))
    
    reader_processed = rmc.Root_reader(runname, detector, run_dir)
    
    truth_maintenance_trigger = reader_processed.maintenance_cut(
            reader_processed.all.trig.raw.time_stream
    )
    cut_maintenance_trigger = np.nonzero(truth_maintenance_trigger)[0]
    truth_maintenance_noise = reader_processed.maintenance_cut(
            reader_processed.all.noise.raw.time_stream
    )
    cut_maintenance_noise = np.nonzero(truth_maintenance_noise)[0]
    
    tree_trig = reader_processed.all.trig.raw
    tree_noise = reader_processed.all.noise.raw
    
    
    fig1, ax = plt.subplots(num='Cut trigger selection')
    fig1.suptitle('Cut trigger selection')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('$\chi^2$')
    ax.set_xlabel('Energy [ADU]')
    ax.grid()
    
    line, = ax.plot(
            tree_trig.Energy_OF_h[:,0],
            tree_trig.chi2_OF_h[:, 0],
            ls='none',
            marker='+',
        )    
        
    def funk(x):
        cut_with_maintenance = np.intersect1d(x, cut_maintenance_trigger)
        np.savetxt(
                save_path_trigger,
                cut_with_maintenance,
                header='indexes of the trig event passing the cut',
                fmt='%i'
        )
        plt.close(fig1)
        
        # for debug
        # print(reader_processed.all.trig.raw.chi2_OF_h[x, 0])
    
    selector = rmc.Data_Selector(ax, line, proceed_func=funk)
    plt.show()
    
    
    fig2, ax = plt.subplots(num='Cut noise selection')
    fig2.suptitle('Cut noise selection')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('$\chi^2$')
    ax.set_xlabel('Energy [ADU]')
    ax.grid()
    
    line, = ax.plot(
            tree_noise.Energy_OF[cut_maintenance_noise,0],
            tree_noise.chi2_OF[cut_maintenance_noise, 0],
            ls='none',
            marker='+',
        )    
        
    def funk(x):
        cut_with_maintenance = np.intersect1d(x, cut_maintenance_noise)
        np.savetxt(
                save_path_noise,
                cut_with_maintenance,
                header='indexes of the noise event passing the cut',
                fmt='%i'
        )
        plt.close(fig2)
    
    selector = rmc.Data_Selector(ax, line, proceed_func=funk)
    
    plt.show()
    
    
    #### SAVING TRUE DATA
    reader_trig = rmc.Root_reader_proto(runname, detector, run_dir, 'trigger')
    reader_noise = rmc.Root_reader_proto(runname, detector, run_dir, 'noise')
    
    cut_trigger = np.loadtxt(save_path_trigger, dtype=int)
    cut_noise = np.loadtxt(save_path_noise, dtype=int)
    
    ### DATA OF INTEREST
    # time array
    fs = reader_processed.all.run_tree.freq
    wlen = reader_processed.all.run_tree.TimeWindow_Heat[0, 0]
    time_array = np.arange(0, wlen, fs**-1)
    
#    freq_array = rmc.psd_freq(time_array)
    
    # pulse
    pulse_data = reader_trig.all.tree.Trace_Heat_A_Raw[cut_trigger]
    
    # noise
    noise_data = reader_noise.all.tree.Trace_Heat_A_Raw[cut_noise]
    
    # t0 pulse
    nepal_shift = 0.00125
    t0_data = reader_processed.all.trig.raw.Time_OF_h[cut_trigger] - nepal_shift
    
    ### SAVING DATA
    
    np.savez(
            '/'.join((save_dir, 'true_data')),
            pulse_data=pulse_data,
            noise_data=noise_data,
            t0_data=t0_data,
            time_array=time_array
    )


# for debug
if __name__ == '__main__':
    
    plt.close('all')
    
    ### RAW DATA
    runname = 'ti04l001'
    detector = 'RED71'
    run_dir = '/home/misiak/Data/data_run59'
    save_dir = '/home/misiak/projects/pulse_fitting/archive'

    data_selection(runname, detector, run_dir, save_dir)
