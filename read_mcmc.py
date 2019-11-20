#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import emcee
import corner
import scipy.stats
import math
import red_magic as rmc
import os

plt.close('all')

def plot_autocorr(y_data, title_addon=None):
    y = y_data
    index = len(y)
    
    title = 'Autocorrelation Time Plot'
    if title_addon:
        title += '\n{}'.format(title_addon)
    
    n = 100*np.arange(1, index+1)
    fig = plt.figure("autocorr time plot")
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.title(title)
    plt.grid()
    plt.tight_layout()


    return fig

def plot_acceptance(acceptance, acc_inf, acc_sup, title_addon=None):

    title = 'Acceptance Fraction Plot'
    if title_addon:
        title += '\n{}'.format(title_addon)
    
    fig = plt.figure('ACCEPTANCE FRACTION')
    barlist = plt.bar(np.arange(acceptance.shape[0]), acceptance)
    
    for i in ind_acc:
        barlist[i].set_color('k')
    
    for thresh in (acc_inf, acc_sup):
        plt.axhline(thresh, color='r', ls='--')
    plt.xlabel('Markov Chain Index')
    plt.ylabel('Acceptance fraction')
    plt.ylim(0., 1.)
    plt.title(title)
    
    plt.grid(axis='y')
    plt.tight_layout()


    return fig

def plot_convergence_full(title_addon=None):
    
    title = 'Converence plot FULL Plot'
    if title_addon:
        title += '\n{}'.format(title_addon)    
    
    fig, axes = plt.subplots(ndim+2, figsize=(10, 7), sharex=True)
    fig.suptitle(title)
    #iteration_array = np.arange(1, chain.shape[0]+1, 1)
    iteration_array = np.arange(1, reader.iteration +1, 1)
    
    for j in range(nwalkers):
        # managing according to acceptance cut
        color='slateblue'
        if cut_acc[j]:
            color = 'k'
        
        line_style = {'color':color, 'lw':1, 'alpha':0.3}
        
        # plotting the chain for each parameter
        for i in range(ndim):
            ax = axes[i]
            ax.set_ylabel(labels[i])
            ax.plot(iteration_array, chain_raw[:, j, i], **line_style)
        
        # plotting the log prob
        axes[-1].plot(iteration_array, log_prob_raw[:,j], **line_style)
        
        # plotting the log prior
        axes[-2].plot(iteration_array, log_prior_raw[:,j], **line_style)
    
    axes[-1].set_ylabel('logprob')
    axes[-2].set_ylabel('logprior')
    axes[-1].set_xlabel("step number")
    
    # representing the burnin
    for ax in axes:
        ax.axvline(burnin, color='r')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    
    return fig

def plot_convergence_roi(title_addon=None):
    
    title = 'Converence plot ROI Plot'
    if title_addon:
        title += '\n{}'.format(title_addon)    
    
    fig, axes = plt.subplots(ndim+2, figsize=(10, 7), sharex=True)
    fig.suptitle(title)
    #iteration_array = np.arange(1, chain.shape[0]+1, 1)
    iteration_array = np.arange(burnin+1, reader.iteration +1, 1)
    
    for j in range(nwalkers):
        
        if j in ind_acc:
            continue
        # managing according to acceptance cut
        color='slateblue'
        if cut_acc[j]:
            color = 'k'
        
        line_style = {'color':color, 'lw':1, 'alpha':0.3}
        
        # plotting the chain for each parameter
        for i in range(ndim):
            ax = axes[i]
            ax.set_ylabel(labels[i])
            ax.plot(iteration_array, chain_raw[burnin:, j, i], **line_style)
        
        # plotting the log prob
        axes[-1].plot(iteration_array, log_prob_raw[burnin:,j], **line_style)
        
        # plotting the log prior
        axes[-2].plot(iteration_array, log_prior_raw[burnin:,j], **line_style)
    
    axes[-1].set_ylabel('logprob')
    axes[-2].set_ylabel('logprior')
    axes[-1].set_xlabel("step number")
    
    # representing the burnin
    for ax in axes:
        ax.axvline(burnin, color='r')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    
    return fig


def plot_correlation(title_addon=None):
    
    title = 'Converence plot ROI Plot'
    if title_addon:
        title += '\n{}'.format(title_addon)   
        
    fig, ax = plt.subplots(2, sharex=True, num='CORRELATION')    
    try:
        
        fig.suptitle(title)
        for a in ax:
            a.grid()
            a.set_xscale('log')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Corr p0')
        ax[0].set_ylabel('p0')
        
        for c in chain[:,:,0].T:
            funk = emcee.autocorr.function_1d(c)
            ax[0].plot(c)
            ax[1].plot(funk)
        
        plt.tight_layout()
    
        
    except:
        print('No correlation plot')
        
    return fig 


def plot_corner_prior(title_addon=None):
    A = ato.init_walkers(nsamples*100)
    
    
    fig_corner = corner.corner(
            chain_flat,
            bins=nbins,
            smooth=1,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            truths=xopt,
            plot_datapoints=True,
            fill_contours=True,
            use_math_text=True,
            color='blue',
            truth_color='crimson',
    #        range=range_corner,
    )
    
    corner.corner(
            A,
            bins=nbins,
            smooth=1,
            fill_contours=True,
            plot_datapoints=True,
            color='grey',
            fig=fig_corner,        
    #        range=range_corner,
    )

    title = 'Corner plot with Prior'
    if title_addon:
        title += '\n{}'.format(title_addon)  
    
    fig_corner.suptitle(title)
    # editing the 1d histogram
    axes_corner = np.reshape(fig_corner.axes, (ndim, ndim))
    axes_diag = np.diag(axes_corner)
    for i in range(ndim):
        ax = axes_diag[i]
        
        # resizing the 1d hist of the prior to match the level of the likelihood
        prior_patch, data_patch = ax.patches
        prior_sup = np.max(prior_patch.get_xy()[:,1])
        data_sup = np.max(data_patch.get_xy()[:,1])
        
        coeff_corr = data_sup / prior_sup
        
        prior_xy = prior_patch.get_xy()
        prior_xy[:, 1] *= coeff_corr
        prior_patch.set_xy(prior_xy)
        
        # good presentation of the result
        med = xmed[i]
        s_inf = sig_inf[i]
        s_sup = sig_sup[i]
        
        med_power = math.trunc(math.log10(abs(med))-1)
        
        med /= 10**med_power
        s_inf /= 10**med_power
        s_sup /= 10**med_power
        
        s_inf = "{:.1g}".format(s_inf)
        s_sup = "{:.1g}".format(s_sup)
        
        ndecimal = len( str(min([s_inf, s_sup])).split('.')[-1] )
    
        text = r"{0}$=\left({{{1:.{p}f}}}_{{-{2}}}^{{+{3}}} \right) \times 10^{{{4}}}$"
        ax.set_title(text.format(labels[i], med, s_inf, s_sup, med_power, p=ndecimal),
                     fontsize=10)
    
    
    fig_corner.tight_layout()
    fig_corner.subplots_adjust(hspace=0, wspace=0)

    return fig_corner


def plot_corner_roi(title_addon=None):
    
    fig_corner = corner.corner(
            chain_flat,
            bins=nbins,
            smooth=1,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            truths=xopt,
            plot_datapoints=True,
            fill_contours=True,
            use_math_text=True,
            color='blue',
            truth_color='crimson',
            range=range_corner,
    )

    title = 'Corner plot ROI'
    if title_addon:
        title += '\n{}'.format(title_addon)  
    
    fig_corner.suptitle(title)
    # editing the 1d histogram
    axes_corner = np.reshape(fig_corner.axes, (ndim, ndim))
    axes_diag = np.diag(axes_corner)
    for i in range(ndim):
        ax = axes_diag[i]

        # good presentation of the result
        med = xmed[i]
        s_inf = sig_inf[i]
        s_sup = sig_sup[i]
        
        med_power = math.trunc(math.log10(abs(med))-1)
        
        med /= 10**med_power
        s_inf /= 10**med_power
        s_sup /= 10**med_power
        
        s_inf = "{:.1g}".format(s_inf)
        s_sup = "{:.1g}".format(s_sup)
        
        ndecimal = len( str(min([s_inf, s_sup])).split('.')[-1] )
    
        text = r"{0}$=\left({{{1:.{p}f}}}_{{-{2}}}^{{+{3}}} \right) \times 10^{{{4}}}$"
        ax.set_title(text.format(labels[i], med, s_inf, s_sup, med_power, p=ndecimal),
                     fontsize=10)
    
    
    fig_corner.tight_layout()
    fig_corner.subplots_adjust(hspace=0, wspace=0)

    return fig_corner


def plot_logprob(title_addon=None):
    title = 'Logprobability histogramm'
    if title_addon:
        title += '\n{}'.format(title_addon)  
    
    fig = plt.figure()
    plt.title(title)

    n,bins,p = plt.hist(logprob, bins=100, range=(range_3, logprob.max()),
                        color='silver', label='All')

    plt.hist(logprob[logprob>range_2], bins=bins,
                        color='orange', label='2-sigma')
    
    plt.hist(logprob[logprob>range_1], bins=bins,
                        color='slateblue', label='1-sigma')
    
    plt.legend(loc='upper left')

    
    plt.ylabel('Counts')
    plt.xlabel('Log-Probability')

    plt.tight_layout()

    
    return fig


def plot_fit(title_addon=None):
    title = 'Pulse fitting'
    if title_addon:
        title += '\n{}'.format(title_addon)  
    
    psd_data = [rmc.psd(fft, ato.fs)[1] for fft in ato.fft_data]
    
    nsamples = np.array(psd_data).size
#    nsamples = ato.nsamples
    
    ### PLOT FOR MANUAL FITTING
    fig, axes = plt.subplots(nrows=2, figsize=(10, 7))
    
    pulse_lines = list()
    psd_lines = list()
    for t0, pulse, psd in zip(ato.t0_data, ato.pulse_data, psd_data):
        line0, = axes[0].plot(ato.time_array-t0, pulse, alpha=0.3)
        line1, = axes[1].loglog(ato.freq_array, psd, alpha=0.3)
        pulse_lines.append(line0)
        psd_lines.append(line1)
    
    axes[0].set_ylabel('Pulse')
    axes[0].set_xlabel('Time [s]')
    axes[1].set_ylabel('PSD Pulse')
    axes[1].set_xlabel('Frequency [Hz]')
    for ax in axes:
        ax.grid()
    
    ### preparing for manual fitting
#    pinit = [0.1, -2, -1,  1e-3]
    pinit = ato.parameters_0
    tmod = ato.wlen/2
    
    def manual_fitting_funk(theta):
        pulse_template = ato.model.function(list(theta)+[tmod,], ato.time_array)
        psd_template = rmc.psd(np.fft.fft(pulse_template), ato.fs)[1]
        return [pulse_template - np.mean(pulse_template), psd_template]
    
#    for theta in theta_samples:
#        pulse_template, psd_template = manual_fitting_funk(theta)
#        line0, = axes[0].plot(ato.time_array-tmod, pulse_template, color='g', alpha=0.1)
#        line1, = axes[1].plot(ato.freq_array, psd_template, color='g', alpha=0.1)  
    
    pulse_template, psd_template = manual_fitting_funk(xopt)
    line0, = axes[0].plot(ato.time_array-tmod, pulse_template, color='k')
    line1, = axes[1].plot(ato.freq_array, psd_template, color='k')


    
    
    dc_pulse = [np.mean(pulse) for pulse in ato.pulse_data]

    def callback_fun(theta):
        chi2_list, amp_list = ato.chi2_amp_fun(theta)
        
        # updataing chi2 value
        chi2 = np.sum(chi2_list)
        title = 'Pulse fitting\n$\chi^2$={:.3e} / ddf={:.3e}'.format(chi2, nsamples)
        if title_addon:
            title += '\n{}'.format(title_addon)  
        fig.suptitle(title)
        
        # updating experimental curves
        for line, pulse, dc, amp in zip(pulse_lines, ato.pulse_data, dc_pulse, amp_list):
            line.set_ydata( (pulse-dc)/amp )
    #
        for line, psd, amp in zip(psd_lines, psd_data, amp_list):
            line.set_ydata( psd/amp**2 )
                  
    callback_fun(xopt)
    
    axes[0].set_ylim(pulse_template.min()-0.2*pulse_template.max(), 1.2*pulse_template.max())
    axes[1].set_ylim(0.5*psd_template.min(), 2*psd_template.max())


    return fig

#%%
if __name__ == '__main__':
    
    
    assert len(sys.argv) == 4
    stream, detector, model = sys.argv[1:]   
    
    OUT_DIR = '/home/misiak/Analysis/pulse_fitting/mcmc_output'
    LIB_DIR = '/home/misiak/Analysis/pulse_fitting/event_library'
    
#    stream = 'tg26l008'
#    detector = 'RED70'
#    model = '2exp'
    
    ato_lab = '-'.join((stream, detector, model))
    
    dir_name = '_'.join((stream, detector, model))
    
    dir_name_lite = '_'.join((stream, detector))
    
    out_dir = '/'.join((OUT_DIR, dir_name))
    
    h5_path = "{}/mcmc_output.h5".format(out_dir)
    cfg_path = "{}/config.json".format(out_dir)
    autocorr_path = "{}/autocorr_time.txt".format(out_dir)
    
    with open(cfg_path, 'r') as cfg:
        config = json.load(cfg)
        
    event_dir = config['Selection']['directory']
    
    reader = emcee.backends.HDFBackend(h5_path, read_only=True)
    
    from _atelier_class import Model_2exp_log, Model_3exp_log, Atelier


    if detector == 'fake':
        ato = Atelier('/'.join((LIB_DIR, dir_name_lite, 'fake_data.npz')), model)
    else:
        ato = Atelier('/'.join((LIB_DIR, dir_name_lite, 'true_data.npz')), model)
        
    if model == '2exp':
        model = Model_2exp_log()
    elif model == '3exp':
        model = Model_3exp_log()
    else:
        raise Exception('Not implemented yet.')
 
    ### EXTRACTING THE DATA
    nwalkers, ndim = reader.shape
    acceptance = reader.accepted / reader.iteration
    
    #labels = config['Parameters']['label']
    labels = model.labels
    
    try:
        tau = reader.get_autocorr_time()
        assert not np.any(np.isnan(tau))
        
        burnin = int(2*np.max(tau))
        thin = int(0.5*np.min(tau))
        
    except:
        # happens if the mcmc is ended before completion
        # this except is to allow some debug
        print('No thinning of burnin, BE CAREFUL /!\ ')
        tau = 10
        burnin = int(20*np.max(tau))
        thin = int(0.5*np.min(tau))
    
    chain_raw = reader.get_chain()
    log_prob_raw = reader.get_log_prob()
    log_prior_raw = reader.get_blobs()
    
    chain = reader.get_chain(discard=burnin, thin=thin)
    
    #chain_raw[:,:,1:3] = 10**chain_raw[:,:,1:3]
    #chain[:,:,1:3] = 10**chain[:,:,1:3]
    
    log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin)
    log_prior_samples = reader.get_blobs(discard=burnin, thin=thin)
    
    
    print("ndim: {0}".format(ndim))
    print("labels: {0}".format(labels))
    print("nwalkers: {0}".format(nwalkers))
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("chain shape: {0}".format(chain.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    print("flat log prior shape: {0}".format(log_prior_samples.shape))
    print("mean acceptance: {0}".format(np.mean(acceptance)))

    
    ### acceptance cut
    acc_inf = 0.2
    acc_sup = 0.8
    cut_acc = acceptance < acc_inf 
    
    ind_acc = np.nonzero(cut_acc)[0]
    # deleting the sterile chain
    #chain_acc = np.copy(chain[:, np.logical_not(cut_acc),:])
    chain_acc = np.delete(chain, ind_acc, axis=1)
    log_prob_acc = np.delete(log_prob_samples, ind_acc, axis=1)
    log_prior_acc = np.delete(log_prior_samples, ind_acc, axis=1)
    
    ### CORNER plot
    chain_flat = np.reshape(chain_acc, (-1, ndim))
    
    arg_opt = np.unravel_index(np.argmax(log_prob_samples), log_prob_samples.shape)
    xopt = chain[arg_opt]
    
    nsamples = chain_flat.shape[0]
    
    xinf, xmed, xsup = np.percentile(chain_flat, [16, 50, 84], axis=0)
    
    #not using med, but xopt
    xmed = xopt
    
    sig_inf = xmed - xinf
    sig_sup = xsup - xmed
    
    
    nbins = 50
    
#    range_corner = [(e.min(), e.max()) for e in chain_flat.T]
    range_corner = (np.percentile(chain_flat, [1, 99], axis=0)).T

    logprob = log_prob_acc.flatten()
    
    range_3, range_2, range_1 = np.percentile(logprob, [3, 100-95, 100-68], axis=0)


    pulse_opt = ato.model.function(list(xopt)+[0.25,], ato.time_array)

    ind_sample = np.random.choice(range(logprob.size), 300)
    ind_ok = ind_sample[logprob[ind_sample]<range_1]
    i_array, j_array = np.unravel_index(ind_ok, log_prob_acc.shape)
    
    theta_samples = list()
    for i,j in zip(i_array, j_array):
        theta_samples.append(chain_acc[i,j])
    

    #### PLOT 
    FIG_DIR = '/home/misiak/Analysis/pulse_fitting/results'
    
    fig_dir = '/'.join((FIG_DIR, dir_name))
    
    # super mkdir for the save directories
    path_list = (fig_dir,)
    for path in path_list:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    
    try:
        autocorr = np.loadtxt(autocorr_path)
        fig_autocorr = plot_autocorr(autocorr, ato_lab)
    except:
        print('The file autocorr_time.txt cannot be found.')
        
    fig_acc = plot_acceptance(acceptance, acc_inf, acc_sup, ato_lab)
    
    fig_conv_full = plot_convergence_full(ato_lab)
    
    fig_conv_roi = plot_convergence_roi(ato_lab)
    
    fig_corr = plot_correlation(ato_lab)
    
    fig_corner_prior = plot_corner_prior(ato_lab)
    
    fig_corner_roi = plot_corner_roi(ato_lab)
    
    fig_logprob = plot_logprob(ato_lab)
    
    fig_fit = plot_fit(ato_lab)
    
    # saving figs
    fig_acc.savefig('/'.join((fig_dir, 'fig_acc.png')))
    fig_conv_full.savefig('/'.join((fig_dir, 'fig_conv_full.png')))
    fig_conv_roi.savefig('/'.join((fig_dir, 'fig_conv_roi.png')))
    fig_corr.savefig('/'.join((fig_dir, 'fig_corr.png')))
    fig_corner_prior.savefig('/'.join((fig_dir, 'fig_corner_prior.png')))
    fig_corner_roi.savefig('/'.join((fig_dir, 'fig_corner_roi.png')))
    fig_logprob.savefig('/'.join((fig_dir, 'fig_logprob.png')))
    fig_fit.savefig('/'.join((fig_dir, 'fig_fit.png')))
    
    raw_xopt = np.vstack((xopt, sig_inf, sig_sup)).T
    np.savetxt('/'.join((fig_dir, 'raw_xopt.txt')), raw_xopt, header='xopt\tsigma_inf\tsigma_sup')
    
    inv_xopt = list()
    for theta in (xopt, xinf, xsup):
        theta_origin = ato.model.inv_vchange(list(theta) + [0,])[:-1]
        inv_xopt.append(theta_origin)
    
    inv_xopt = np.array(inv_xopt).T
    
    fine_xopt = np.zeros(inv_xopt.shape)
    fine_xopt[:,0] = inv_xopt[:,0]
    fine_xopt[:,1] = inv_xopt[:,0] - inv_xopt[:,1] 
    fine_xopt[:,2] = inv_xopt[:,2] - inv_xopt[:,0]     
        
    np.savetxt('/'.join((fig_dir, 'fine_xopt.txt')), fine_xopt, header='xopt\tsigma_inf\tsigma_sup')    
        
        
