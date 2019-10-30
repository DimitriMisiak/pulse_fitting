#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""

import numpy as np
import emcee
import red_magic as rmc

from comparator_class import Comparator

def run_mcmc(
        data_path,
        h5_path,
        model,
        pinit,
        prior_dist=None,
        prior_arg1=None,
        prior_arg2=None,
        walkers_per_dim=10,
        max_n=int(1e5),
        tau_rtol=0.01,
        autocorr_save='autocorr_time.txt',
    ):
    
    
    ### for test
    data = np.load(data_path)
    com = Comparator(data, model)
    
    # processing parameters
    ndim = len(pinit)
    nwalkers = walkers_per_dim * ndim
    
    if prior_dist is None:
        prior_dist = ('norm',)*ndim
    if prior_arg1 is None:
        prior_arg1 = pinit
    if prior_arg2 is None:
        prior_arg2 = [abs(0.1*p) for p in pinit]
        
    
    def lnprior_list(theta):
        lnprior_list = list()
        for p, dist, arg1, arg2 in zip(theta, prior_dist, prior_arg1, prior_arg2):
            lnpdf = rmc.logpdf(p, arg1, arg2, dist)
            lnprior_list.append(lnpdf)
        return lnprior_list
    
    def lnprior(theta):
        return np.sum(lnprior_list(theta))
    
    
    def init_walkers(nwalkers):
        coord_list = list()
        for dist, arg1, arg2 in zip(prior_dist, prior_arg1, prior_arg2):
            coord = rmc.rvs(arg1, arg2, dist, size=(nwalkers, 1))
            coord_list.append(coord)
        return np.concatenate(coord_list, axis=1)
    
    coords = init_walkers(nwalkers)
    
    
    # The definition of the log probability function
    # We'll also use the "blobs" feature to track the "log prior" for each step
    def log_prob(theta):
        log_prior = lnprior(theta)
        log_prob = -0.5 * com.chi2_fun(theta) + log_prior
    #    log_prob = log_prior
        return log_prob, log_prior
    
    
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = emcee.backends.HDFBackend(h5_path)
    backend.reset(nwalkers, ndim)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    
    # This will be useful to testing convergence
    old_tau = np.inf
    
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
    
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
    
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < tau_rtol)
        if converged:
            break
        old_tau = tau
    
    if str(autocorr_save):
        np.savetxt(str(autocorr_save), autocorr[:index])


if __name__ == '__main__':
    
    data_path = 'archive/true_data.npz'
    h5_path = "archive/mcmc_output.h5"
    
    pinit = [0.262, -2.006, -1.222, 0.005252]
    prior_dist = ['norm', 'norm', 'norm', 'norm']
    prior_arg1 = [0.262, -2.006, -1.222, 0.005252]
    prior_arg2 = [0.0262, 0.2006, 0.1222, 0.0005252]
    
    run_mcmc(data_path, h5_path, '2exp', pinit,
             prior_dist, prior_arg1, prior_arg2,
             walkers_per_dim=16,
             max_n=200,
             tau_rtol=0.02,
             autocorr_save='archive/autocorr.txt')
