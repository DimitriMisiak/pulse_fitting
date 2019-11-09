#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""

import numpy as np
import emcee

from _atelier_class import Atelier


def mcmc_routine(log_prob, coords, ndim, nwalkers, max_iter, output_dir,
                 **kwargs):
    """
    Launch a mcmc from the given parameters:
        - log_prob function
        - coords, initialization of the walkers
        - ndim, number of parameters
        - nwalkers
        - max_iter, mcmc should stop before when it has converged
        - output_dir
        
    **kwargs arguments passed to the emcee.EnsembleSampler.sample function.
    """
    h5_path = '/'.join((output_dir, 'mcmc_output.h5'))
    tau_path = '/'.join((output_dir, 'autocorr_time.txt'))
    
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = emcee.backends.HDFBackend(h5_path)
    backend.reset(nwalkers, ndim)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_iter)
    
    # This will be useful to testing convergence
    old_tau = np.inf
    
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_iter, **kwargs):
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
        # hard coding the sampler size > 100 *tau and tau_rtol = 0.01
        converged = np.all(tau * 100 < sampler.iteration)
        tau_rtol = 0.01
        converged &= np.all(np.abs(old_tau - tau) / tau < tau_rtol)
        if converged:
            break
        old_tau = tau
    
    np.savetxt(str(tau_path), autocorr[:index])


def mcmc_from_atelier(
        atelier_instance,
        walkers_per_dim,
        max_iter,
        output_dir,
        **kwargs
    ):
    """
    Launch the mcmc routine, extracting some parameters from the atelier
    instance.
    
    See also: mcmc_routine
    """
    assert isinstance(atelier_instance, Atelier)
    
    # all the info are in the Atelier instance
    logprob_fun = atelier_instance.logprob
    ndim = atelier_instance.ndim
    nwalkers = walkers_per_dim * ndim
    coords = atelier_instance.init_walkers(nwalkers)
    
    mcmc_routine(logprob_fun, coords, ndim, nwalkers, max_iter, output_dir,
                 **kwargs)


if __name__ == '__main__':
    
    output_dir = 'archive'
    
    data_path = 'archive/true_data.npz'
    ato = Atelier(data_path, '2exp')
    
    mcmc_from_atelier(
            ato,
            walkers_per_dim=16,
            max_iter=int(2),
            output_dir=output_dir,
            progress=True
    )
    