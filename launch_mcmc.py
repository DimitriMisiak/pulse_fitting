#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""

import emcee

import numpy as np

import json
import red_magic as rmc
from comparator_class import model_function, Comparator

# parameters to launch mcmc
config_mcmc_path = 'mcmc_config.json'
walkers_per_dim = 10
max_n = 10000
tau_rtol = 0.01
filename = "tutorial.h5"
fake_data = np.load('fake_data.npz')

### for test
com = Comparator(fake_data, model_function)


# mcmc config
with open(config_mcmc_path, 'r') as configfile:
    config =json.load(configfile)
pinit = config['Parameters']['pinit']
prior_arg1 = config['Prior']['arg1']
prior_arg2 = config['Prior']['arg2']
prior_dist = config['Prior']['distribution']

# processing parameters
ndim = len(pinit)
nwalkers = walkers_per_dim * ndim


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
#    log_prob = -0.5 * com.chi2_fun(theta) + log_prior
    log_prob = log_prior
    return log_prob, log_prior


# Set up the backend
# Don't forget to clear it in case the file already exists
backend = emcee.backends.HDFBackend(filename)
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

#%%
    
import matplotlib.pyplot as plt

n = 100*np.arange(1, index+1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");


