#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""

import emcee

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

#%%
import corner

ndim=4
filename = "tutorial.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
#tau = 2
burnin = int(2*np.max(tau))
#burnin=5000
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))
#
all_samples = np.concatenate((
    samples, log_prob_samples[:, None], log_prior_samples[:, None]
), axis=1)

#all_samples = np.concatenate((
#    samples[:,:1], np.log10(samples[:,1:3]), samples[:,3:], log_prob_samples[:, None], log_prior_samples[:, None]
#), axis=1)


labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
labels += ["log prob", "log prior"]

#corner.corner(all_samples, labels=labels);

corner.corner(
                all_samples,
                bins=50, smooth=1,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                truths=all_samples[np.argmax(log_prob_samples)],
                title_kwargs={"fontsize": 12}
        )

plt.show()

