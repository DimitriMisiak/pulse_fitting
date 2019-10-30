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


filename = "archive/cca_true_complete.h5"
reader = emcee.backends.HDFBackend(filename, read_only=True)

nwalkers, ndim = reader.shape
#tau = reader.get_autocorr_time()
tau = 2
burnin = int(2*np.max(tau))
#burnin=5000
thin = int(0.5*np.min(tau))
#samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

samples = reader.get_chain(discard=burnin, thin=thin)


log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

acceptance = reader.accepted / reader.iteration

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))
print("acceptance: {0}".format(acceptance))


#all_samples = np.concatenate((
#    samples, log_prob_samples[:, None], log_prior_samples[:, None]
#), axis=1)

#all_samples = np.concatenate((
#    samples[:,:1], np.log10(samples[:,1:3]), samples[:,3:], log_prob_samples[:, None], log_prior_samples[:, None]
#), axis=1)


labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
labels += ["log prob", "log prior"]

#corner.corner(all_samples, labels=labels);
#
#corner.corner(
#                all_samples,
#                bins=50, smooth=1,
#                labels=labels,
#                quantiles=[0.16, 0.5, 0.84], show_titles=True,
#                truths=all_samples[np.argmax(log_prob_samples)],
#                title_kwargs={"fontsize": 12}
#        )
#
#plt.show()


labs = ['$\\theta_{1}$',
 '$\\theta_{2}$',
 '$\\theta_{3}$',
 '$\\theta_{4}$',]

plt.close('all')

    

# =============================================================================
# AUTOCORR TIME
# =============================================================================

try:
    autocorr = np.loadtxt('autocorr_time.txt')
    y = autocorr
    index = len(y)
    
    n = 100*np.arange(1, index+1)
    
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
except:
    print('The file autocorr_time.txt cannot be found.')


# =============================================================================
# ACCEPTANCE PLOT
# =============================================================================
fig_acceptance = plt.figure('ACCEPTANCE FRACTION')
plt.bar(np.arange(acceptance.shape[0]), acceptance)
# acceptance fraction cut
tracc = (0.2, 0.8)
for thresh in tracc:
    plt.axhline(thresh, color='r', ls='--')
plt.xlabel('Marker Chain Index')
plt.ylabel('Acceptance fraction')
plt.ylim(0., 1.)
plt.tight_layout()


### acceptance cut

cut_acc = acceptance < tracc[0]

#ind = np.where(np.logical_or(acc < tracc[0], acc > tracc[1]))
#bam = chain[ind]
ind = np.nonzero(cut_acc)
samples = np.delete(samples, ind, axis=1)
samples = np.reshape(samples, (-1, ndim))
#samples = np.ravel(samples)
#lnprob = np.delete(lnprob, ind, axis=0)
#
#print('shape chain: {}'.format(chain.shape) )
#print('shape lnprob: {}'.format(lnprob.shape) )

# =============================================================================
# CONVERGENCE PLOT
# =============================================================================

fig, axes = plt.subplots(ndim+2, figsize=(10, 7), sharex=True)
chain = reader.get_chain()

for j in range(nwalkers):
    color='k'
    if cut_acc[j]:
        color = 'r'
        print(j)
    for i in range(ndim):
        ax = axes[i]
        
        ax.plot(np.arange(reader.iteration)+1, chain[:, j, i], color)
    #    ax.set_xlim(0, len(chain))
        ax.set_ylabel(labs[i])
    #    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].plot(np.arange(reader.iteration)+1, reader.get_log_prob()[:,j], color)
    
    
    axes[-2].plot(np.arange(reader.iteration)+1, reader.get_blobs()[:,j], color)


axes[-1].set_ylabel('lnprob')
axes[-2].set_ylabel('lnprior')
axes[-1].set_xlabel("step number")

for ax in axes:
    ax.set_xscale('log')

### convergence cut
    
#### CONVERGENCE plot
#fig_convergence, ax = plt.subplots(ndim+1, 1, sharex=True, figsize=(7, 8),
#                       num='CONVERGENCE')
#ax[-1].set_xlabel('Iterations')
#ax[-1].set_yscale('log')
#ax[-1].set_xscale('log')
#for a, l in zip(ax, labels + ('lnprob',)):
#
#    a.set_ylabel(l)
#    a.grid()
#
## loop over the parameters
#for n in range(ndim):
#    if scale == 'log' :
#        ax[n].set_yscale('log')
#    if len(bam) > 0:
#        # plotting the chains discarded by the acceptance cut
#        ax[n].plot(bam[:, :, n].T, color='r', lw=1., alpha=0.4)
#
##    # convergence cut with mean
##    lnlncut = np.mean(np.log10(-lnprob))
##    burnin_list = list()
##    for lnk in lnprob:
##        try:
##            burn = np.where(np.log10(-lnk) > lnlncut)[0][-1] + 100
##        except:
##            burn = 0
##        burnin_list.append(burn)
##
##    ax[-1].axhline(np.power(10,lnlncut), color='r')
#
## convergence cut with best prob
#lncut = 1.5 * lnprob.max()
#burnin_list = list()
#for lnk in lnprob:
#    try:
#        burn = np.where(lnk <  lncut)[0][-1] + 100
#    except:
#        print('Could not apply convergence cut properly')
#        burn = 0
#    burnin_list.append(burn)
#
#ax[-1].axhline(-lncut, color='r')
#
## plotting the log10(-lnprob) array and the cut threshold
#ax[-1].plot(-lnprob.T, color='k')
#
#chain_ok_list = list()
## loop over the chains
#for chk, brn, lnk in zip(chain, burnin_list, lnprob):
#
#    # iterations array
#    ite = range(chk.shape[0])
#
#    # converged chain and saving it
#    ite_ok = ite[brn:]
#    chk_ok = chk[brn:, :]
#    lnk_ok = lnk[brn:]
#    chain_ok_list.append(chk_ok)
#
#    # not converged chain
#    ite_no = ite[:brn]
#    chk_no = chk[:brn, :]
#
#    # loop over the parameters
#    for n in range(ndim):
#
#        # plotting the accepted chain and their respective burnin
#        ax[n].plot(ite_ok, chk_ok[:,n].T, color='b', lw=1., alpha=1.)
#        ax[n].plot(ite_no, chk_no[:,n].T, color='k', lw=1., alpha=0.4)
#        ax[n].scatter([0], chk[0, n], color='r', marker='o')
#
#    # plotting converged chain lnprob
#    ax[-1].plot(ite_ok, -lnk_ok.T, color='b')
#
#fig_convergence.tight_layout(h_pad=0.0)
#
## samples = reduce(lambda a,b: np.append(a,b, axis=0), chain_ok_list)
#samples = np.vstack(chain_ok_list)
#
#best_ind = np.unravel_index(lnprob.argmax(), lnprob.shape)
#best_chi2 = -2 * lnprob[best_ind]
#xopt = chain[best_ind]
#
#
### CORRELATION plot
fig_correlation, ax = plt.subplots(2, sharex=True, num='CORRELATION')
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


### CORNER plot
fig_corner = corner.corner(
        samples,
        bins=50, smooth=1,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
#            truths=xopt,
        title_kwargs={"fontsize": 12}
)
fig_corner.tight_layout()

# quantiles of the 1d-histograms
inf, med, sup = np.percentile(samples, [16, 50, 84], axis=0)

# Analysis end message
print("MCMC results :")
for n in range(ndim):
    print(labels[n]+'= {:.2e} + {:.2e} - {:.2e}'.format(
        med[n], sup[n]-med[n], med[n]-inf[n]
    ))
#for n in range(ndim):
#    print(labels[n]+'\in [{:.3e} , {:.3e}] with best at {:.3e}'.format(
#            inf[n], sup[n], xopt[n]
#    ))
#if not np.all(np.logical_and(inf<xopt, xopt<sup)):
#    print('Optimal parameters out the 1-sigma range ! Good luck fixing that :P')
#
#print('Chi2 = {}'.format(best_chi2))



#mcmc_results(ndim, chain, lnprob, acc, tuple(labs))



plt.show()

