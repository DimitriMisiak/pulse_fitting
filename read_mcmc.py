#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:54:57 2019

@author: misiak
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import emcee
import corner
import scipy.stats
import math
import red_magic as rmc

plt.close('all')
# =============================================================================
# GETTING THE FILES
# =============================================================================
SAVE_DIR = '/home/misiak/Analysis/pulse_fitting/mcmc_output/ti13l002_RED71_3exp'
#SAVE_DIR = '/home/misiak/Analysis/pulse_fitting/mcmc_output/ti12l000_RED71_2exp'

h5_path = "{}/mcmc_output.h5".format(SAVE_DIR)
cfg_path = "{}/mcmc_config.json".format(SAVE_DIR)
autocorr_path = "{}/autocorr_time.txt".format(SAVE_DIR)

reader = emcee.backends.HDFBackend(h5_path, read_only=True)
with open(cfg_path, 'r') as cfg:
    config = json.load(cfg)

# =============================================================================
# EXTRACTING THE DATA
# =============================================================================
    
nwalkers, ndim = reader.shape
acceptance = reader.accepted / reader.iteration

labels = config['Parameters']['label']

try:
    tau = reader.get_autocorr_time()
except:
    # happens if the mcmc is ended before completion
    # this except is to allow some debug
    print('No thinning of burnin, BE CAREFUL /!\ ')
    tau = 2

burnin = int(2*np.max(tau))
#burnin = 20000
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


# =============================================================================
# AUTOCORR TIME
# =============================================================================
try:
    autocorr = np.loadtxt(autocorr_path)
    y = autocorr
    index = len(y)
    
    n = 100*np.arange(1, index+1)
    plt.figure("autocorr time plot")
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
except:
    print('The file autocorr_time.txt cannot be found.')

# =============================================================================
# ACCEPTANCE CUT & PLOT
# =============================================================================
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

fig_acceptance = plt.figure('ACCEPTANCE FRACTION')
barlist = plt.bar(np.arange(acceptance.shape[0]), acceptance)

for i in ind_acc:
    barlist[i].set_color('k')

for thresh in (acc_inf, acc_sup):
    plt.axhline(thresh, color='r', ls='--')
plt.xlabel('Markov Chain Index')
plt.ylabel('Acceptance fraction')
plt.ylim(0., 1.)
plt.tight_layout()

# =============================================================================
# CONVERGENCE PLOT
# =============================================================================

fig, axes = plt.subplots(ndim+2, figsize=(10, 7), sharex=True)

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

# =============================================================================
# CORRELATION PLOT
# =============================================================================
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

# =============================================================================
# CORNER PLOT
# =============================================================================
### CORNER plot
chain_flat = np.reshape(chain_acc, (-1, ndim))

arg_opt = np.unravel_index(np.argmax(log_prob_samples), log_prob_samples.shape)
xopt = chain_acc[arg_opt]

nsamples = chain_flat.shape[0]

xinf, xmed, xsup = np.percentile(chain_flat, [16, 50, 84], axis=0)
sig_inf = xmed - xinf
sig_sup = xsup - xmed


#dist_list = [getattr(scipy.stats, dist) for dist in config['Prior']['distribution']]
dist_list = config['Prior']['distribution']
arg1_list = config['Prior']['arg1']
arg2_list = config['Prior']['arg2']

nbins = 50

range_corner = [(e.min(), e.max()) for e in chain_flat.T]


def init_walkers(nwalkers):
    coord_list = list()
    for dist, arg1, arg2 in zip(dist_list, arg1_list, arg2_list):
        coord = rmc.rvs(arg1, arg2, dist, size=(nwalkers, 1))
        coord_list.append(coord)
    return np.concatenate(coord_list, axis=1)


A = init_walkers(nsamples*100)

fig_corner = corner.corner(
        A[A[:,2]>-1.40],
        bins=nbins,
        smooth=1,
        fill_contours=True,
        plot_datapoints=True,
        color='grey',
        range=range_corner,
)

corner.corner(
        chain_flat,
        bins=nbins,
        smooth=1,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        fig=fig_corner,
        truths=xopt,
        plot_datapoints=True,
        fill_contours=True,
        use_math_text=True,
        color='blue',
        truth_color='crimson',
#        show_titles=True,
#        title_fmt=".2g",        
#        title_kwargs={"fontsize": 10},
        range=range_corner,
)

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

    text = r"${0}=\left({{{1:.{p}f}}}_{{-{2}}}^{{+{3}}} \right) \times 10^{{{4}}}$"
    ax.set_title(text.format(labels[i], med, s_inf, s_sup, med_power, p=ndecimal),
                 fontsize=10)
    
    
##    xinf, xsup = np.min(patch_xy[:,0]), np.max(patch_xy[:,0])
##    x_array = np.linspace(xinf, xsup, 100)
#    
#    # getting the pdf array with appropriate (rigourous?) normalization
#    bin_size = (xsup-xinf)/nbins
#    pdf_array = dist.pdf(x_array, arg1, arg2) * nsamples * bin_size
#    
#    ax.plot(x_array, pdf_array, 'r')


## plotting prior function on histogramm (diag axes)
#for i in range(ndim):
#    ax = axes_diag[i]
#    
#    # getting the axes limit
#    axes_lim.append(ax.get_xlim())
#    
#    # getting the info on the prior distribution
#    dist = dist_list[i]
#    arg1 = arg1_list[i]
#    arg2 = arg2_list[i]
#    
#    # getting the bin array
#    patch_xy = ax.patches[0].get_xy()
#    xinf, xsup = np.min(patch_xy[:,0]), np.max(patch_xy[:,0])
#    x_array = np.linspace(xinf, xsup, 100)
#    
#    # getting the pdf array with appropriate (rigourous?) normalization
#    bin_size = (xsup-xinf)/nbins
#    pdf_array = dist.pdf(x_array, arg1, arg2) * nsamples * bin_size
#    
#    ax.plot(x_array, pdf_array, 'r')
#
#
## plotting prior contour on 2d histrogramm (lower triangle axes)
## getting the row and column indexes of the 2d hist axes
#row_ind, col_ind = np.tril_indices(ndim, k=-1)
#for i,j in zip(row_ind, col_ind):
#    ax = axes_corner[i,j]
#
#    x_array = np.linspace(*ax.get_xlim(), 50)
#    y_array = np.linspace(*ax.get_ylim(), 50)
#    
#    x_mesh, y_mesh = np.meshgrid(x_array, y_array)
#
#    dist_x = dist_list[j]
#    arg1_x = arg1_list[j]
#    arg2_x = arg2_list[j]
#    
#    dist_y = dist_list[i]
#    arg1_y = arg1_list[i]
#    arg2_y = arg2_list[i]
#    
#    pdf_fun = lambda x,y: dist_x.pdf(x, arg1_x, arg2_x) + dist_y.pdf(y, arg1_y, arg2_y)
#    
#    pdf_mesh = pdf_fun(x_mesh, y_mesh)
#    print(pdf_mesh)
#    ax.contour(x_mesh, y_mesh, pdf_mesh, 20, cmap='RdGy')



fig_corner.tight_layout()
fig_corner.subplots_adjust(hspace=0, wspace=0)


## quantiles of the 1d-histograms
#inf, med, sup = np.percentile(samples, [16, 50, 84], axis=0)
#
## Analysis end message
#print("MCMC results :")
#for n in range(ndim):
#    print(labels[n]+'= {:.2e} + {:.2e} - {:.2e}'.format(
#        med[n], sup[n]-med[n], med[n]-inf[n]
#    ))
##for n in range(ndim):
##    print(labels[n]+'\in [{:.3e} , {:.3e}] with best at {:.3e}'.format(
##            inf[n], sup[n], xopt[n]
##    ))
##if not np.all(np.logical_and(inf<xopt, xopt<sup)):
##    print('Optimal parameters out the 1-sigma range ! Good luck fixing that :P')
##
##print('Chi2 = {}'.format(best_chi2))
#
#
#
##mcmc_results(ndim, chain, lnprob, acc, tuple(labs))
#
#
#
#plt.show()

