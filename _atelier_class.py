#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:24:11 2019

@author: misiak
"""
import abc
import numpy as np
import math as m
import red_magic as rmc
import scipy.stats as st


class Model(object):
    """
    Instanciate a model with its:
        - function
        - parameters
        - prior distribution for parameters
        - labels for parameters
        - change of variables
        - inverse change of variable
        - more ?
        
    Abstract class. One should use the subclass Model_analytical_2exp,
    Model_analytical_3exp, Model_ethem, etc..
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        return
    
    @abc.abstractproperty
    def parameters_0(self):
        return
    
    @abc.abstractproperty
    def nparam(self):
        return

    @abc.abstractproperty
    def prior_dist(self):
        return

    @abc.abstractproperty
    def function_origin(self):
        return

    @abc.abstractproperty
    def labels(self):
        return

    def vchange(self, theta_origin):
        theta = theta_origin
        return theta
    
    def inv_vchange(self, theta):
        theta_origin = theta
        return theta_origin

    @abc.abstractmethod
    def function(self):
        return
    
    # hard condition on parameters
    def prior_condition(self, theta):
        return True

    def prior_sampling(self, size=1):
        param_samples = np.array([dist.rvs(size) for dist in self.prior_dist]).T
        return param_samples
    
    def _rejection_sampling_crude(self, size=1):
        sample_array = self.prior_sampling(size=size)
        
        try:
            truth_list = self.prior_condition_broadcastable(sample_array)
        except:
            truth_list = [self.prior_condition(sample) for sample in sample_array]
            
        sample_array = sample_array[truth_list]
        return sample_array        
        
    def rejection_sampling(self, size=1, complete=True, recursion_cap=100):
        """
        Return samples from the prior and apply the prior_condition
        with rejection algorithm (should check if it is rigorous), so 
        some loss in size is to be expected if the parameter "complete" is set
        to False. If True, this function is called recursively until the 
        enough samples are generated.
        """
        sample_array = self._rejection_sampling_crude(size)
        
        if complete:
            
            recursion_counter = 0
            loss_fraction = sample_array.shape[0]/size
            
            while sample_array.shape[0] < size:
                
                recursion_counter += 1
                if recursion_counter > recursion_cap:
                    raise Exception((
                            'Breaking out of while loop in rejection_sampling.'
                            'Number of recursion exceeded the given cap ({})'
                    ).format(recursion_cap))
                
                
                # adapting the size of the new array
                # compensate for the loss of the rejection sampling
                
                addon_size = int((size-sample_array.shape[0])/loss_fraction)
                
                ### for debug
#                print('recursion! sample_array {}'.format(sample_array.shape[0]))
#                print('demanding {}'.format(addon_size))
#                print('counteracting {} loss fraction'.format(loss_fraction))
                
                addon_array = self._rejection_sampling_crude(
                        size=addon_size
                )
                sample_array = np.vstack((sample_array, addon_array))

        return sample_array[:size]
    
class Model_analytical(Model):
    """
    Abstract class for analytical models based on red_magic.model_physics.
    Subclass of abstract Model.
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(Model_analytical, self).__init__()

    @abc.abstractproperty
    def model_origin(self):
        return
    
    @property
    def function_origin(self):
        return self.model_origin.function
    
    @property
    def parameters_0(self):
        return self.vchange(self.model_origin.parameters_0)
    
    def function(self, theta, time_array):
        theta_origin = self.inv_vchange(theta)
        return self.function_origin(theta_origin, time_array)

 
class Model_2exp_log(Model_analytical):
    """ Self explanatory """
    def __init__(self):
        super(Model_2exp_log, self).__init__()        
    
    @property
    def model_origin(self):
        return rmc.Model_pulse('2exp')
    
    @property
    def nparam(self):
        return 5
    
    @property
    def labels(self):
        labs = [
                r"$\epsilon$",
                r"$log_{{10}}(\tau_1)$", 
                r"$log_{{10}}(\tau_2)$",
                r"$log_{{10}}(\tau_{{th}})$",
                r"$t_0$"
        ]
        return labs
    
    @property
    def prior_dist(self):
        dist_list  = [
                st.uniform(0, 1),
                st.norm(-1.5, 1),
                st.norm(-0.5, 1),
                st.norm(-2.5, 1),
                st.norm(0.5, 1),
        ]
        return dist_list
    
    def prior_condition(self, theta):
        eps, t1, t2, s, t0 = theta
        cond_eps = (0 <= eps <= 1)
        cond_tau = (s <= t1 <= t2)
        if (cond_eps and cond_tau):
            return True
        else:
            return False

    def prior_condition_broadcastable(self, theta):
        eps, t1, t2, s, t0 = theta.T
        cond_eps = (0 <= eps)
        cond_eps2 = (eps <= 1)
        cond_s = (s <= t1)
        cond_t1 = (t1 <= t2)
        
        truth_array = cond_eps
        for cond in (cond_eps2, cond_s, cond_t1):
            truth_array = np.logical_and(truth_array, cond)
        
        return truth_array
    
    def vchange(self, theta_origin):
        eps, t1, t2, s, t0 = theta_origin
        logt1 = m.log10(t1)
        logt2 = m.log10(t2)
        logs = m.log10(s)
        return eps, logt1, logt2, logs, t0        
        
    def inv_vchange(self, theta):
        eps, logt1, logt2, logs, t0 = theta
        t1 = 10**logt1
        t2 = 10**logt2
        s = 10**logs
        return eps, t1, t2, s, t0


class Model_3exp_log(Model_analytical):
    """ Self explanatory """
    def __init__(self):
        super(Model_3exp_log, self).__init__()        
    
    @property
    def model_origin(self):
        return rmc.Model_pulse('3exp')
    
    @property
    def nparam(self):
        return 7
    
    @property
    def labels(self):
        labs = [
                r"$\epsilon$",
                r"$\upsilon$",
                r"$log_{{10}}(\tau_1)$", 
                r"$log_{{10}}(\tau_2)$",
                r"$log_{{10}}(\tau_3)$",
                r"$log_{{10}}(\tau_{{th}})$",
                r"$t_0$"
        ]
        return labs
    
    @property
    def prior_dist(self):
        dist_list  = [
                st.uniform(0, 1),
                st.uniform(0, 1),
                st.norm(-1.5, 1),
                st.norm(-0.5, 1),
                st.norm(-2.5, 1),
                st.norm(-2.5, 1),
                st.norm(0.5, 1),
        ]
        return dist_list
    
    def prior_condition(self, theta):
        eps, ups, t1, t2, t3, s, t0 = theta
        cond_eps = (0 <= eps)
        cond_ups = (0 <= ups)
        cond_fraction = (eps+ups <= 1)
        cond_tau = (s <= t1 <= t2 <= t3)
        if (cond_eps and cond_ups and cond_fraction and cond_tau):
            return True
        else:
            return False

    def prior_condition_broadcastable(self, theta):
        eps, ups, t1, t2, t3, s, t0 = theta.T
        cond_eps = (0 <= eps)
        cond_ups = (0 <= ups)
        cond_fraction = (eps+ups <= 1)
        cond_s = (s <= t1)
        cond_t1 = (t1 <= t2)
        cond_t2 = (t2 <= t3)
        
        truth_array = cond_eps
        for cond in (cond_ups, cond_fraction, cond_s, cond_t1, cond_t2):
            truth_array = np.logical_and(truth_array, cond)
        
        return truth_array
        
    def vchange(self, theta_origin):
        eps, ups, t1, t2, t3, s, t0 = theta_origin
        logt1 = m.log10(t1)
        logt2 = m.log10(t2)
        logt3 = m.log10(t3)
        logs = m.log10(s)
        return eps, ups, logt1, logt2, logt3, logs, t0        
        
    def inv_vchange(self, theta):
        eps, ups, logt1, logt2, logt3, logs, t0 = theta
        t1 = 10**logt1
        t2 = 10**logt2
        t3 = 10**logt3
        s = 10**logs
        return eps, ups, t1, t2, t3, s, t0


class Atelier(object):
    """
    Instance containing all that is necessary for a fitting analysis:
        - Data
        - Model
        - Chi2 function
        - lnprior
        - lnprob
        - hard conditions
        - handy methods (rejction sampling, manual fitting)
        
    Should be designed to be called by the mcmc routine function.
    """
    
    def __init__(self, npz_path, model):
        
        # loading the data from the event_livrary npz.
        self._data_path = npz_path
        self._data_npz = np.load(self._data_path)
        
        # creating attributes according to the different data arrays
        attr_lab = ('pulse_data', 'noise_data', 't0_data', 'time_array')
        for lab in attr_lab:
            setattr(self, lab, self._data_npz[lab])

        # computing some basic values
        # sampling frequency
        self.fs = (self.time_array[1]-self.time_array[0])**-1
        # number of samples
        self.nsamples = self.time_array.size
        # window time length
        self.wlen = self.nsamples / self.fs
        
        # some data processing to pass into Fourier space
        self.freq_array = rmc.psd_freq(self.time_array)
        self.fft_data = [np.fft.fft(pulse) for pulse in self.pulse_data]

        self.noise_psd = [rmc.psd(np.fft.fft(noise), self.fs)[1] for noise in self.noise_data]
        self.err_array = np.mean(self.noise_psd, axis=0)

        # model used for the Atelier instance
        if model == '2exp':
            self.model = Model_2exp_log() 
        elif model == '3exp':
            self.model = Model_3exp_log()
        else:
            raise Exception('Model "{}" is not implemented yet.'.format(model))
    
    @property
    def ndim(self):
        # not using t0
        return self.model.nparam - 1 
    
    @property
    def labels(self):
        # not using t0
        return self.model.labels[:-1]

    @property
    def prior_dist(self):
        # not using t0
        return self.model.prior_dist[:-1]
    
    @property
    def parameters_0(self):
        # not using t0
        return self.model.parameters_0[:-1]

    def chi2_amp_fun(self, theta):
        amp_list = list()
        chi2_list = list()
        for fft_pulse, t0 in zip(self.fft_data, self.t0_data):
            param = list(theta) + [t0,]
            mod_pulse = self.model.function(param, self.time_array)
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
    
    def lnprior_list(self, theta):
        lnprior_list = list()
        for p, dist in zip(theta, self.prior_dist):
            lnpdf = dist.logpdf(p)
            lnprior_list.append(lnpdf)
        return lnprior_list
    
    def lnprior(self, theta):
        theta_expanded = list(theta) + [0,]
        bonus_condition = self.model.prior_condition(theta_expanded)
        if bonus_condition:
            return np.sum(self.lnprior_list(theta))
        else:
            return -np.inf

    # The definition of the log probability function
    # We'll also use the "blobs" feature to track the "log prior" for each step
    def logprob(self, theta):
        log_prior = self.lnprior(theta)
        log_prob = -0.5 * self.chi2_fun(theta) + log_prior
    #    log_prob = log_prior
        return log_prob, log_prior
    
    def init_walkers(self, nwalkers):
        #not using t0
        return self.model.rejection_sampling(nwalkers)[:,:-1]


# =============================================================================
# ALMOST THERE, DO NOT GIVE UP ALMOST THERE, and it looks beautiful :3
# =============================================================================

# for debug
if __name__ == '__main__':
    
#    data_path = 'archive/2exp_fake/fake_data.npz'
    data_path = 'archive/ti04l001_RED71_2exp/true_data.npz'    
    
    ato = Atelier(data_path, '3exp')
    
    p0 = ato.parameters_0
    
    print("Number of parameters is:\n{}".format(ato.ndim))
    print("Labels of parameters are:\n{}".format(ato.labels))
    print("Default parameters are:\n{}".format(p0))

    # testing Atelier methods
    chi2_amp_list = ato.chi2_amp_fun(p0)
    chi2 = ato.chi2_fun(p0)
    lnprior_list = ato.lnprior_list(p0)
    lnprior = ato.lnprior(p0)
    logprob = ato.logprob(p0)

    samples = ato.init_walkers(100000)
    
    import matplotlib.pyplot as plt
    plt.close('all')

    import corner
    fig_corner = corner.corner(
        samples,
        bins=50,
        smooth=1,
        fill_contours=True,
        plot_datapoints=True,
        color='grey',
        labels=ato.labels,
    )   
    
