#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:38 2024

@author: MCR

Stuff.
"""

import copy
import emcee
import numpy as np
from scipy.stats import norm, truncnorm

from exouprf.model import Model


class Dataset:
    """Primary exoUPRF class. Stores a set of light curve observations and
    performs light curve fits.
    """

    def __init__(self, input_parameters, t, linear_regressors=None,
                 observations=None, gp_regressors=None, ld_model='quadratic',
                 silent=False):
        """Initialize the Dataset class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters and values. Should have form
            {parameter: value}.
        t : dict
            Dictionary of timestamps for each instrument. Should have form
            {instrument: times}.
        linear_regressors : dict
            Dictionary of regressors for linear models. Should have form
            {instrument: regressors}.
        observations : dict
            Dictionary of observed data. Should have form
            {instrument: {flux: values, flux_err: values}}
        gp_regressors : dict
            Dictionary of regressors for Gaussian Process models. Should have
            form {instrument: regressors}.
        ld_model : str
            Limb darkening model identifier
        silent : bool
            If True, do not print any outputs.
        """

        # Initialize easy attributes.
        self.t = t
        self.ld_model = ld_model
        self.observations = observations
        self.linear_regressors = linear_regressors
        self.gp_regressors = gp_regressors
        self.pl_params = input_parameters
        self.silent = silent
        self.mcmc_sampler = None
        self.flux_decomposed = None
        self.flux = None

        for param in self.pl_params:
            dist = self.pl_params[param]['distribution']
            if dist == 'fixed':
                self.pl_params[param]['function'] = None
            elif dist == 'uniform':
                self.pl_params[param]['function'] = logprior_uniform
            elif dist == 'loguniform':
                self.pl_params[param]['function'] = logprior_loguniform
            elif dist == 'normal':
                self.pl_params[param]['function'] = logprior_normal
            elif dist == 'truncated_normal':
                self.pl_params[param]['function'] = logprior_truncatednormal
            else:
                msg = 'Unknown distribution {0} for parameter ' \
                      '{1}'.format(dist, param)
                raise ValueError(msg)

    def fit(self, sampler='mcmc', mcmc_start=None, mcmc_steps=10000):

        if sampler == 'mcmc':
            msg = 'Starting positions must be provided for MCMC sampling.'
            assert mcmc_start is not None, msg


            log_prob_args = (self.pl_params, self.t, self.observations,
                             self.linear_regressors, self.gp_regressors,
                             self.ld_model)

            mcmc_sampler = fit_emcee(mcmc_start, log_probability,
                                     silent=self.silent, mcmc_steps=mcmc_steps,
                                     log_probability_args=log_prob_args)
            self.mcmc_sampler = mcmc_sampler


def fit_emcee(initial_pos, log_prob, silent=False, mcmc_steps=10000,
              log_probability_args=None):
    """

    Parameters
    ----------
    initial_pos
    log_prob
    silent
    mcmc_steps
    log_probability_args

    Returns
    -------

    """

    nwalkers, ndim = initial_pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    args=log_probability_args)
    output = sampler.run_mcmc(initial_pos, mcmc_steps, progress=not silent)

    return sampler


def set_logprior(theta, param_dict):
    """

    Parameters
    ----------
    theta
    param_dict

    Returns
    -------

    """

    logprior = 0
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        thisprior = param_dict[param]['function'](theta[pcounter],
                                                  param_dict[param]['value'])
        logprior += thisprior
        pcounter += 1

    return logprior


def log_likelihood(theta, param_dict, time, observations,
                   linear_regressors=None, gp_regressors=None,
                   ld_model='quadratic'):
    """

    Parameters
    ----------
    theta
    param_dict
    time
    observations
    linear_regressors
    gp_regressors
    ld_model

    Returns
    -------

    """

    log_like = 0
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        param_dict[param]['value'] = (theta[pcounter])
        pcounter += 1

    thismodel = Model(param_dict, time, linear_regressors=linear_regressors,
                      observations=observations, gp_regressors=gp_regressors,
                      ld_model=ld_model, silent=True)
    thismodel.compute_lightcurves()
    for inst in observations.keys():
        mod = thismodel.flux_decomposed[inst]['total']
        dat = observations[inst]['flux']
        t = time[inst]
        err = param_dict['sigma_{}'.format(inst)]['value']
        log_like -= 0.5 * np.log(2 * np.pi * err**2) * len(t)
        log_like -= 0.5 * np.sum((dat - mod)**2 / err**2)

    return log_like


def log_probability(theta, param_dict, time, observations,
                    linear_regressors=None, gp_regressors=None,
                    ld_model='quadratic'):
    """

    Parameters
    ----------
    theta
    param_dict
    time
    observations
    linear_regressors
    gp_regressors
    ld_model

    Returns
    -------

    """

    lp = set_logprior(theta, param_dict)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, copy.deepcopy(param_dict), time, observations,
                        linear_regressors, gp_regressors, ld_model)
    log_prob = lp + ll

    return log_prob


def logprior_uniform(x, hyperparams):
    """Evaluate uniform log prior.
    """

    low_bound, up_bound = hyperparams
    if low_bound <= x <= up_bound:
        return np.log(1 / (up_bound - low_bound))
    else:
        return -np.inf


def logprior_loguniform(x, hyperparams):
    """Evaluate log-uniform log prior.
    """

    low_bound, up_bound = hyperparams
    if low_bound <= x <= up_bound:
        return np.log(1 / (x * (np.log(up_bound) - np.log(low_bound))))
    else:
        return -np.inf


def logprior_normal(x, hyperparams):
    """Evaluate normal log prior.
    """

    mu, sigma = hyperparams
    return np.log(norm.logpdf(x, loc=mu, scale=sigma))


def logprior_truncatednormal(x, hyperparams):
    """Evaluate trunctaed normal log prior.
    """

    mu, sigma, low_bound, up_bound = hyperparams
    return np.log(truncnorm.ppf(x, (low_bound - mu) / sigma,
                                (up_bound - mu) / sigma, loc=mu, scale=sigma))


