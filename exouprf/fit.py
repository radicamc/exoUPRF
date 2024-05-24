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

from exouprf.model import Model


class Dataset:

    def __init__(self, input_parameters, t, linear_regressors=None,
                 observations=None, gp_regressors=None, ld_model='quadratic',
                 silent=False):

        self.t = t
        self.ld_model = ld_model
        self.flux_decomposed = None
        self.flux = None
        self.observations = observations
        self.linear_regressors = linear_regressors
        self.gp_regressors = gp_regressors
        self.pl_params = input_parameters
        self.mcmc_sampler = None
        self.silent = silent

        for param in self.pl_params:
            dist = self.pl_params[param]['distribution']
            if dist == 'fixed':
                self.pl_params[param]['function'] = None
            elif dist == 'uniform':
                self.pl_params[param]['function'] = logprior_uniform
            else:
                msg = 'Unknown distribution {0} for parameter ' \
                      '{1}'.format(dist, param)
                raise ValueError(msg)

    def fit(self, sampler='mcmc', mcmc_start=None, mcmc_steps=10000,
            silent=False):

        if sampler == 'mcmc':
            msg = 'Starting positions must be provided for MCMC sampling.'
            assert mcmc_start is not None, msg

            log_prob_args = (self.pl_params, self.t, self.observations,
                             self.linear_regressors, self.gp_regressors,
                             self.ld_model)
            mcmc_sampler = fit_emcee(mcmc_start, log_probability,
                                     silent=silent, mcmc_steps=mcmc_steps,
                                     log_probability_args=log_prob_args)
            self.mcmc_sampler = mcmc_sampler


def fit_emcee(initial_pos, log_prob, silent=False, mcmc_steps=10000,
              log_probability_args=None):

    nwalkers, ndim = initial_pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    args=log_probability_args)
    output = sampler.run_mcmc(initial_pos, mcmc_steps, progress=not silent)

    return sampler


def set_logprior(theta, param_dict):

    logprior = 0
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        v1, v2 = param_dict[param]['value']
        thisprior = param_dict[param]['function'](theta[pcounter], v1, v2)
        logprior += thisprior
        pcounter += 1

    return logprior


def log_likelihood(theta, param_dict, time, observations,
                   linear_regressors=None, gp_regressors=None,
                   ld_model='quadratic'):

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

    lp = set_logprior(theta, param_dict)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, copy.deepcopy(param_dict), time, observations,
                        linear_regressors, gp_regressors, ld_model)
    log_prob = lp + ll

    return log_prob


def logprior_uniform(x, low_bound, up_bound):

    if low_bound <= x <= up_bound:
        return np.log(1 / (up_bound - low_bound))
    else:
        return -np.inf
