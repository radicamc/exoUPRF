#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:38 2024

@author: MCR

Functions for fitting light curve models to data.
"""

import copy
from datetime import datetime
import h5py
import emcee
import numpy as np
import os
from scipy.stats import norm, truncnorm

from exouprf.light_curve_models import LightCurveModel
import exouprf.plotting as plotting
import exouprf.utils as utils


class Dataset:
    """Primary exoUPRF class. Stores a set of light curve observations and
    performs light curve fits.
    """

    def __init__(self, input_parameters, t, lc_model_type,
                 linear_regressors=None, observations=None, gp_regressors=None,
                 ld_model='quadratic',  silent=False):
        """Initialize the Dataset class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters and values. Should have form
            {parameter: value}.
        t : dict
            Dictionary of timestamps for each instrument. Should have form
            {instrument: times}.
        lc_model_type : dict
            Dictionary of light curve models for each planet and instrument.
            Should have form {instrument: {pl: model}}.
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
        self.lc_model = lc_model_type
        self.observations = observations
        self.linear_regressors = linear_regressors
        self.gp_regressors = gp_regressors
        self.pl_params = input_parameters
        self.silent = silent
        self.mcmc_sampler = None
        self.flux_decomposed = None
        self.flux = None
        self.output_file = None

        # For each parameter, get the prior function to be used based on the
        # indicated prior distribution.
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

    def fit(self, output_file, sampler='mcmc', mcmc_start=None,
            mcmc_steps=10000, continue_mcmc=False):

        # Set up and save output file name.
        if output_file[-3:] != '.h5':
            output_file += '.h5'
        self.output_file = output_file

        # For MCMC sampling with emcee.
        if sampler == 'mcmc':
            if continue_mcmc is False:
                msg = 'Starting positions must be provided for MCMC sampling.'
                assert mcmc_start is not None, msg

            # Arguments for the log probability function call.
            log_prob_args = (self.pl_params, self.t, self.observations,
                             self.lc_model, self.linear_regressors,
                             self.gp_regressors, self.ld_model)

            # Initialize and run the emcee sampler.
            mcmc_sampler = fit_emcee(log_probability, initial_pos=mcmc_start,
                                     silent=self.silent, mcmc_steps=mcmc_steps,
                                     log_probability_args=log_prob_args,
                                     output_file=output_file,
                                     continue_run=continue_mcmc)
            self.mcmc_sampler = mcmc_sampler

    def get_param_dict_from_mcmc(self, method='median', burnin=None, thin=15):
        """Reformat MCMC fit outputs into the parameter dictionary format
        expected by Model.

        Parameters
        ----------
        method : str
            Method via which to get best fitting parameters from MCMC chains.
            Either "median" or "maxlike".
        burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain
            length.
        thin : int
            Increment by which to thin chains.

        Returns
        -------
        param_dict : dict
            Dictionary of light curve model parameters.
        """

        param_dict = utils.get_param_dict_from_mcmc(self.output_file,
                                                    method=method,
                                                    burnin=burnin, thin=thin)
        return param_dict

    def get_fit_results_from_mcmc(self, burnin=None, thin=15):
        """Extract MCMC posterior sample statistics (median and 1 sigma bounds)
        for each fitted parameter.

        Parameters
        ----------
        burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain
            length.
        thin : int
            Increment by which to thin chains.

        Returns
        -------
        results_dict : dict
            Dictionary of posterior medians and 1 sigma bounds for each fitted
            parameter.
        """

        results_dict = utils.get_fit_results_from_mcmc(self.output_file,
                                                       burnin=burnin,
                                                       thin=thin)
        return results_dict

    def plot_mcmc_chains(self, labels=None):
        """Plot MCMC chains.

        Parameters
        ----------
        labels : list(str)
            Fitted parameter names.
        """

        plotting.plot_mcmc_chains(self.output_file, labels=labels)

    def make_corner_plot(self, burnin=None, thin=15, labels=None):
        """Make a corner plot of fitted posterior distributions.

        Parameters
        ----------
        burnin : int
            Number of steps to discard as burn in. Defaults to 75% of chain
            length.
        thin : int
            Increment by which to thin chains.
        labels : list(str)
            Fitted parameter names.
        """

        plotting.make_corner_plot(self.output_file, burnin=burnin, thin=thin,
                                  labels=labels)


def fit_emcee(log_prob, output_file, initial_pos=None, continue_run=False,
              silent=False, mcmc_steps=10000, log_probability_args=None):
    """Run a light curve fit via MCMC using the emcee sampler.

    Parameters
    ----------
    log_prob : function
        Callable function to evaluate the fit log probability.
    output_file : str
        File to which to save outputs. If continuing a run, this should also
        be the input file containing the previous MCMC chains.
    initial_pos : ndarray(float), None
        Starting positions for the MCMC sampling.
    continue_run : bool
        If True, continue a run from the state of previous MCMC chains.
    silent : bool
        If True, do not show any progress.
    mcmc_steps : int
        Number of MCMC steps before stopping.
    log_probability_args : tuple
        Arguments for the passed log_prob function.

    Returns
    -------
    sampler : emcee.ensemble.EnsembleSampler
        ecmee sampler.
    """

    # If we want to restart from a previous chain, make sure all is good.
    if continue_run is True:
        # Override any passed initial positions.
        if initial_pos is not None:
            msg = 'continue_run option selected. Ignoring passed initial ' \
                  'positions.'
            print(msg)
            initial_pos = None

    # If we are starting a new run, we want to create the output h5 file
    # and append useful information such as metadata and priors used for
    # the fit.
    if continue_run is False:
        # Create all the metadata for this fit.
        hf = h5py.File(output_file, 'w')
        hf.attrs['Author'] = os.environ.get('USER')
        hf.attrs['Date'] = datetime.utcnow().replace(microsecond=0).isoformat()
        hf.attrs['Code'] = 'exoUPRF'
        hf.attrs['Sampling'] = 'MCMC'

        # Add prior info.
        inputs = log_probability_args[0]
        for i, param in enumerate(inputs.keys()):
            g = hf.create_group('inputs/{}'.format(param))
            g.attrs['location'] = i
            dt = h5py.string_dtype()
            g.create_dataset('distribution',
                             data=inputs[param]['distribution'], dtype=dt)
            g.create_dataset('value', data=inputs[param]['value'])
        hf.close()

        # Initialize the emcee backend.
        backend = emcee.backends.HDFBackend(output_file)
        nwalkers, ndim = initial_pos.shape
        backend.reset(nwalkers, ndim)

    # If we're continuing a run, the metadata should already be there.
    else:
        # Don't reset the backend if we want to continue a run!!
        backend = emcee.backends.HDFBackend(output_file)
        nwalkers, ndim = backend.shape
        print('Restarting fit from file {}.'.format(output_file))
        print('{} steps already completed.'.format(backend.iteration))

    # Do the sampling.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend,
                                    args=log_probability_args)
    output = sampler.run_mcmc(initial_pos, mcmc_steps, progress=not silent)

    return sampler


def set_logprior(theta, param_dict):
    """Calculate the fit prior based on a set of input values and prior
    functions.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.

    Returns
    -------
    log_prior : float
        Result of prior evaluation.
    """

    log_prior = 0
    pcounter = 0
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        thisprior = param_dict[param]['function'](theta[pcounter],
                                                  param_dict[param]['value'])
        log_prior += thisprior
        pcounter += 1

    return log_prior


def log_likelihood(theta, param_dict, time, observations, lc_model,
                   linear_regressors=None, gp_regressors=None,
                   ld_model='quadratic'):
    """Evaluate the log likelihood for a dataset and a given set of model
    parameters.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.
    time : dict
        Dictonary of timestamps corresponding to the observations.
    observations : dict
        Dictionary of observations.
    lc_model : dict
        Dictionary of light curve model calls.
    linear_regressors : dict
        Dictionary of regressors for linear models.
    gp_regressors : dict
        Dictionary of regressors for GP models.
    ld_model : str
        Limb darkening model to use.

    Returns
    -------
    log_like : float
        Result of likelihood evaluation.
    """

    log_like = 0
    pcounter = 0
    # Update the planet parameter dictionary based on current values.
    for param in param_dict:
        if param_dict[param]['distribution'] == 'fixed':
            continue
        param_dict[param]['value'] = (theta[pcounter])
        pcounter += 1

    # Evaluate the light curve model for all instruments.
    thismodel = LightCurveModel(param_dict, time,
                                linear_regressors=linear_regressors,
                                observations=observations,
                                gp_regressors=gp_regressors,
                                ld_model=ld_model, silent=True)
    thismodel.compute_lightcurves(lc_model_type=lc_model)

    # For each instrument, calculate the likelihood.
    for inst in observations.keys():
        mod = thismodel.flux_decomposed[inst]['total']
        dat = observations[inst]['flux']
        t = time[inst]
        err = param_dict['sigma_{}'.format(inst)]['value']
        log_like -= 0.5 * np.log(2 * np.pi * err**2) * len(t)
        log_like -= 0.5 * np.sum((dat - mod)**2 / err**2)

    return log_like


def log_probability(theta, param_dict, time, observations, lc_model,
                    linear_regressors=None, gp_regressors=None,
                    ld_model='quadratic'):
    """Evaluate the log probability for a dataset and a given set of model
    parameters.

    Parameters
    ----------
    theta : list(float)
        List of values for each fitted parameter.
    param_dict : dict
        Dictionary of input parameter values and prior distributions.
    time : dict
        Dictonary of timestamps corresponding to the observations.
    observations : dict
        Dictionary of observations.
    lc_model : dict
        Dictionary of light curve model calls.
    linear_regressors : dict
        Dictionary of regressors for linear models.
    gp_regressors : dict
        Dictionary of regressors for GP models.
    ld_model : str
        Limb darkening model to use.

    Returns
    -------
    log_prob : float
        Result of probability evaluation.
    """

    lp = set_logprior(theta, param_dict)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, copy.deepcopy(param_dict), time, observations,
                        lc_model, linear_regressors, gp_regressors, ld_model)
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
