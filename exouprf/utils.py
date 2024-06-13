#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:35 2024

@author: MCR

Miscellaneous tools.
"""

from datetime import datetime
import h5py
import numpy as np


def fancyprint(message, msg_type='INFO'):
    """Fancy printing statement mimicking logging. Basically a hack to get
    around complications with the STScI pipeline logging.

    Parameters
    ----------
    message : str
        Message to print.
    msg_type : str
        Type of message. Mirrors the jwst pipeline logging.
    """

    time = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    print('{} - exoUPRF - {} - {}'.format(time, msg_type, message))


def get_param_dict_from_mcmc(filename, method='median', burnin=None, thin=15):
    """Reformat MCMC fit outputs into the parameter dictionary format
    expected by Model.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
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

    fancyprint('Importing fitted parameters from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        mcmc = f['mcmc']['chain'][()]
        # Discard burn in and thin chains.
        if burnin is None:
            burnin = int(0.75 * np.shape(mcmc)[0])
        # Cut steps for burn in.
        mcmc = mcmc[burnin:]
        nwalkers, nchains, ndim = np.shape(mcmc)
        # Flatten chains.
        mcmc = mcmc.reshape(nwalkers * nchains, ndim)[::thin]
        # Either get maximum likelihood solution...
        if method == 'maxlike':
            lp = f['mcmc']['log_prob'][()].flatten()[burnin:][::thin]
            ii = np.argmax(lp)
            bestfit = mcmc[ii]
        # ...or take median of samples.
        elif method == 'median':
            bestfit = np.nanmedian(mcmc, axis=0)

        # HDF5 groups are in alphabetical order. Reorder to match original
        # inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed
        # parameters from the original inputs and the MCMC results.
        param_dict = {}
        pcounter = 0
        for param in params:
            param_dict[param] = {}
            dist = f['inputs'][param]['distribution'][()].decode()
            # Used input values for fixed parameters.
            if dist == 'fixed':
                param_dict[param]['value'] = f['inputs'][param]['value'][()]
            # Use fitted values for others.
            else:
                param_dict[param]['value'] = bestfit[pcounter]
                pcounter += 1

    return param_dict


def get_fit_results_from_mcmc(filename, burnin=None, thin=15):
    """Extract MCMC posterior sample statistics (median and 1 sigma bounds)
    for each fitted parameter.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
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

    fancyprint('Importing fit results from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        mcmc = f['mcmc']['chain'][()]
        # Discard burn in and thin chains.
        if burnin is None:
            burnin = int(0.75 * np.shape(mcmc)[0])
        # Cut steps for burn in.
        mcmc = mcmc[burnin:]
        nwalkers, nchains, ndim = np.shape(mcmc)
        # Flatten chains.
        mcmc = mcmc.reshape(nwalkers * nchains, ndim)[::thin]

        # HDF5 groups are in alphabetical order. Reorder to match original
        # inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed
        # parameters from the original inputs and the MCMC results.
        results_dict = {}
        pcounter = 0
        for param in params:
            dist = f['inputs'][param]['distribution'][()].decode()
            # Skip fixed paramaters.
            if dist == 'fixed':
                continue
            # Get posterior median and 1 sigma range for fitted paramters.
            else:
                results_dict[param] = {}
                med = np.nanmedian(mcmc[:, pcounter], axis=0)
                low, up = np.diff(np.nanpercentile(mcmc[:, pcounter], [16, 50, 84]))
                results_dict[param]['median'] = med
                results_dict[param]['low_1sigma'] = low
                results_dict[param]['up_1sigma'] = up
                pcounter += 1

    return results_dict


def ld_q2u(q1, q2):
    """Convert from Kipping to normal limb darkening parameters.
    """
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)
    return u1, u2


def ld_u2q(u1, u2):
    """Convert from normal to Kipping limb darkening parameters.
    """
    q1 = (u1+u2)**2
    q2 = u1/(2*(u1+u2))
    return q1, q2
