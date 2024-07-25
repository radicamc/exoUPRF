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


def get_param_dict_from_fit(filename, method='median', mcmc_burnin=None,
                            mcmc_thin=15, silent=False, drop_chains=None):
    """Reformat fit outputs from MCMC or NS into the parameter dictionary
    format expected by Model.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    method : str
        Method via which to get best fitting parameters from MCMC chains.
        Either "median" or "maxlike".
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    param_dict : dict
        Dictionary of light curve model parameters.
    """

    if not silent:
        fancyprint('Importing fitted parameters from file '
                   '{}.'.format(filename))

    # Get sample chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
            sampler = 'mcmc'
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
            sampler = 'ns'
        else:
            msg = 'No MCMC or Nested Sampling results in file ' \
                  '{}.'.format(filename)
            raise KeyError(msg)
        # Either get maximum likelihood solution...
        if method == 'maxlike':
            if sampler == 'mcmc':
                lp = f['mcmc']['log_prob'][()].flatten()[mcmc_burnin:][::mcmc_thin]
                ii = np.argmax(lp)
                bestfit = chain[ii]
            else:
                bestfit = chain[-1]
        # ...or take median of samples.
        elif method == 'median':
            bestfit = np.nanmedian(chain, axis=0)

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


def get_results_from_fit(filename, mcmc_burnin=None, mcmc_thin=15,
                         silent=False, drop_chains=None):
    """Extract posterior sample statistics (median and 1 sigma bounds) for
    each fitted parameter.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    results_dict : dict
        Dictionary of posterior medians and 1 sigma bounds for each fitted
        parameter.
    """

    if not silent:
        fancyprint('Importing fit results from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
        else:
            msg = 'No MCMC or Nested Sampling results in file ' \
                  '{}.'.format(filename)
            raise KeyError(msg)

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
                med = np.nanmedian(chain[:, pcounter], axis=0)
                low, up = np.diff(np.nanpercentile(chain[:, pcounter], [16, 50, 84]))
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
