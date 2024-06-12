#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:36 2024

@author: MCR

Stuff.
"""

import batman
import celerite
from celerite import terms
import h5py
import numpy as np

# TODO: 1. Sampler output reading
# TODO: 2. Output plotting
# TODO: 3. Eclipse model
# TODO: 4. Nested sampling
# TODO: 5. Kipping limb darkening
class Model:
    """Class to create a light curve model.
    """

    def __init__(self, input_parameters, t, linear_regressors=None,
                 observations=None, gp_regressors=None, ld_model='quadratic',
                 silent=False):
        """Initialize the Model class.

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
        self.silent = silent
        self.flux_decomposed = None
        self.flux = None
        self.gp_kernel = None
        self.flux_decomposed = {}
        self.flux = {}

        # Unpack the input parameter dictionary into a form more amenable to
        # create models.
        # Go through input params once to get number of different instruments.
        self.multiplicity = {}
        for param in input_parameters.keys():
            param_split = param.split('_')
            # Length of param_split should be at least 3 -- key name, planet
            # number, and instrument.
            # First chunk is always parameter key, so start at chunk number 2.
            for chunk in param_split[1:]:
                # Ignore planet identifiers.
                if chunk[0] == 'p' and chunk[1].isdigit():
                    pass
                # Ignore GP parameters.
                elif param[:2] == 'GP':
                    pass
                # If it is a new instrument, add to list.
                elif chunk not in self.multiplicity.keys():
                    self.multiplicity[chunk] = []
                    # Make sure time axis is passed for each instrument.
                    if chunk not in t.keys():
                        msg = 'No timestamps passed for instrument ' \
                              '{}'.format(chunk)
                        raise ValueError(msg)

        # Now go through a second time to get the number of planets.
        for param in input_parameters.keys():
            param_split = param.split('_')
            for inst in self.multiplicity.keys():
                if inst in param_split:
                    for chunk in param_split[1:]:
                        # If it is a new planet identifier, increase
                        # multiplicity.
                        if chunk[0] == 'p' and chunk[1].isdigit():
                            if chunk not in self.multiplicity[inst]:
                                self.multiplicity[inst].append(chunk)

        for inst in self.multiplicity.keys():
            if not self.silent:
                print('Importing parameters for {0} planet(s) from instrument '
                      '{1}.'.format(len(self.multiplicity[inst]), inst))

        # Set up storage dictionaries for properties of each planet.
        self.pl_params = {}
        for inst in self.multiplicity.keys():
            self.pl_params[inst] = {}
            for pl in self.multiplicity[inst]:
                self.pl_params[inst][pl] = {}

        # Populate parameters dictionary from input data.
        for param in input_parameters.keys():
            param_split = param.split('_')
            # First chunk is always parameter key.
            prop = param_split[0]
            # Error inflation parameter -- property of instrument.
            if prop == 'sigma':
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        self.pl_params[inst][prop] = input_parameters[param]['value']
            # Linear systematics -- property of instrument.
            elif prop[:5] == 'theta':
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        self.pl_params[inst][prop] = input_parameters[param]['value']
            # GP systematics -- property of instrument.
            elif prop == 'GP':
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        thisprop = param_split[0] + '_' + param_split[1]
                        self.pl_params[inst][thisprop] = input_parameters[param]['value']
            # Assume everything else is an astrophysical parameter.
            else:
                # Add to correct instrument and planet dictionary.
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        for pl in self.multiplicity[inst]:
                            if pl in param_split:
                                self.pl_params[inst][pl][prop] = input_parameters[param]['value']

        # Convert timestamps to float64 -- avoids issue with batman.
        for inst in self.t.keys():
            self.t[inst] = self.t[inst].astype(np.float64)

    def compute_lightcurves(self):
        """Given a set of input parameters, compute a light curve model.
        """

        if not self.silent:
            print('Computing light curves for all instruments.')

        # Individually treat each instrument.
        for inst in self.multiplicity.keys():
            # For each instrument, model will be decomposed into astrophysical,
            # linear model, and GP components, and stored in a dictionary for
            # easy interpretation.
            self.flux_decomposed[inst] = {}
            self.flux_decomposed[inst]['pl'] = {}
            self.flux[inst] = np.ones_like(self.t[inst])
            use_lm, use_gp = False, False

            # Detect and format linear model and GP parameters.
            self.flux_decomposed[inst]['lm'] = None
            self.flux_decomposed[inst]['gp'] = None
            # Unpack LM multipliers.
            thetas, gp_params = [], []
            for param in self.pl_params[inst].keys():
                # Note if LMs are to be used.
                if param[:5] == 'theta':
                    # Ensure that there are appropriate regressors.
                    if self.linear_regressors is None or inst not in self.linear_regressors.keys():
                        msg = 'No regressors passed for instrument ' \
                              '{}'.format(inst)
                        raise ValueError(msg)
                    thetas.append(self.pl_params[inst][param])
                    use_lm = True
                # Note if a GP is to be used.
                elif param[:2] == 'GP':
                    # Ensure that there are appropriate regressors.
                    if self.gp_regressors is None or inst not in self.gp_regressors.keys():
                        msg = 'No GP regressors passed for instrument ' \
                              '{}'.format(inst)
                        raise ValueError(msg)
                    gp_params.append(param)
                    use_gp = True
                else:
                    pass

            # === Astrophysical Model ===
            # Generate astrophysical light curve for each planet.
            for pl in self.multiplicity[inst]:
                # Pack limb darkening parameters.
                ld_params = []
                for param in self.pl_params[inst][pl].keys():
                    if param in ['u1', 'u2', 'u3', 'u4']:
                        ld_params.append(self.pl_params[inst][pl][param])
                # Calculate a basic transit model using the input parameters.
                pl_flux = batman_transit(self.t[inst],
                                         self.pl_params[inst][pl]['t0'],
                                         self.pl_params[inst][pl]['per'],
                                         self.pl_params[inst][pl]['rp'],
                                         self.pl_params[inst][pl]['a'],
                                         self.pl_params[inst][pl]['inc'],
                                         self.pl_params[inst][pl]['ecc'],
                                         self.pl_params[inst][pl]['w'],
                                         ld_params, ld_model=self.ld_model)
                # Store the model for each planet seperately.
                self.flux_decomposed[inst]['pl'][pl] = pl_flux
                # Add contribution of planet to the total astrophysical model.
                self.flux[inst] -= (1 - pl_flux)
            self.flux_decomposed[inst]['pl']['total'] = np.copy(self.flux[inst])

            # === Linear Models ===
            if use_lm is True:
                if not self.silent:
                    print('Linear model(s) detected for instrument '
                          '{}.'.format(inst))
                self.flux_decomposed[inst]['lm'] = {}
                regressors = np.array(self.linear_regressors[inst])
                # Make sure that the number of regressors equals the number of
                # lm parameters.
                msg = 'Number of linear model parameters does not match ' \
                      'number of regressors for instrument {}.'.format(inst)
                assert np.shape(regressors)[0] == len(thetas), msg

                self.flux_decomposed[inst]['lm']['total'] = np.zeros_like(regressors[0])
                # For each LM regressor, add it to the model.
                for i, theta in enumerate(thetas):
                    thismodel = theta * regressors[i]
                    self.flux_decomposed[inst]['lm']['regressor{}'.format(i)] = thismodel
                    self.flux_decomposed[inst]['lm']['total'] += thismodel

                # Add the total LM model to the total light curve model.
                self.flux[inst] += self.flux_decomposed[inst]['lm']['total']

            # === GP Models ===
            # Acceptable GP kernels.
            gp_kernels = {'SHO-gran': ['GP_ag', 'GP_bg', 'GP_Q'],
                          'SHO': ['GP_S0', 'GP_omega0', 'GP_Q'],
                          'Matern 3/2': ['GP_sigma', 'GP_rho']}
            if use_gp is True:
                # Ensure observations are passed for this instrument.
                if self.observations is None or self.observations[inst] is None:
                    msg = 'Observations must be passed for instrument {} to ' \
                          'use a GP.'.format(inst)
                    raise ValueError(msg)

                # Identify GP kernel to use (if any).
                for kernel in gp_kernels.keys():
                    if np.all(np.sort(gp_kernels[kernel]) == np.sort(gp_params)):
                        self.gp_kernel = kernel
                        if not self.silent:
                            print('GP kernel {} identified.'.format(kernel))
                if self.gp_kernel is None:
                    msg = 'No recognized GP kernel with parameters ' \
                          '{}.'.format(gp_params)
                    raise ValueError(msg)

                # Calculate GP model.
                self.flux_decomposed[inst]['gp'] = {}
                # Initialize the appropriate kernel.
                if self.gp_kernel == 'SHO-gran':
                    # Convert from granulation parameters to SHO parameters.
                    omega = 2 * np.pi * self.pl_params[inst]['GP_bg']
                    S0 = 2 * self.pl_params[inst]['GP_ag']**2 / self.pl_params[inst]['GP_bg']
                    Q = self.pl_params[inst]['GP_Q']
                    kernel = terms.SHOTerm(log_S0=np.log(S0),
                                           log_omega0=np.log(omega),
                                           log_Q=np.log(Q))
                elif self.gp_kernel == 'SHO':
                    kernel = terms.SHOTerm(log_S0=np.log(self.pl_params[inst]['GP_S0']),
                                           log_omega0=np.log(self.pl_params[inst]['GP_omega0']),
                                           log_Q=np.log(self.pl_params[inst]['GP_Q']))
                elif self.gp_kernel == 'Matern 3/2':
                    kernel = terms.SHOTerm(log_sigma=np.log(self.pl_params[inst]['GP_sigma']),
                                           log_rho=np.log(self.pl_params[inst]['GP_rho']))
                else:
                    raise ValueError('Bad GP kernel.')

                # Use the GP to make a prediction based on the observations
                # and current light curve model.
                gp = celerite.GP(kernel, mean=0)
                gp.compute(self.t[inst], self.pl_params[inst]['sigma'])
                thismodel = gp.predict(self.observations[inst]['flux'] - self.flux[inst],
                                       self.t[inst], return_cov=False,
                                       return_var=False)
                # Add GP model to light curve model.
                self.flux_decomposed[inst]['gp']['total'] = thismodel
                self.flux[inst] += self.flux_decomposed[inst]['gp']['total']

            self.flux_decomposed[inst]['total'] = self.flux[inst]

    def simulate_observations(self):
        """Given a light curve model and expected scatter, simulate some fake
        observations.
        """

        # Make sure that light curves have already been calculated.
        if self.flux_decomposed is None:
            msg = 'It looks like the compute_lightcurves method has not yet ' \
                  'been run.\n compute_lightcurves must be run before ' \
                  'create_observations.'
            raise ValueError(msg)

        # Don't run if observations already exist.
        if self.observations is not None:
            msg = 'Observational data already exists. I imagine you do not ' \
                  'want to overwrite it!'
            raise ValueError(msg)

        if not self.silent:
            print('Simulating observations for all instruments.')

        # Add scatter to light curves.
        self.observations = {}
        for inst in self.multiplicity.keys():
            self.observations[inst] = {}
            jitter = self.pl_params[inst]['sigma']
            flux_jitter = np.random.normal(self.flux[inst], scale=jitter)
            self.observations[inst]['flux'] = flux_jitter
            self.observations[inst]['flux_err'] = jitter


def batman_transit(t, t0, per, rp, a, inc, ecc, w, ld, ld_model='quadratic'):
    """Calculate a simple transit model.

    Parameters
    ----------
    t : ndarray(float)
        Time stamps at which to calculate the light curve.
    t0 : float
        Time of mid-transit.
    per : float
        Planet orbital period in days.
    rp : float
        Planet-to-star radius ratio.
    a : float
        Planet semi-major axis in units of stellar radii.
    inc : float
        Planet orbital inclination in degrees.
    ecc : float
        Planet orbital eccentricity.
    w : float
        Planet argument of periastron.
    ld : list(float)
        List of limb darkening parameters.
    ld_model : str
        BATMAN limb darkening identifier.

    Returns
    -------
    flux : ndarray(float)
        Model light curve.
    """

    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = ld_model
    params.u = ld

    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    return flux


def get_param_dict_from_mcmc(filename, method='median', burnin=None, thin=15):
    print('Importing fitted parameters from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        mcmc = f['mcmc']['chain'][()]
        nwalkers, nchains, ndim = np.shape(mcmc)
        # Discard burn in and thin chains.
        if burnin is None:
            burnin = int(0.75 * nwalkers * nchains)
        mcmc = mcmc.reshape(nwalkers * nchains, ndim)[burnin:][::thin]
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
    print('Importing fit results from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        mcmc = f['mcmc']['chain'][()]
        nwalkers, nchains, ndim = np.shape(mcmc)
        # Discard burn in and thin chains.
        if burnin is None:
            burnin = int(0.75 * nwalkers * nchains)
        mcmc = mcmc.reshape(nwalkers * nchains, ndim)[burnin:][::thin]

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
