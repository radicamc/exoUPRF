#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jun 13 09:52 2024

@author: MCR

Functions for creating light curve models.
"""

import batman
import celerite
from celerite import terms
import numpy as np
import warnings

import exouprf.utils as utils
from exouprf.utils import fancyprint


class LightCurveModel:
    """Secondary exoUPRF class. Creates light curve models given a set of input parameters and
    light curve model function.
    """

    def __init__(self, input_parameters, t, linear_regressors=None, observations=None,
                 gp_regressors=None, ld_model='quadratic', silent=False):
        """Initialize the Model class.

        Parameters
        ----------
        input_parameters : dict
            Dictionary of input parameters and values. Should have form {parameter: value}.
        t : dict
            Dictionary of timestamps for each instrument. Should have form {instrument: times}.
        linear_regressors : dict
            Dictionary of regressors for linear models. Should have form {instrument: regressors}.
        observations : dict
            Dictionary of observed data. Should have form
            {instrument: {flux: values, flux_err: values}}.
        gp_regressors : dict
            Dictionary of regressors for Gaussian Process models. Should have form
            {instrument: regressors}.
        ld_model : str
            Limb darkening model identifier.
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
        self.gp = {}

        # Unpack the input parameter dictionary into a form more amenable to create models.
        # Go through input params once to get number of different instruments.
        self.multiplicity = {}
        for param in input_parameters.keys():
            param_split = param.split('_')
            # Length of param_split should be at least 3 -- key name, planet number, and instrument.
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
                        msg = 'No timestamps passed for instrument {}'.format(chunk)
                        raise ValueError(msg)

        # Now go through a second time to get the number of planets.
        for param in input_parameters.keys():
            param_split = param.split('_')
            for inst in self.multiplicity.keys():
                if inst in param_split:
                    for chunk in param_split[1:]:
                        # If it is a new planet identifier, increase multiplicity.
                        if chunk[0] == 'p' and chunk[1].isdigit():
                            if chunk not in self.multiplicity[inst]:
                                self.multiplicity[inst].append(chunk)

        for inst in self.multiplicity.keys():
            if not self.silent:
                fancyprint('Importing parameters for {0} planet(s) from instrument {1}.'
                           .format(len(self.multiplicity[inst]), inst))

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
            # Zero point -- property of instrument
            if prop == 'zero':
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        self.pl_params[inst][prop] = input_parameters[param]['value']
            # Error inflation parameter -- property of instrument.
            elif prop == 'sigma':
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
                    # Orbital parameters are not instrument dependent.
                    orb = ['per', 't0', 'a', 'inc', 'ecc', 'w']
                    lds = ['u1', 'u2', 'u3', 'u4', 'q1', 'q2']
                    if inst in param_split or prop in orb:
                        for pl in self.multiplicity[inst]:
                            if pl in param_split or prop in lds:
                                self.pl_params[inst][pl][prop] = input_parameters[param]['value']

        # Convert timestamps to float64 -- avoids issue with batman.
        for inst in self.t.keys():
            self.t[inst] = self.t[inst].astype(np.float64)

    def compute_lightcurves(self, lc_model_type, lc_model_functions=None):
        """Given a set of input parameters, compute a light curve model.

        Parameters
        ----------
        lc_model_type : dict
            Dictionary of light curve model types for each instrument and planet. Should have
            form {inst: {pl: model}}. "transit" and "eclipse" models are supported by default.
            Custom models are possible, and, in this case, model should be the call to the
            custom model function.
        """

        if not self.silent:
            fancyprint('Computing light curves for all instruments.')

        # Individually treat each instrument.
        for inst in self.multiplicity.keys():
            # For each instrument, model will be decomposed into astrophysical, linear model, and
            # GP components, and stored in a dictionary for easy interpretation.
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
                        msg = 'No regressors passed for instrument {}'.format(inst)
                        raise ValueError(msg)
                    thetas.append(self.pl_params[inst][param])
                    use_lm = True
                # Note if a GP is to be used.
                elif param[:2] == 'GP':
                    # Ensure that there are appropriate regressors.
                    if self.gp_regressors is None or inst not in self.gp_regressors.keys():
                        msg = 'No GP regressors passed for instrument {}'.format(inst)
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
                    if param in ['u1', 'u2', 'u3', 'u4', 'q1', 'q2']:
                        ld_params.append(self.pl_params[inst][pl][param])
                # Convert from Kipping to normal LD if necessary.
                try:
                    if self.ld_model.split('-')[1] == 'kipping':
                        assert len(ld_params) == 2
                        msg = 'LD parameters must be >= 0 to use the Kipping parameterization.'
                        assert np.all(np.array(ld_params) >= 0), msg
                        u1, u2 = utils.ld_q2u(ld_params[0], ld_params[1])
                        ld_params = [u1, u2]
                        ld_model = self.ld_model.split('-')[0]
                    else:
                        ld_model = self.ld_model
                except IndexError:
                    ld_model = self.ld_model

                # === Do the Light Curve Calculation ===
                if lc_model_type[inst][pl] == 'transit':
                    # Calculate a basic transit model using the input parameters.
                    pl_flux = simple_transit(self.t[inst], self.pl_params[inst][pl], ld_params,
                                             ld_model=ld_model)
                elif lc_model_type[inst][pl] == 'eclipse':
                    # Calculate a basic eclipse model using the input parameters.
                    pl_flux = simple_eclipse(self.t[inst], self.pl_params[inst][pl])
                elif lc_model_type[inst][pl] == 'custom-transit':
                    # For custom transit models.
                    custom_call = lc_model_functions[inst][pl]
                    pl_flux = custom_call(self.t[inst], self.pl_params[inst][pl], ld_params,
                                          ld_model=ld_model)
                elif lc_model_type[inst][pl] == 'custom-eclipse':
                    # For custom eclipse models.
                    custom_call = lc_model_functions[inst][pl]
                    pl_flux = custom_call(self.t[inst], self.pl_params[inst][pl])
                else:
                    msg = 'Unknown light curve model type {}.'.format(lc_model_type[inst][pl])
                    raise ValueError(msg)
                # Store the model for each planet seperately.
                self.flux_decomposed[inst]['pl'][pl] = pl_flux
                # Add contribution of planet to the total astrophysical model.
                self.flux[inst] -= (1 - pl_flux)
            # Add in the zero point for a given light curve.
            self.flux[inst] += self.pl_params[inst]['zero']

            self.flux_decomposed[inst]['pl']['total'] = np.copy(self.flux[inst])

            # === Linear Models ===
            if use_lm is True:
                if not self.silent:
                    fancyprint('Linear model(s) detected for instrument {}.'.format(inst))
                self.flux_decomposed[inst]['lm'] = {}
                regressors = np.array(self.linear_regressors[inst])
                # Make sure that the number of regressors equals the number of
                # lm parameters.
                msg = 'Number of linear model parameters does not match number of regressors for ' \
                      'instrument {}.'.format(inst)
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
            gp_kernels = {'SHO-gran': ['GP_ag', 'GP_bg'],
                          'SHO': ['GP_S0', 'GP_omega0', 'GP_Q'],
                          'Matern 3/2': ['GP_sigma', 'GP_rho']}
            if use_gp is True:
                # Ensure observations are passed for this instrument.
                if self.observations is None or self.observations[inst] is None:
                    msg = 'Observations must be passed for instrument {} to use a GP.'.format(inst)
                    raise ValueError(msg)

                # Identify GP kernel to use (if any).
                for kernel in gp_kernels.keys():
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=FutureWarning)
                        if np.all(np.sort(gp_kernels[kernel]) == np.sort(gp_params)):
                            self.gp_kernel = kernel
                            if not self.silent:
                                fancyprint('GP kernel {} identified.'.format(kernel))
                if self.gp_kernel is None:
                    msg = 'No recognized GP kernel with parameters {}.'.format(gp_params)
                    raise ValueError(msg)

                # Calculate GP model.
                self.flux_decomposed[inst]['gp'] = {}
                # Initialize the appropriate kernel.
                if self.gp_kernel == 'SHO-gran':
                    # Convert from granulation parameters to SHO parameters.
                    omega = 2 * np.pi * self.pl_params[inst]['GP_bg']
                    s0 = self.pl_params[inst]['GP_ag']**2 / omega / np.sqrt(2)
                    q = 1/np.sqrt(2)
                    kernel = terms.SHOTerm(log_S0=np.log(s0), log_omega0=np.log(omega),
                                           log_Q=np.log(q))
                elif self.gp_kernel == 'SHO':
                    kernel = terms.SHOTerm(log_S0=np.log(self.pl_params[inst]['GP_S0']),
                                           log_omega0=np.log(self.pl_params[inst]['GP_omega0']),
                                           log_Q=np.log(self.pl_params[inst]['GP_Q']))
                elif self.gp_kernel == 'Matern 3/2':
                    kernel = terms.Matern32Term(log_sigma=np.log(self.pl_params[inst]['GP_sigma']),
                                                log_rho=np.log(self.pl_params[inst]['GP_rho']))
                else:
                    raise ValueError('Bad GP kernel.')

                # Use the GP to make a prediction based on the observations
                # and current light curve model.
                gp = celerite.GP(kernel, mean=0)
                try:
                    gp.compute(self.t[inst], self.pl_params[inst]['sigma'])
                    thismodel = gp.predict(self.observations[inst]['flux'] - self.flux[inst],
                                           self.t[inst], return_cov=False, return_var=False)
                    self.gp[inst] = gp
                except Exception as err:
                    if str(err) == 'failed to factorize or solve matrix':
                        self.flux_decomposed[inst]['total'] = -np.inf*self.flux[inst]
                        continue
                    else:
                        raise err

                # Add GP model to light curve model.
                self.flux_decomposed[inst]['gp']['total'] = thismodel
                self.flux[inst] += self.flux_decomposed[inst]['gp']['total']

            self.flux_decomposed[inst]['total'] = self.flux[inst]

    def simulate_observations(self):
        """Given a light curve model and expected scatter, simulate some fake observations.
        """

        # Make sure that light curves have already been calculated.
        if self.flux_decomposed is None:
            msg = 'It looks like the compute_lightcurves method has not yet been run.\n ' \
                  'compute_lightcurves must be run before create_observations.'
            raise ValueError(msg)

        # Don't run if observations already exist.
        if self.observations is not None:
            msg = 'Observational data already exists. I imagine you do not want to overwrite it!'
            raise ValueError(msg)

        if not self.silent:
            fancyprint('Simulating observations for all instruments.')

        # Add scatter to light curves.
        self.observations = {}
        for inst in self.multiplicity.keys():
            self.observations[inst] = {}
            jitter = self.pl_params[inst]['sigma']
            flux_jitter = np.random.normal(self.flux[inst], scale=jitter)
            self.observations[inst]['flux'] = flux_jitter
            self.observations[inst]['flux_err'] = jitter


def simple_eclipse(t, pl_params):
    """Calculate a simple eclipse model.

    Parameters
    ----------
    t : ndarray(float)
        Time stamps at which to calculate the light curve.
    pl_params : dict
        Dictionary of input parameters. Must contain the following:
        t0, time of mid-transit
        per, planet orbital period in days
        rp, planet-to-star radius ratio
        a, planet semi-major axis in units of stellar radii
        inc, planet orbital inclination in degrees
        ecc, planet orbital eccentricity
        w, planet argument of periastron
        tsec, time of secondary eclipse
        fp, planet-to-star flux ratio.

    Returns
    -------
    flux : ndarray(float)
        Model light curve.
    """

    params = batman.TransitParams()
    params.t0 = pl_params['t0']
    params.per = pl_params['per']
    params.rp = pl_params['rp']
    params.a = pl_params['a']
    params.inc = pl_params['inc']
    params.ecc = pl_params['ecc']
    params.w = pl_params['w']
    params.limb_dark = 'quadratic'
    params.u = [0.1, 0.1]
    params.t_secondary = pl_params['tsec']
    params.fp = pl_params['fp']

    m = batman.TransitModel(params, t, transittype='secondary')
    flux = m.light_curve(params)

    return flux


def simple_transit(t, pl_params, ld, ld_model='quadratic'):
    """Calculate a simple transit model.

    Parameters
    ----------
    t : ndarray(float)
        Time stamps at which to calculate the light curve.
    pl_params : dict
        Dictionary of input parameters. Must contain the following:
        t0, time of mid-transit
        per, planet orbital period in days
        rp, planet-to-star radius ratio
        a, planet semi-major axis in units of stellar radii
        inc, planet orbital inclination in degrees
        ecc, planet orbital eccentricity
        w, planet argument of periastron.
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
    params.t0 = pl_params['t0']
    params.per = pl_params['per']
    params.rp = pl_params['rp']
    params.a = pl_params['a']
    params.inc = pl_params['inc']
    params.ecc = pl_params['ecc']
    params.w = pl_params['w']
    params.limb_dark = ld_model
    params.u = ld

    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    return flux


def transit_exp_ramp(t, pl_params, ld, ld_model='quadratic'):
    """Calculate a transit model with an exponential ramp.

        Parameters
        ----------
        t : ndarray(float)
            Time stamps at which to calculate the light curve.
        pl_params : dict
            Dictionary of input parameters. Must contain the following:
            t0, time of mid-transit
            per, planet orbital period in days
            rp, planet-to-star radius ratio
            a, planet semi-major axis in units of stellar radii
            inc, planet orbital inclination in degrees
            ecc, planet orbital eccentricity
            w, planet argument of periastron.
            ramp-amp, amplitude of the exponential ramp.
            ramp-tmsc, Decay (or growth) timescale of the exponential ramp.
        ld : list(float)
            List of limb darkening parameters.
        ld_model : str
            BATMAN limb darkening identifier.

        Returns
        -------
        flux : ndarray(float)
            Model light curve.
        """

    flux = simple_transit(t, pl_params, ld, ld_model=ld_model)
    tt = (t - np.mean(t))/np.std(t)
    flux += pl_params['ramp-amp'] * np.exp(pl_params['ramp-tmsc'] * tt)

    return flux


def transit_quad_curvature(t, pl_params, ld, ld_model='quadratic'):
    """Calculate a transit model with quadratic curvature in the baseline with
    variable amplitude and offset from mid-transit.

        Parameters
        ----------
        t : ndarray(float)
            Time stamps at which to calculate the light curve.
        pl_params : dict
            Dictionary of input parameters. Must contain the following:
            t0, time of mid-transit
            per, planet orbital period in days
            rp, planet-to-star radius ratio
            a, planet semi-major axis in units of stellar radii
            inc, planet orbital inclination in degrees
            ecc, planet orbital eccentricity
            w, planet argument of periastron.
            curv-amp, amplitude of curvature.
            curv-pos, position of curvature mid-point.
        ld : list(float)
            List of limb darkening parameters.
        ld_model : str
            BATMAN limb darkening identifier.

        Returns
        -------
        flux : ndarray(float)
            Model light curve.
        """

    flux = simple_transit(t, pl_params, ld, ld_model=ld_model)
    offset = pl_params['t0'] - pl_params['curv-off']
    flux += pl_params['curv-amp'] * (t - offset)**2

    return flux


def transit_spot_crossing(t, pl_params, ld, ld_model='quadratic'):
    """Calculate a transit model with a star spot crossing. The star spot will
    be modelled as a Gaussian bump in the light curve.

    Parameters
    ----------
    t : ndarray(float)
        Time stamps at which to calculate the light curve.
    pl_params : dict
        Dictionary of input parameters. Must contain the following:
        t0, time of mid-transit
        per, planet orbital period in days
        rp, planet-to-star radius ratio
        a, planet semi-major axis in units of stellar radii
        inc, planet orbital inclination in degrees
        ecc, planet orbital eccentricity
        w, planet argument of periastron.
        spot-amp, amplitude of spot crossing bump.
        spot-pos, position of spot crossing bump.
        spot-dur, duration of spot crossing bump.
    ld : list(float)
        List of limb darkening parameters.
    ld_model : str
        BATMAN limb darkening identifier.

    Returns
    -------
    flux : ndarray(float)
        Model light curve.
    """

    def gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

    flux = simple_transit(t, pl_params, ld, ld_model=ld_model)
    flux += gauss(t, pl_params['spot-amp'], pl_params['spot-pos'],
                  pl_params['spot-dur'])

    return flux
