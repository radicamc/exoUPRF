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
import numpy as np


class Model:

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
        self.silent = silent
        self.gp_kernel = None

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
            # Keys for physical planet parameters.
            if prop in ['per', 't0', 'rp', 'a', 'inc', 'u1', 'u2', 'u3', 'u4',
                        'ecc', 'w']:
                # Add to correct instrument and planet dictionary.
                for inst in self.multiplicity.keys():
                    if inst in param_split:
                        for pl in self.multiplicity[inst]:
                            if pl in param_split:
                                self.pl_params[inst][pl][prop] = input_parameters[param]['value']
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
            else:
                msg = 'Unrecognized input parameter: {}'.format(param)
                raise ValueError(msg)

    def compute_lightcurves(self):

        if not self.silent:
            print('Computing lighcturves for all instruments.')
        self.flux_decomposed = {}
        self.flux_decomposed = {}
        self.flux = {}

        # Individually treat each instrument.
        for inst in self.multiplicity.keys():
            self.flux_decomposed[inst] = {}
            self.flux_decomposed[inst]['pl'] = {}
            self.flux[inst] = np.ones_like(self.t[inst])
            use_lm, use_gp = False, False

            # === Astrophysical Model ===
            # Generate astrophysical light curve for each planet.
            for pl in self.multiplicity[inst]:
                ld_params = []
                for param in self.pl_params[inst][pl].keys():
                    if param in ['u1', 'u2', 'u3', 'u4']:
                        ld_params.append(self.pl_params[inst][pl][param])
                pl_flux = batman_transit(self.t[inst],
                                         self.pl_params[inst][pl]['t0'],
                                         self.pl_params[inst][pl]['per'],
                                         self.pl_params[inst][pl]['rp'],
                                         self.pl_params[inst][pl]['a'],
                                         self.pl_params[inst][pl]['inc'],
                                         self.pl_params[inst][pl]['ecc'],
                                         self.pl_params[inst][pl]['w'],
                                         ld_params, ld_model=self.ld_model)
                self.flux_decomposed[inst]['pl'][pl] = pl_flux
                self.flux[inst] -= (1 - pl_flux)
            self.flux_decomposed[inst]['pl']['total'] = np.copy(self.flux[inst])

            # === Linear Models ===
            # Add in linear systematics model, if any.
            self.flux_decomposed[inst]['lm'] = None
            # Unpack multipliers.
            thetas = []
            for param in self.pl_params[inst].keys():
                if param[:5] == 'theta':
                    msg = 'No regressors passed for instrument {}'.format(inst)
                    assert inst in self.linear_regressors.keys(), msg
                    thetas.append(self.pl_params[inst][param])
                    # Note that we want to add linear models.
                    use_lm = True

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
                for i, theta in enumerate(thetas):
                    thismodel = theta * regressors[i]
                    self.flux_decomposed[inst]['lm']['regressor{}'.format(i)] = thismodel
                    self.flux_decomposed[inst]['lm']['total'] += thismodel

                self.flux[inst] += self.flux_decomposed[inst]['lm']['total']

            # === GP Models ===
            # Add in GP systematics model, if any.
            self.flux_decomposed[inst]['gp'] = None
            gp_params = []
            for param in self.pl_params[inst].keys():
                # Note if a GP is to be used.
                if param[:2] == 'GP':
                    gp_params.append(param)
                    use_gp = True
                    if self.gp_regressors is None or inst not in self.gp_regressors.keys():
                        msg = 'GP parameters provided for instrument {}, ' \
                              'but no regressors.'.format(inst)
                        raise ValueError(msg)

            gp_kernels = {'SHO': ['GP_ag', 'GP_bg', 'GP_Q']}
            if use_gp is True:
                # Ensure observations are passed for this instrument.
                if self.observations is None or self.observations[inst] is None:
                    msg = 'Observations must be passed for instrument {} to ' \
                          'use a GP.'.format(inst)
                    raise ValueError(msg)
                self.flux_decomposed[inst]['gp'] = {}
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
                if self.gp_kernel == 'SHO':
                    # Convert from granulation parameters to SHO parameters.
                    omega = 2 * np.pi * self.pl_params[inst]['GP_bg']
                    S0 = 2 * self.pl_params[inst]['GP_ag']**2 / self.pl_params[inst]['GP_bg']
                    Q = self.pl_params[inst]['GP_Q']
                    err = self.pl_params[inst]['sigma']
                    kernel = terms.SHOTerm(log_S0=np.log(S0),
                                           log_omega0=np.log(omega),
                                           log_Q=np.log(Q))
                    gp = celerite.GP(kernel, mean=0)
                    gp.compute(self.t[inst], err)
                    thismodel = gp.predict(self.observations[inst]['flux'] - self.flux[inst], self.t[inst],
                                           return_cov=False, return_var=False)

                self.flux_decomposed[inst]['gp']['total'] = thismodel
                self.flux[inst] += self.flux_decomposed[inst]['gp']['total']

            self.flux_decomposed[inst]['total'] = self.flux[inst]

    def simulate_observations(self):

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
