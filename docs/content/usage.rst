exoUPRF Usage Guide
====================
exoUPRF is designed to be easy-to-use and hackable, allowing flexibility in light curve model selection necessary for JWST exoplanet light curves.

Built-in Light Curve Models
---------------------------
The following is a list of light curve models built in to exoUPRF, as well as their necessary parameters and calls.

Simple Transit
""""""""""""""
call: ``transit``

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``P_p1``         | The orbital period of the planet (days).                              |
+------------------+-----------------------------------------------------------------------+
| ``t0_p1``        | The time of mid-transit of the planet (days).                         |
+------------------+-----------------------------------------------------------------------+
| ``p_p1``         | Planet-to-star radius ratio (Rp/Rs).                                  |
+------------------+-----------------------------------------------------------------------+
| ``inc_p1``       | Orbital inclination (deg).                                            |
+------------------+-----------------------------------------------------------------------+
| ``a_p1``         | Scaled orbital semi-major axis (a/R*).                                |
+------------------+-----------------------------------------------------------------------+
| ``ecc_p1``       | Orbital eccentricity.                                                 |
+------------------+-----------------------------------------------------------------------+
| ``w_p1``         | Argument of periastron (deg).                                         |
+------------------+-----------------------------------------------------------------------+

Additionally, transit models include limb darkening. Supported limb darkening models are: ``linear``, ``quadratic``, ``quadratic-kipping``, ``square-root``, or ``nonlinear``.

Transit Model with a Spot Crossing
""""""""""""""""""""""""""""""""""
call: ``transit_spot_crossing``

A transit model which includes a Gaussian "bump" to include a potential star spot crossing.
In addition to the simple transit parameters listed above, the transit with spot crossing model has three extras:

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``spot-amp``     | Amplitude of the Gaussian bump.                                       |
+------------------+-----------------------------------------------------------------------+
| ``spot-pos``     | Position of the Gaussian bump (days).                                 |
+------------------+-----------------------------------------------------------------------+
| ``spot-dur``     | Width of the Gaussian bump (days).                                    |
+------------------+-----------------------------------------------------------------------+

Transit Model with Quadratic Baseline Curvature
"""""""""""""""""""""""""""""""""""""""""""""""
call: ``transit_quad_curvature``

A transit model which includes quadratic curvature (with a variable center of curvature) in the baseline.
In addition to the simple transit parameters listed above, the transit with quadratic curvature model has two extras:

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``curv-amp``     | Amplitude of the curvature.                                           |
+------------------+-----------------------------------------------------------------------+
| ``curv-pos``     | Position center of curvature, relative to mid-transit (days).         |
+------------------+-----------------------------------------------------------------------+

Simple Eclipse
""""""""""""""
call: ``eclipse``

In addition to the simple transit parameters listed above, the eclipse model has two extras:

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``tsec_p1``      | The time of secondary eclipse (days).                                 |
+------------------+-----------------------------------------------------------------------+
| ``fp_p1``        | Planet-to-star flux ratio (Fp/Fs).                                    |
+------------------+-----------------------------------------------------------------------+

Built-in Systematics Models
---------------------------
Systematics are handled in exoUPRF via two methods: linear models or Gaussian processes.

Linear Models
"""""""""""""
Linear model detrending is one of the simplest forms of systematics correction.
A user-defined vector is linearlly scaled to attempt to remove systematic noise froma light curve, i.e.,

.. math::

    \text{light curve} = \text{transit} + \sum_{i=1}^N X_i\theta_i,

where :math:`\theta_i` is a set of :math:`N` user-defined systematics vectors, and :math:`X_i` is the set of scalars multiplying the systematics vectors.

Gaussian Processes
""""""""""""""""""
Gaussian processes (GPs) are non-parametric models often used to handle systematic trends for which we don't have an analytical model.

Currently supported GP kernels and their parameters are:

**Matérn 3/2**

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``GP_sigma``     | Characteristic amplitude of the GP.                                   |
+------------------+-----------------------------------------------------------------------+
| ``GP_rho``       | Characteristic timescale of the GP.                                   |
+------------------+-----------------------------------------------------------------------+

**Simple Harmonic Oscillator**

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``GP_S0``        | Characteristic amplitude of the GP.                                   |
+------------------+-----------------------------------------------------------------------+
| ``GP_omega0``    | Characteristic frequency of the GP.                                   |
+------------------+-----------------------------------------------------------------------+
| ``GP_Q``         | Quality factor of the oscillation.                                    |
+------------------+-----------------------------------------------------------------------+

For more information on the SHO kernel see `Foreman-Mackey et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract>`_.

**SHO for Stellar Granulation**

This is a slight transformation of the above SHO kernel to model the impacts of stellar granulation in exoplanet light curves.
The transformations are outlined in equation 12 of `Pereira et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5764P/abstract>`_.

+------------------+-----------------------------------------------------------------------+
| Parameter Name   |           Description                                                 |
+==================+=======================================================================+
| ``GP_ag``        | Characteristic amplitude of the granulation signal.                   |
+------------------+-----------------------------------------------------------------------+
| ``GP_bg``        | Characteristic frequency of the granulation signal.                   |
+------------------+-----------------------------------------------------------------------+


Using Custom Light Curve Models
-------------------------------
exoUPRF comes with two built-in light curve models: a simple transit and a simple eclipse model with no extra bells and whistles.
However, exoUPRF is also made to allow for users to easily swap in their own light curve models, as long as they obey a couple of key rules.
The custom function call must take the observation time array, and the planet parameters dictionary as inputs, and return a single flux array with the same shape as the time array.
Here's an example function definition:

.. code-block:: python

    def custom_transit(time,            # Array of timestamps.
                       pl_params        # exoUPRF planet parameters dictionary.
                       ):
        flux = ...                      # Code to calculate the custom transit model.
        assert len(flux) == len(time)   # Flux must have the same length as time.
        return flux

Getting exoUPRF to use the custom model is also incredibly easy! First, in the ``model_type`` dictionary, instead of indicating ``transit`` or ``eclipse`` as usual, instead specify ``custom-transit`` or ``custom-eclipse``, depending on the light curve in question.

.. code-block:: python

    model_type = {'inst': {'p1': 'custom-transit'}}

Then, define a ``model_functions`` dictionary, to which we will pass the call to the ``custom_transit`` function defined above.

.. code-block:: python

    from custom_models import custom_transit
    model_functions = {'inst': {'p1': custom_transit}}

Finally, simply pass both of these dictionaries to either the ``compute_lightcurves`` or ``fit`` function calls.

Tutorial Notebooks
------------------
Below is a tutorial notebook that will walk you through the basics fitting JWST light curves with exoUPRF.

.. toctree::
   :maxdepth: 2

   notebooks/tutorial_basic-transit-fitting
