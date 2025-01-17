#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:13 2024

@author: MCR

Light curve plotting functions.
"""

import corner
import h5py
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def make_allan_plot(residuals, integration_time=None, labels=None,
                    just_calculate=False):
    """Make an allan variance plot for a given set of residuals.
    Original routine by Hannah Wakeford, adapted by MCR.

    Parameters
    ----------
    residuals : list
        List of arrays of residuals to bin.
    integration_time : float
        Integration time for the observations.
    labels : list(str)
        List of labels corresponding to each of the passed residuals. Must
        have the same size as residuals.
    just_calculate : bool
        If True, don't plot anything. Just calculate the binning statistics
        and return them.
    """

    if labels is not None:
        assert len(labels) == len(residuals)

    # Bin data into multiple bin sizes.
    maxnbins = len(residuals[0]) / 4
    binstep = 1
    # Create an array of the bin sizes in integrations.
    bins = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)

    if just_calculate is False:
        f, ax1 = plt.subplots(facecolor='white', figsize=(6, 4))
    # Loop over each pack of residuals in the input.
    for r in range(len(residuals)):
        thisres = residuals[r]

        # Initialize storage arrays.
        nbins = np.zeros(len(bins), dtype=int)
        rms = np.zeros(len(bins))

        # Loop over each bin size and bin the requisite number of integrations.
        for i in range(len(bins)):
            # Total number of bins binning binz[i] integrations per bin.
            nbins[i] = int(np.floor(thisres.size / bins[i]))
            bindata = np.zeros(nbins[i], dtype=float)

            # Do the binning.
            for j in range(nbins[i]):
                bindata[j] = np.nanmean(thisres[j*bins[i]:(j+1)*bins[i]])

            # Get rms.
            rms[i] = np.sqrt(np.nanmean(bindata**2))

        # Get expected white noise trend.
        phot_noise = (np.nanstd(thisres) / np.sqrt(bins)) * np.sqrt(nbins / (nbins-1))

        # Plot up the binned statistic against the expected statistic.
        if just_calculate is False:
            if labels is not None:
                ax1.plot(bins, rms*1e6, label=labels[r])
            else:
                ax1.plot(bins, rms*1e6)
            ax1.plot(bins, phot_noise*1e6, color='black', ls='--')

    if just_calculate is False:
        # Plot formatting.
        ax1.set_xlabel('Bin Size [integrations]', fontsize=12)
        ax1.set_ylabel('RMS [ppm]', fontsize=12)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        if labels is not None:
            ax1.legend()

        # Now with time as the x-axis.
        if integration_time is not None:
            ax2 = ax1.twiny()
            ax2.plot(bins * integration_time, rms*1e6, lw=0)
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlabel('Bin Size [mins]', fontsize=12)
            ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())

        plt.show()

        return
    else:
        return rms, bins, phot_noise


def make_corner_plot(filename, mcmc_burnin=None, mcmc_thin=15, labels=None,
                     outpdf=None, log_params=None, drop_chains=None):
    """Make a corner plot of fitted posterior distributions.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. MCMC only.
    mcmc_thin : int
        Increment by which to thin chains. MCMC only.
    labels : list(str)
        Fitted parameter names.
    outpdf : PdfPages
        Path to file to save plot.
    log_params : list(int), None
        Indices of parameters to show on log scale.
    drop_chains : list(int), None
        Indices of chains to drop.
    """

    # Get chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            samples = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(samples)[0])
            # Cut steps for burn in.
            samples = samples[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                samples = np.delete(samples, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(samples)
            # Flatten chains.
            samples = samples.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
        elif 'ns' in list(f.keys()):
            samples = f['ns']['chain'][()]

    # Log certain parameters if necessary.
    if log_params is not None:
        log_params = np.atleast_1d(log_params)
        for i in log_params:
            samples[:, i] = np.log10(samples[:, i])

    # Make corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(figure)
        else:
            figure.savefig(outpdf)
        figure.clear()
        plt.close(figure)
    else:
        plt.show()


def make_lightcurve_plot(t, data, model, scatter, errors=None, nfit=None,
                         outpdf=None,  title=None, systematics=None,
                         rasterized=False, nbin=10):
    """Plot results of a light curve fit.

    Parameters
    ----------
    t : ndarray(float)
        Time axis for obsevrations (ideally in hours from mid-transit).
    data : ndarray(float)
        Light curve observations.
    model : ndarray(float)
        Light curve model.
    scatter : float
        Fitted scatter value or inflated errors.
    errors : float
        Original data errors.
    nfit : int
        Number of fitted parameters.
    outpdf : PdfPages
        Path to file to save plot.
    title : str
        Plot title.
    systematics : ndarray(float)
        Fitted systematics model.
    rasterized : bool
        If True, rasterize plot.
    nbin : int
        Number of data points to bin in plot.
    """

    def gaus(x, m, s):
        return np.exp(-0.5*(x - m)**2/s**2)/np.sqrt(2*np.pi*s**2)

    def chi2(o, m, e):
        return np.nansum((o - m)**2/e**2)

    if systematics is not None:
        fig = plt.figure(figsize=(13, 9), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(5, 1, height_ratios=[3, 3, 1, 0.3, 1])
    else:
        fig = plt.figure(figsize=(13, 7), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(4, 1, height_ratios=[3, 1, 0.3, 1])

    # Light curve with full systematics + astrophysical model.
    ax1 = plt.subplot(gs[0])
    assert len(data) == len(model)
    nint = len(data)  # Total number of data points
    # Full dataset
    ax1.errorbar(t, data, yerr=scatter, fmt='o', capsize=0,
                 color='royalblue', ms=5, alpha=0.25)
    # Binned points
    rem = nint % nbin
    if rem != 0:
        trim_i = np.random.randint(0, rem)
        trim_e = -1*(rem-trim_i)
        t_bin = t[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
        d_bin = data[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        t_bin = t.reshape((nint-rem)//nbin, nbin)
        d_bin = data.reshape((nint-rem)//nbin, nbin)
    t_bin = np.nanmean(t_bin, axis=1)
    d_bin = np.nanmean(d_bin, axis=1)
    ax1.errorbar(t_bin, d_bin, yerr=scatter/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax1.plot(t, model, color='black', zorder=10)
    ax1.set_ylabel('Relative Flux', fontsize=18)
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    txt = ''
    if errors is not None:
        if nfit is not None:
            chi2_v = chi2(data * 1e6, model * 1e6, errors * 1e6) / (len(t) - nfit)
            txt += r'$\chi_\nu^2 = {:.2f}$''\n'.format(chi2_v)
        mean_err = np.nanmean(errors)
        err_mult = scatter / mean_err
        txt += r'$\sigma={:.2f}$ppm''\n'r'$e={:.2f}$'.format(mean_err*1e6, err_mult)
    ax1.text(t[2], np.min(model), txt, fontsize=14)

    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    # Detrended Light curve.
    if systematics is not None:
        ax2 = plt.subplot(gs[1])
        assert len(model) == len(systematics)
        model_detrended = model - systematics
        data_detrended = data - systematics
        # Full dataset.
        ax2.errorbar(t, data_detrended, yerr=scatter, fmt='o',
                     capsize=0, color='salmon', ms=5, alpha=0.25)
        # Binned points.
        if rem != 0:
            d_bin = data_detrended[trim_i:trim_e].reshape((nint-rem)//nbin,
                                                          nbin)
        else:
            d_bin = data_detrended.reshape((nint-rem)//nbin, nbin)
        d_bin = np.nanmean(d_bin, axis=1)
        ax2.errorbar(t_bin, d_bin, yerr=scatter/np.sqrt(nbin), fmt='o',
                     mfc='red', mec='white', ecolor='red', ms=8, alpha=1,
                     zorder=11)
        # Other stuff.
        ax2.plot(t, model_detrended, color='black', zorder=10)
        ax2.set_ylabel('Relative Flux\n(Detrended)', fontsize=18)
        ax2.set_xlim(np.min(t), np.max(t))
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)

    # Residuals.
    if systematics is not None:
        ax3 = plt.subplot(gs[2])
    else:
        ax3 = plt.subplot(gs[1])
    # Full dataset.
    res = (data - model)*1e6
    ax3.errorbar(t, res, yerr=scatter, alpha=0.25, ms=5,
                 c='royalblue', fmt='o', zorder=10)
    # Binned points.
    if rem != 0:
        r_bin = res[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        r_bin = res.reshape((nint-rem)//nbin, nbin)
    r_bin = np.nanmean(r_bin, axis=1)
    ax3.errorbar(t_bin, r_bin, yerr=scatter/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax3.axhline(0, ls='--', c='black')
    xpos = np.percentile(t, 1)
    plt.text(xpos, np.max((data - model)*1e6),
             r'{:.2f}$\,$ppm'.format(scatter*1e6))
    ax3.fill_between(t, -scatter*1e6, scatter*1e6, color='black', alpha=0.1)
    ax3.set_xlim(np.min(t), np.max(t))
    ax3.set_ylabel('Residuals\n(ppm)', fontsize=18)
    ax3.set_xlabel('Time from Transit Midpoint [hrs]', fontsize=18)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)

    # Histogram of residuals.
    if systematics is not None:
        ax4 = plt.subplot(gs[4])
    else:
        ax4 = plt.subplot(gs[3])
    bins = np.linspace(-10, 10, 41) + 0.25
    hist = ax4.hist(res/(scatter*1e6), edgecolor='grey', color='lightgrey',
                    bins=bins)
    area = np.sum(hist[0] * np.diff(bins))
    ax4.plot(np.linspace(-15, 15, 500),
             gaus(np.linspace(-15, 15, 500), 0, 1) * area, c='black')
    ax4.set_ylabel('Counts', fontsize=18)
    ax4.set_xlabel('Residuals/Scatter', fontsize=18)
    ax4.set_xlim(-5, 5)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_mcmc_chains(filename, labels=None, log_params=None,
                     highlight_chains=None, drop_chains=None):
    """Plot MCMC chains.

    Parameters
    ----------
    filename : str
        MCMC output file.
    labels : list(str)
        Fitted parameter names.
    log_params : list(str), None
        Indices of parameters to plot in log-space.
    highlight_chains : list(int), None
        Indices of chains to highlight.
    drop_chains : list(int), None
        Indices of chains to drop.
    """

    # Get MCMC chains.
    with h5py.File(filename, 'r') as f:
        samples = f['mcmc']['chain'][()]

    # Convert to log-scale if necessary.
    if log_params is not None:
        log_params = np.atleast_1d(log_params)
        for i in log_params:
            samples[:, :, i] = np.log10(samples[:, :, i])

    # Drop chains if necessary.
    if drop_chains is not None:
        drop_chains = np.atleast_1d(drop_chains)
        samples = np.delete(samples, drop_chains, axis=1)

    nwalkers, nchains, ndim = np.shape(samples)
    # Plot chains.
    fig, axes = plt.subplots(ndim,
                             figsize=(10, np.ceil(ndim / 1.25).astype(int)),
                             sharex=True)

    if highlight_chains is not None:
        highlight_chains = np.atleast_1d(highlight_chains)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], c='black', alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if labels is not None:
            ax.set_ylabel(labels[i])
        if highlight_chains is not None:
            for j in highlight_chains:
                ax.plot(samples[:, j, i], c='red', alpha=0.5)

    axes[-1].set_xlabel('Step Number')
    plt.show()
