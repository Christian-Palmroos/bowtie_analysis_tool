#!/usr/bin/env python3
"""
A module for plotting utility functions.
"""
__author__ = "Philipp Oleynik"
__credits__ = ["Philipp Oleynik"]

import matplotlib.patches as pt
import matplotlib.pyplot as plt


def setup_latex(rcParams, no_fourier=False):
    """
    Sets LaTeX environment for better text formatting on plots.
    :param rcParams: rcParams imported locally from matplotlib
    :param no_fourier: True if there is no fourier package in your LaTeX distribution and it is impossible to install it (e.g. on Dione)
    """
    rcParams['text.usetex'] = True
    if no_fourier:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{amssymb}' + '\n' + r'\usepackage{graphicx}'
    else:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{amssymb}' + '\n' + r'\usepackage{graphicx}' + '\n' + r'\usepackage{fourier}'
        rcParams['font.family'] = 'Open Sans'
        rcParams['font.serif'] = 'TeX Gyre Bonum Math'
        rcParams['figure.dpi'] = 120
        rcParams['savefig.dpi'] = 300
        rcParams['savefig.bbox'] = 'tight'
        rcParams['text.hinting_factor'] = 1


def setup_plotstyle(rcParams):
    """
    Sets various defaults for nice and sustainable style of plots
    :param rcParams:
    """
    rcParams['grid.alpha'] = 0.3
    rcParams['legend.framealpha'] = 1.0
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.labelsize'] = 14
    rcParams['figure.autolayout'] = False


def plotsave_transparent(rcParams, transparency=True):
    """
    Sets transparent background for the plots.
    :param rcParams: The global matplotlib rcParams.
    :param transparency: boolean, if True, the plots are saved with transparent background.
    """
    rcParams['savefig.transparent'] = transparency


def set_log_axes(axes: plt.Axes, aset=False, aspect=0.36):
    """
    Sets log-log scale for an Axis object. Adds 1-3-10 major ticks for the X-axis. Enables grid with 30% alpha.
    :param aset: True if aspect must be set.
    :param aspect: aspect to set.
    :param axes: an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_xscale("log", nonpositive='clip', subs=[2, 3, 4, 5, 6, 7, 8, 9])
    axes.set_yscale("log", nonpositive='clip')
    axes.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3,
                     10, 30, 50, 100, 300, 1000, 3000,
                     10000, 30000, 100000, 300000, 1000000],
                    minor=False)
    axes.set_xticklabels([r'0.01', r'0.03', r'0.1', r'0.3', r'1',
                          r'3', r'10', r'30', r'50', r'100', r'300', r'1000', r'3G',
                          r'10G', r'30G', r'100G', r'300G', r'1T'],
                         minor=False)
    axes.grid(True, which='both', alpha=0.3, zorder=0)
    if aset:
        plt.Axes.set_aspect(axes, aspect=aspect, adjustable='box')


def set_log_axes_simple(axes: plt.Axes):
    """
    The same as set_log_axes_noaspect, but without setting X-axis ticks.
    :param axes: an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_xscale("log", nonpositive='clip', subs=[2, 3, 4, 5, 6, 7, 8, 9])
    axes.set_yscale("log", nonpositive='clip')
    axes.grid(True, which='both', alpha=0.3, zorder=0)


def set_time_log_axes_simple(axes: plt.Axes):
    """
    Sets the Y-axis to log scale and enables grid.
    :param axes: an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_yscale("log", nonpositive='clip')
    axes.grid(True, which='both', alpha=0.3, zorder=0)


def set_lin_axes_simple(axes: plt.Axes):
    """
    Sets linear scale for X and Y axes
    :param axes: an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_yscale("linear")
    axes.set_xscale("linear")
    axes.grid(True, which='both', alpha=0.3, zorder=0)


def set_log_axes_2048(axes: plt.Axes):
    """
    Sets log-log scale with the major ticks on the powers of two. The ticks span from 1 to 2048.
    :param axes:  an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_xscale("log", nonpositive='clip', basex=2.0)
    axes.set_yscale("log", nonpositive='clip', basey=2.0)
    axes.grid(True, which='both', alpha=0.3, zorder=0)
    axes.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], minor=False)
    axes.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], minor=False)
    axes.set_xticklabels([r'1', r'2', r'4', r'8', r'16', r'32', r'64', r'128', r'256', r'512', r'1024', r'2048'], minor=False)
    axes.set_yticklabels([r'1', r'2', r'4', r'8', r'16', r'32', r'64', r'128', r'256', r'512', r'1024', r'2048'], minor=False)
    axes.set_ylim(1, 2048)
    axes.set_xlim(1, 2048)


def set_log_axes_bin16(axes: plt.Axes):
    """
    Sets log-log scale with the major ticks on the powers of two. The ticks span from 1 to 65536.

    :param axes: an Axis object to operate on.
    """
    axes.tick_params(direction='in', which='both', zorder=4)
    axes.set_xscale("log", nonpositive='clip', basex=2.0)
    axes.set_yscale("log", nonpositive='clip', basey=2.0)
    axes.grid(True, which='both', alpha=0.3, zorder=0)
    axes.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256,
                     512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], minor=False)
    axes.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256,
                     512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], minor=False)
    axes.set_xticklabels([r'1', r'2', r'4', r'8', r'16', r'32', r'64', r'128', r'256',
                          r'512', r'1024', r'2048', r'4096', r'8192', r'16384', r'32768', r'65536'],
                         minor=False)
    axes.set_yticklabels([r'1', r'2', r'4', r'8', r'16', r'32', r'64', r'128', r'256',
                          r'512', r'1024', r'2048', r'4096', r'8192', r'16384', r'32768', r'65536'],
                         minor=False)
    axes.set_ylim(1, 65536)
    axes.set_xlim(1, 65536)


def draw_bar_text(axis, begin, end, ypos, text='text', height=10, color='bisque', xposcorr=0.0):
    """
    Draws a box with text, similar to Gant chart.
    :param axis: an Axis object to operate on.
    :param begin: The X-axis start position of a box
    :param end: The X-axis end position of a box
    :param ypos: The Y-axis position
    :param text: Text to be printed inside the box
    :param height: The height of the box
    :param color: Fill color.
    :param xposcorr: A correction for a misalignment caused by some fonts.
    """
    rect = pt.Rectangle((begin, ypos), end - begin,
                        height, alpha=1, ec='k', fc=color, zorder=0)
    axis.add_artist(rect)
    axis.text(begin + xposcorr + (end - begin) / 2.0, ypos + height / 2, text,
              fontsize=12, ha='center', va='center_baseline',
              alpha=1, zorder=1)
