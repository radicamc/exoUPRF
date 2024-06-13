#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:35 2024

@author: MCR

Miscellaneous tools.
"""

from datetime import datetime
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
