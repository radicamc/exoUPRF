#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='exouprf',
      version='0.0.1',
      license='BSD 3-Clause License',
      author='Michael Radica',
      author_email='michael.radica@umontreal.ca',
      packages=['exouprf'],
      include_package_data=True,
      url='https://github.com/radicamc/exoUPRF',
      description='Tools for Light Curve Fitting',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['batman-package', 'celerite', 'corner', 'dynesty',
                        'emcee', 'h5py', 'matplotlib', 'numpy', 'scipy'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
