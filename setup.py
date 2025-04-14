#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='exouprf',
      version='1.0.2',
      license='MIT',
      author='Michael Radica',
      author_email='radicamc@uchicago.edu',
      packages=['exouprf'],
      include_package_data=True,
      url='https://github.com/radicamc/exoUPRF',
      description='Tools for Light Curve Fitting',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['batman-package', 'celerite', 'corner', 'dynesty',
                        'emcee', 'h5py', 'matplotlib', 'numpy',
                        'scipy'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
