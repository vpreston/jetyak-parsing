#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 9 2021
@author: vpreston
"""

from setuptools import setup

setup(name='ParseYak',
      version='0.1',
      description='functions for parsing data from a ChemYak platform',
      url='http://github.com/vpreston/',
      author='Victoria Preston',
      author_email='vpreston@whoi.edu',
      license='MIT',
      packages=['parseyak', 'parseyak.utils'],
      zip_safe=False)
