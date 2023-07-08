#!/usr/bin/env python

import os
from setuptools import setup
from libv3dac import soc, get_kernels, clone_build_v3dlib

__version__ = '0.0.3'

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), encoding='utf-8') as r:
  requirements = [req.strip() for req in r.readlines()]

# kernels = get_kernels()
# clone_build_v3dlib(kernels, soc)

setup(name='waffle',
  version=__version__,
  description='A hardware accelerated machine learning inference framework for Raspberry Pi',
  author='Pasan Perera, Kavin Amantha, Afkar Ahamed',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages = ['waffle'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License"
  ],
  install_requires=requirements,
  python_requires='>=3.7',
  include_package_data=True
)
