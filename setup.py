#!/usr/bin/env python

import os
from setuptools import setup

__version__ = '0.0.4'

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), encoding='utf-8') as r:
  requirements = [req.strip() for req in r.readlines()]

setup(name='waffle',
  version=__version__,
  description='A simple machine learning inference framework for the edge',
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
  extras_require={'testing':['torch', 'pytest']},
  python_requires='>=3.8',
  include_package_data=True
)
