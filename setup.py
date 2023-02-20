#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='waffle',
  version='0.0.1',
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
  install_requires=['numpy'],
  python_requires='>=3.8',
  include_package_data=True
)
