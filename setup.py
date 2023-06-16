#!/usr/bin/env python

import os
import platform
from setuptools import setup
from waffle import __version__


def clone_build_v3dlib():
  current_dir = os.getcwd()
  os.chdir('waffle/backend/gpu_backend')
  os.system('git clone https://github.com/wimrijnders/V3DLib.git')
  os.system('git clone https://github.com/wimrijnders/CmdParameter.git')

  os.chdir('CmdParameter')
  os.system('ls') # change to the build command
  os.chdir('..')

  os.system('cp generate.sh V3DLib/script')
  os.system('cp make_kernels V3DLib')
  
  os.chdir('V3DLib')
  os.system('./scripts/generate.sh')
  os.system('ls') # change to the build command
  
  os.chdir(current_dir)

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), encoding='utf-8') as r:
  requirements = [req.strip() for req in r.readlines()]

processor = platform.processor()
if "armv7l" in processor and "BCM2711" in processor:
  clone_build_v3dlib()

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
  python_requires='>=3.8',
  include_package_data=True
)
