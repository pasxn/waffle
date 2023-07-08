#!/usr/bin/env python

import os
import platform


def get_kernels():
  kernels = []; directory = os.path.dirname(os.path.abspath(__file__)) + '/waffle/backend/gpu_backend/kernels'

  for files in os.walk(directory):
    for file in files[2]:
      if file.endswith(".cpp") and not any(char.isupper() for char in file):
        kernels.append(file.split('.')[0])
  
  return kernels

def clone_build_v3dlib(kernels, soc):
  current_dir = os.getcwd()
  os.chdir('waffle/backend/gpu_backend')
  os.system('git clone https://github.com/wimrijnders/V3DLib.git')
  os.system('git clone https://github.com/wimrijnders/CmdParameter.git')

  os.chdir('CmdParameter')
  os.system('make DEBUG=1 all' if soc == 'X86' else 'make DEBUG=0 all')
  os.chdir('..')

  os.system('cp generate.sh V3DLib/script')
  os.system('cp make_kernels V3DLib')
  os.system('cp makefile V3DLib/mesa')

  
  os.chdir('V3DLib')
  os.system('./script/generate.sh')
  os.system('make --file=make_kernels DEBUG=1 QPU=0 all' if soc == 'X86' else 'make --file=make_kernels DEBUG=0 QPU=1 all')

  for kernel in kernels:
    os.system('make --file=make_kernels DEBUG=1 QPU=0 ' + kernel if soc == 'X86' else 'make --file=make_kernels DEBUG=0 QPU=1 ' + kernel)
  
  os.chdir('..')
  os.system('./shared.sh')
  
  os.chdir(current_dir)

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), encoding='utf-8') as r:
  requirements = [req.strip() for req in r.readlines()]

processor = platform.processor()
soc = 'QPU' if "armv7l" in processor and "BCM2711" in processor else 'X86'

kernels = get_kernels()
clone_build_v3dlib(kernels, soc)
