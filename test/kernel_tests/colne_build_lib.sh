#!/bin/bash

git clone https://github.com/wimrijnders/V3DLib.git
git clone https://github.com/wimrijnders/CmdParameter.git

cd CmdParameter
ls # change to the build command
cd ..

cp generate.sh V3DLib/script
cp make_kernels V3DLib
  
cd V3DLib
./script/generate.sh
ls # change to the build command
  
cd ..
