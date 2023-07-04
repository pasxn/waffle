#!/bin/bash

cd V3DLib
make --file=make_kernels QPU=0 DEBUG=1 $1