#!/bin/bash

cd V3DLib
script/gen.sh

filename="sources.mk"

mv "${filename}" "../${filename}"

