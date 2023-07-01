#!/bin/bash

cd . # part to directory should be here

file_list=$(find . -type f -name "*.sh")

for file in $file_list; do
  new_file="${file%%.*}.so" # passe hadapan
  # g++ -shared -o $new_file -fPIC $file
  echo $new_file
  echo $file
done
