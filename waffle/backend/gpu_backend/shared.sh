#!/bin/bash

cd . # part to directory should be here

file_list=$(find . -type f -name "*.o")

for file in $file_list; do
  new_file=$(echo "$file" | sed 's/\.o$/.so/')
  g++ -shared -o $new_file -fPIC $file
done
