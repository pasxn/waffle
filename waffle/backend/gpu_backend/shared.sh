#!/bin/bash

machine_arch=$(uname -m)

if [[ $machine_arch == "x86_64" ]]; then
  cd target/emu-debug/bin
else
  cd target/qpu/bin
fi

file_list=$(find . -type f -name "*.o")

for file in $file_list; do
  new_file=$(echo "$file" | sed 's/\.o$/.so/')
  g++ -shared -o $new_file -fPIC $file
done
