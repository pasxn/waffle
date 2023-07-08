#!/bin/bash

g++ -c lib.cpp -fPIC -o lib.o
g++ -shared -o lib.so -fPIC lib.o

gcc -shared -o main.so main.c -ldl
gcc -o output main.c -ldl -g
