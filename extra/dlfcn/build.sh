#!/bin/bash

g++ -c lib.cpp -fPIC -o lib.o
g++ -shared -o lib.so -fPIC lib.o

g++ -c main.cpp -fPIC -o main.o
g++ -shared -o main.so -fPIC main.o

g++ -o main.out main.cpp -ldl
