#!/bin/bash

nvcc -arch=sm_20 $([[ "$1" == "-d" ]] && echo -n " -DDEBUG=3 -DTHRUST_DEBUG -G -g") -o genA{,.cu} -lcurand
