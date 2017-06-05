# -*- Makefile -*-

CC_DEBUG_FLAGS = -DDEBUG=3 -DTHRUST_DEBUG -G -g

CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_20 

LDFLAGS = -G -g
LDLIBS = -lcurand

genA: genA.o parse.o load.o

genA.o: genA.cu
	$(CC) -c $< -O $@
