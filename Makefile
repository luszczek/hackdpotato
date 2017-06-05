# -*- Makefile -*-

CFLAGS_DEBUG = -DDEBUG=3 -DTHRUST_DEBUG -G -g

CC = nvcc
CFLAGS = -arch=sm_20  #CFLAGS_DEBUG

LDFLAGS = -G -g
LIBS = -lcurand

.PHONY: all
all: genA

genA: genA.cu
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o genA parmfile.* scores.*.dat *scorep_init.c try.*.frcmod
