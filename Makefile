# -*- Makefile -*-

CFLAGS_DEBUG = -DDEBUG=3 -DTHRUST_DEBUG -G -g

CFLAGS_GPROF = -Xcompiler "-g -pg"
# CC = nvcc
CC = nvcc
CFLAGS = -arch=sm_20 #$(CFLAGS_GPROF)  #$(CFLAGS_DEBUG)

LDFLAGS = -G -g
LIBS = -lcurand

.PHONY: all
all: genA

genA: genA.cu
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o genA parmfile.* scores.*.dat *scorep_init.c try.*.frcmod
