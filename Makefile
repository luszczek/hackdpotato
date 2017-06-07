# -*- Makefile -*-

CFLAGS_DEBUG = -DDEBUG=3 -DTHRUST_DEBUG -G -g

CFLAGS_GPROF = -Xcompiler "-g -pg"
# CC = nvcc
CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_20 -Xcompiler -fopenmp #$(CFLAGS_GPROF)  #$(CFLAGS_DEBUG)

LDFLAGS = -G -g
LIBS = -lcurand -lgomp

all: genA genAmultigpu

genA: genA.cu parse.o 
	$(CC) $(CFLAGS) $< parse.o -o $@ $(LDFLAGS) $(LIBS)

genAmultigpu: genAmultigpu.o load.o parse.o
	$(CC) $(LDFLAGS) $< load.o parse.o -o $@ $(LIBS)

genAmultigpu.o: genAmultigpu.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o genA parmfile.* scores.*.dat *scorep_init.c try.*.frcmod

.PHONY: clean all
