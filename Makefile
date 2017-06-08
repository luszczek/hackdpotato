# -*- Makefile -*-

CFLAGS_DEBUG = -DDEBUG=3 -DTHRUST_DEBUG -G -g

CFLAGS_GPROF = -Xcompiler "-g -pg"
NVCC = nvcc
CC = nvcc
CXX = nvcc
CFLAGS = -arch=sm_20 -Xcompiler -fopenmp #$(CFLAGS_GPROF)  #$(CFLAGS_DEBUG)
MPIFLAGS = -I/hpcgpfs01/software/openmpi/openmpi-1.10.2/usr/include -L/hpcgpfs01/software/openmpi/openmpi-1.10.2/usr/lib -lmpi

LDFLAGS = -G -g
LIBS = -lcurand -lgomp

all: genA genAmultigpu test

genA: genA.cu parse.o 
	$(CC) $(CFLAGS) $< parse.o -o $@ $(LIBS)

genAmultigpu: genAmultigpu.o load.o parse.o
	$(CC) $(LDFLAGS) $< load.o parse.o -o $@ $(LIBS)

genAmultigpu.o: genAmultigpu.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

simple_query: simple_query.o
	$(NVCC) $(CFLAGS) $(MPIFLAGS) $< -o $@

simple_query.o: simple_query.cu
	$(NVCC) $(CFLAGS) $(MPIFLAGS) -c $< -o $@

test: test.o load.o parse.o
	$(CC) $(MPIFLAGS) $(LDFLAGS) $< load.o parse.o -o $@ $(LIBS)

test.o: test.cu
	$(CC) $(MPIFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o genA parmfile.* scores.*.dat *scorep_init.c try.*.frcmod

.PHONY: clean all
