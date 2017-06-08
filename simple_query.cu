
#include <stdio.h>

#include <cuda.h>
#include <curand.h>

#include <mpi.h>

int WorldRank, WorldSize;

void
query_one_device(int device) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("device[%d].major %d.\n", device, deviceProp.major);
  printf("device[%d].minor %d.\n", device, deviceProp.minor);
}

void
query_all() {
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  for (int device=0; device < deviceCount; ++device)
    query_one_device(device);
}

void
random_generate() {
  curandGenerator_t gen;
  int seed=1;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, NULL, 1);
  curandDestroyGenerator(gen);
}

int
main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &WorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &WorldRank);

  for (int rank = 0; rank < WorldSize; ++rank) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == WorldRank) {
      printf("%d/%d: ", WorldRank, WorldSize );
      query_all();
      fflush(stdout);
    }
  }

  MPI_Finalize();
  return 0;
}
