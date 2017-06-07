
#include <cstdlib>

#include <iostream>

#include <cuda.h>

__global__ void increment(float *val)
{
  val[0]++;
}

void
run_par_gpu() {
  int device;
  const int ngpu = 2;
  float *values[ngpu], currentDevice, *fromDevice;

  fromDevice = (float *)malloc(ngpu * sizeof(float));

  for (device = 0; device < ngpu; device++) {
    cudaSetDevice(device);

    cudaMalloc((void **)&(values[device]), 1 * sizeof(float));

    currentDevice = device;

    cudaMemcpy(values[device], &currentDevice, 1*sizeof(float), cudaMemcpyHostToDevice);

    increment<<<1,1>>>(values[device]);

    cudaMemcpy(&fromDevice[device], values[device], 1*sizeof(float), cudaMemcpyDeviceToHost);
  }

  for (device = 0; device < ngpu; device++) {
    std::cout << fromDevice[device] << std::endl;
  }
}

int
main(void) {
  run_par_gpu();
  return 0;
}
