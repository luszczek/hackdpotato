
#include <iostream>
#include <map>

#include "load.hpp"
#include "parse.hpp"

// this value is per single node
#define MAX_SUPPORTED_GPUS 16

/* specifying # of threads for a given block, 256 block threads (index 0 to 255) */
const int BLOCK_SIZE=256;

struct DevicePointers {
  int *ptrs_d, *breaks_d;
  float *rands_d, *Vs_d, *tset_d, *tgts_d, *wts_d, *xx_d, *scores_d, *areas_d;
};

struct Parameters {
  int pSize; // "Population Size (pSize): " << pSize << "\n\n";
  int nGen; // "Number of Generations (nGen): " << nGen << "\n\n";
  float pMut; // "Probability of Mutations (pMut): " << pMut << "\n\n";
  float max; // "Maximal permissible mutation (max): " << max << "\n\n";
  float pCross; // "Probability of crossover (pCross): " << pCross << "\n\n";
  int rseed; // "Random seed (rseed): " << rseed << "\n\n";
  int peng; // "Print scores every  " << peng << "generations (peng)\n\n";
  int ncp; // "Print scores of only " << ncp << " chromosomes every peng \n\n";

/* Hardcoding these input but we will make user options 
  nisland is the number of subpopulation, iTime is the isolation time, nMig is the number of
  migrants added to migrant pool. nEx number of exchange btwn migrant pool and subpop */
  int nIsland, iTime, nMig, nEx;

  int genomeSize, nConf, N;
  float *Vs, *tset, *tgts, *wts, *scores; 
  int *ptrs, *breaks, nBreaks;
  size_t nRands; 
  
  DevicePointers devicePointers[MAX_SUPPORTED_GPUS];
};

void ParseArgs(int argc, char *argv[], Parameters &parameters) {
  if (!(argv[1]=="-p")) std::cout << "please use -p for param file";
  ConfigFile cfg(argv[2]);

  // check if keys exixt
  if (!(cfg.keyExists("pSize"))) std::cout << "oops you forgot pSize"; //make it for others  

  //add the rest of parameters if exist as line above 
  // Retreive the value of keys pSize if key dont exist return value will be 1
  //add new parameters here
  parameters.pSize = cfg.getValueOfKey<int>("pSize", 1);
  std::cout << "Population Size (pSize): " << parameters.pSize << "\n\n";
  parameters.nGen = cfg.getValueOfKey<int>("nGen", 1);
  std::cout << "Number of Generations (nGen): " << parameters.nGen << "\n\n";
  parameters.pMut = cfg.getValueOfKey<float>("pMut", 1);
  std::cout << "Probability of Mutations (pMut): " << parameters.pMut << "\n\n";
  parameters.max = cfg.getValueOfKey<float>("max", 1);
  std::cout << "Maximal permissible mutation (max): " << parameters.max << "\n\n";
  parameters.pCross = cfg.getValueOfKey<float>("pCross", 1);
  std::cout << "Probability of crossover (pCross): " << parameters.pCross << "\n\n";
  parameters.rseed = cfg.getValueOfKey<int>("rseed", 1);
  std::cout << "Random seed (rseed): " << parameters.rseed << "\n\n";
  parameters.peng  = cfg.getValueOfKey<int>("peng", 1);
  std::cout << "Print scores every  " << parameters.peng << "generations (peng)\n\n";
  parameters.ncp  = cfg.getValueOfKey<int>("ncp", 1);
  std::cout << "Print scores of only " << parameters.ncp << " chromosomes every peng \n\n";

  /*specify the string of the savefile, scorefile, loadfile name */
  std::string saveFile,loadFile,scoreFile;

  /* dealing with loading the input file and save file string name */
  for (int i=2;i<argc;i++){
    if(i+1<argc){
      if(argv[i][0]=='-'&&argv[i][1]=='r')saveFile=argv[++i];
      else if(argv[i][0]=='-'&&argv[i][1]=='c')loadFile=argv[++i];
      else if(argv[i][0]=='-'&&argv[i][1]=='s')scoreFile=argv[++i];
    }
  }


}

void LoadParameters(Parameters &parameters, std::map<std::string,DihCorrection> &correctionMap) {
  load(std::cin,
    &parameters.tset,
    &parameters.tgts,
    &parameters.wts,
    &parameters.nConf,
    &parameters.breaks,
    &parameters.nBreaks,
    &parameters.genomeSize,
    correctionMap);
}

void AllocateArrays(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

  cudaMalloc((void **)&parameters.devicePointers[gpuDevice].breaks_d, parameters.nBreaks*sizeof(int));
  cudaMalloc((void **)&parameters.devicePointers[gpuDevice].tgts_d, (parameters.nBreaks-1+parameters.nConf*(1+parameters.genomeSize))*sizeof(float));
  parameters.devicePointers[gpuDevice].wts_d=parameters.devicePointers[gpuDevice].tgts_d+parameters.nConf;
  parameters.devicePointers[gpuDevice].tset_d=parameters.devicePointers[gpuDevice].wts_d+parameters.nBreaks-1;

/**********************| initiate GPU blocks and # of random variable |*************************** 
*          we need randoms, new pop 3xcrossover, genomeSizexmut                                  *    
*        genome size is the number of genes which is all the parameters,                         *
*   e.g for 4 periodicity and three dihedral fitting, then genomesize will be 4 * 3 = 12         *
*   nRands is number of randoms we need for each set of parameters                               *
*   e.g if psize (population size) is 10, then number of random number we will need is           *
*                   (3+(# of periodicity x # of dihedral)) * psize                               *
* so for 4 periodicity and 3 dihedral fitting (chi1 chi2 chi3), then nRands = 3+12 * 10 = 150    *
*________________________________________________________________________________________________*  
*  nBlocks is dependent on the population size, it is use to figure out how many GPU blocks      *
*  we need to initialize the arrays for calculations. Each block has 256 threads.                *
*  one thread represent one individual (chromosome with soln parameters) from the population     *
*   e.g population size of 2000 will require (2000+256-1)/256 = 8.81 => 8 blocks                 *
*                                                                                                *
*************************************************************************************************/
  parameters.nRands=(3+parameters.genomeSize)*parameters.pSize;
  int nBlocks=(parameters.pSize+BLOCK_SIZE-1)/BLOCK_SIZE;

/*******************************| initializing more host and device variables|************************
*         N (bitwise operation below) is the pSize (1st input) multiply by 2;                   *
*       initiating the chromosomes  which have the solns                                        *
************************************************************************************************/

  float *rands=(float *)malloc(parameters.nRands*sizeof(float));
  //cudaMalloc((void **)&rands_d, nRands*sizeof(float));
  parameters.N=(parameters.pSize<<1);
  error=cudaMalloc((void **)&parameters.devicePointers[gpuDevice].Vs_d, (parameters.N*(parameters.genomeSize+4)+parameters.pSize*parameters.nConf+parameters.nRands)*sizeof(float));
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  parameters.devicePointers[gpuDevice].rands_d=parameters.devicePointers[gpuDevice].Vs_d+parameters.N*parameters.genomeSize;
  parameters.devicePointers[gpuDevice].scores_d=parameters.devicePointers[gpuDevice].rands_d+parameters.nRands;
  parameters.devicePointers[gpuDevice].areas_d=parameters.devicePointers[gpuDevice].scores_d+(parameters.N<<1);
  parameters.devicePointers[gpuDevice].xx_d=parameters.devicePointers[gpuDevice].areas_d+(parameters.N<<1);
  parameters.scores=(float *)malloc(sizeof(*parameters.scores)*parameters.N);
  float *scores_ds[2];
  scores_ds[0]=parameters.devicePointers[gpuDevice].scores_d;
  scores_ds[1]=parameters.devicePointers[gpuDevice].scores_d+parameters.N;

  parameters.Vs=(float *)malloc(parameters.N*parameters.genomeSize*sizeof(float));
  /*allocate the memory space to hold array of pointers (prts) of size N (2*pSize)
  these pointers point to the individuals (chromosome) in the population */
  parameters.ptrs=(int *)malloc(sizeof(int)*parameters.N);
  parameters.ptrs[0]=0;
  for (int g=1;g<parameters.N;g++)parameters.ptrs[g]=parameters.ptrs[g-1]+parameters.genomeSize;
  error = cudaMalloc((void **)&parameters.devicePointers[gpuDevice].ptrs_d, parameters.N*2*sizeof(int));
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  int *ptrs_ds[2];
  ptrs_ds[0]=parameters.devicePointers[gpuDevice].ptrs_d;
  ptrs_ds[1]=parameters.devicePointers[gpuDevice].ptrs_d+parameters.N;
}


void CopyArrays(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

/* 
copying over the arrays from the CPU to GPU
nbreaks is the # of dataset + 1. e.g if you are doing alpha and beta backbone set then nbreaks=3
genomesize is the # of fitting dihedral * periodicity, e.g 3 set of dihedral * 4 periodicity = 12
nconf is the # of conformations you are fitting
tset is (E_QMi-E_MMi) + (E_MMref-E_QMref) for each conformation, which = nconf, see load.cpp
tgts is the cos(dih*periodicity) for 4 periodicity for a dihedral for each conformation
so 20 conf will give tgts of 20 (nconf) * 12 (# of dih * periodicity) = 120 
*/

  error = cudaMemcpy(parameters.devicePointers[gpuDevice].breaks_d, parameters.breaks, parameters.nBreaks*sizeof(parameters.breaks[0]), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.devicePointers[gpuDevice].tset_d, parameters.tset, parameters.nConf*parameters.genomeSize*sizeof(float), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.devicePointers[gpuDevice].tgts_d, parameters.tgts, parameters.nConf*sizeof(float), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.devicePointers[gpuDevice].wts_d, parameters.wts, (parameters.nBreaks-1)*sizeof(*parameters.wts), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.devicePointers[gpuDevice].ptrs_d, parameters.ptrs, sizeof(int)*parameters.N, cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
}

int
main(int argc, char *argv[]) {
  Parameters parameters;
  std::map<std::string,DihCorrection> correctionMap;
  int maxGpuDevices = 2;

  ParseArgs(argc, argv, parameters);

  LoadParameters(parameters, correctionMap);

  for (int device = 0; device < maxGpuDevices; device++)
    AllocateArrays(device, parameters);

  for (int device = 0; device < maxGpuDevices; device++)
    CopyArrays(device, parameters);
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateNormal(gen, Vs_d, N*genomeSize, 0, 1);


  return 0;
}
