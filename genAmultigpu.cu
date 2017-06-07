
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>

#include <cuda.h>
#include <curand.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "load.hpp"
#include "parse.hpp"

// this value is per single node
#define MAX_SUPPORTED_GPUS 16

/* specifying # of threads for a given block, 256 block threads (index 0 to 255) */
const int BLOCK_SIZE=256;

/*************************************************************************************************
                  Defining the six pivotal functions for the genetic algorithm
(1) mateIt, (2) mutateIt, (3) scoreIt, (4) calcAreas, (5) moveEm, (6) getSumAreas
getSumAreas uses two other functions sumEm and sumEmIndex
*************************************************************************************************/

/************************************************************************************************
                                | function1: mateIt | 

 *  creates offspring from a population, generating crossovers according to pCross
 *
 *  @param Vs a global array of all the parent and child genomes
 *  @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
 *  @param areas the probabilities for choosing each individual for mating
 *  @param sumArea pointer to the sum of all the individual areas
 *  @param rands array of random numbers
 *  @param pCross probability that crossover occurs
 *  @param pSize number of individuals in the population
 *  @param genomeSize number of genes in a genome
************************************************************************************************/

__global__ void mateIt(float *Vs, int *ptrs, const float *areas, const float *sumArea, const float *rands, const float pCross, const int pSize, const int genomeSize)
{
/* 
figure out index  blockId.x is the index by blocks, blockDIM.x is the elements per blocks (# of threads ina block)
threadIdx is the index for threads
*/
  int i=blockIdx.x * blockDim.x + threadIdx.x;
/* 
first parent, second parent, crossover random numbers 
randi is three arrays with block and thread index of randoms numbers from 0 to 255;
The cross over random numbers 
*/
  int randi=i*3;
//multiply i by 2, as we will have 2 parents and 2 offspring multiplication is done using a left bitwise (<<) by 1
  i<<=1;
/* 
if we're in the population (sometimes warps may go past) 
The statement if (i<psize) is common in cuda:
  Before i index is used to access array elements, its value is checked against the number of elements, n, 
 to ensure there are no out-of-bounds memory accesses. This check is required for cases where the 
 number of elements in an array is not evenly divisible by the thread block size, 
  and as a result the number of threads launched by the kernel is larger than the array size. 
*/
  if(i<pSize){
    int parent[2];
    int j;
/* figure out parents */
    parent[0]=parent[1]=-1;
/* 
find parent where cumulative (cum) area (A) is less than random target (tgt) area
*/
    float cumA=0.0f, tgtA=rands[randi++]* *sumArea;
    while(cumA<=tgtA){
      ++parent[0];
      cumA+=areas[ptrs[parent[0]]/genomeSize];
/* rands[randi-1] is the index back to zero since it is the first set of parents */
    }
  #if DEBUG>2
    printf("rands[%d] ; %f ; %f=%f * %f\n",randi, cumA, tgtA, rands[randi-1], *sumArea);
    printf("first parent\n");
  #endif
    cumA=0.0f; tgtA=rands[randi++]*
          (*sumArea-areas[ptrs[parent[0]]/genomeSize]);
    while(cumA<=tgtA){
      ++parent[1];
      if(parent[1]==parent[0])
        ++parent[1];
      cumA+=areas[ptrs[parent[1]]/genomeSize];
    }
  #if DEBUG>2
    printf("Make offspring %d from %d and %d (%f=%f*(%f-%f)) %d\n", i, parent[0], parent[1], tgtA, rands[randi-1], *sumArea, areas[ptrs[parent[0]]/genomeSize], randi);
  #endif
    /* add offset of pSize to i because it is a child (next population) */
    i+=pSize;
    /* use ptrs to get indices into Vs */
    int i0=ptrs[i], i1=ptrs[i+1];
    parent[0]=ptrs[parent[0]];
    parent[1]=ptrs[parent[1]];
    /* set j to index for the next set of Vs */
    j=i0+genomeSize;
    /* put parent[0], parent[1], and i1 relative to i0, so we can just add i0 for index */
    parent[0]-=i0;
    parent[1]-=i0;
    i1-=i0;
    /* start with crossover pt at the end (no crossover) */
    int crossPt=j;
    /* check for crossover */
    if(rands[randi]<pCross){
      crossPt=i0+1+(int)(rands[randi]/pCross*(float)(genomeSize-1));
    }
    while(i0<crossPt){
    /* load next bit from parent and increment i */
      Vs[i0]=Vs[parent[0]+i0];
      Vs[i1+i0]=Vs[parent[1]+i0];
      ++i0;
    }
    while(i0<j){
      Vs[i0]=Vs[parent[1]+i0];
      Vs[i1+i0]=Vs[parent[0]+i0];
      ++i0;
    }
  }
}

/************************************************************************************************
                                | function 2: mutateIt |

 * @brief introduces mutations to the genomes in Vs, according to probability pMut, with a max perturbation of maxMut
 *
 * @param Vs a global array of all the parent and child genomes
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
   @param rands array of random numbers
 * @param pSize number of individuals in the population
 * @param pMut probability that a mutation occurs, evaluated for each gene
 * @param maxMut maximum perturbation to an allele
 * @param genomeSize number of genes in a genome
*************************************************************************************************/

__global__ void mutateIt(float *Vs, int *ptrs, const float *rands, const int pSize, const float pMut, const float maxMut, const int genomeSize)
{
/* figure out index */
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<pSize){
    // get index into random number array
    int r=i*genomeSize;
    i=ptrs[i];
    int j=i+genomeSize;
    // want random numbers from [-maxMut, maxMut). will subtract maxMut later
    float scale=2.0f*maxMut/pMut;
    // iterate through genome
    while(i<j){
      if(rands[r]<pMut){
        // mutate the amplitude by adding perturbation
        Vs[i]+=rands[r]*scale-maxMut;
      }
      ++i;
      ++r;
    }
  }
}


/************************************************************************************************
                                | function 3: scoreIt | 

 * @brief calculates a score indicating the closeness of fit for each individual/chromosome (set of parameters) against the training set

 * @param scores score for each conformation, calculated here
 * @param areas weighting for each conformation, was formerly calculated here
 * @param Vs a global array of all the parent and child genomes
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
 * @param tset training set
 * @param tgts targets for training
 * @param wts weights of each point in the training set
 * @param breaks breaks in training set, where different data should not be compared across breaks
 * @param nConf number of conformations in training set
 * @param pSize number of individuals in the population
 * @param genomeSize number of genes in a genome
 * @param xx space to store energy differences for each conformation with test parameters
************************************************************************************************/

__global__ void scoreIt(float *scores, float *areas, const float *Vs, const int *ptrs, const float *tset, const float *tgts, const float *wts, const int *breaks, const int nConf, const int pSize, const int genomeSize, float *xx)
{
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  //if((i<<1)<(pSize-1)*pSize){
  if(i<pSize){
  float *x=xx+i*nConf;  // for the error of each conformation
  // get reference to score
  float *S=scores+i;
  // set score to 0
  *S=0.0f;
  // accumulate little s for each set
  float s;
  // get first index in genome
  int i0=ptrs[i];
  // get index of next genome space for looping bounds
  int j=i0+genomeSize;
  // start with the first element in the training set
  int t=0;
  /* start at break 0 */
  int b=0;
  /* loop over conformations c */
  int c=0;
  while(c<nConf){
    //int nP=0;
    s=0.0f;
    /* loop only in units without break points */
    while(c<breaks[b+1]){
  /* 
  start with delta E (tgts) for a given conformation (c) within a break; see load.cpp 
  conf (c) goes through until it reach a break. the loop will set delta E 
  */
      x[c]=tgts[c];
  /* 
  subtract contributions from each parameter for conformation c 
  for each conformation 
  e.g deltaE - cos (dihedral * periodicity) * parameter generated from chromosomes
  */
      for(i=i0;i<j;i++,t++){
        x[c]-=tset[t]*Vs[i];
      }
      /* add differences in this error from all other errors */
      for(int c2=breaks[b];c2<c;c2++){
        float err=x[c]-x[c2];
        s+=(err<0.0f?-err:err);
      }
      /* next conformation */
      ++c;
    }
    /* add little error to big error S, weighted by number of pairs */
    *S+=s*wts[b];
    /* go to next breakpoint */
    ++b;
  }
#if DEBUG>1
  printf("areas[%d]=%f\n",i0/genomeSize,areas[i0/genomeSize]);
#endif
  }
}

/**************************************************************************************************
*                                 | function 4: calcAreas |                                       *
*                                                                                                 *
*     calculates the areas (the probability) each individual has of mating                        *
*___________________________________Parameters____________________________________________________*
* @param scores scores for each individual (set of parameters)                                    *
* @param areas fitness for each individual, in terms of probability of mating                     *
* @param ptrs array of pointers from logical indices to actual indices into Vs for each individual*
* @param pSize number of individuals in the population                                            *
* @param genomeSize number of genes in a genome                                                   *
**************************************************************************************************/

__global__ void calcAreas(float *scores, float *areas, const int *ptrs, const int pSize, const int genomeSize) {
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<pSize){
    areas[ptrs[i]/genomeSize]=__expf(-scores[i]/scores[0]);
  }
}

/************************************************************************************************
*                                | function 5: moveEm |
*
* @brief simple helper function for copying data from oldF, oldI to neWF, newI
*
* @param newF pointer to new float array
* @param newI pointer to new int array
* @param oldF pointer to old float array
* @param oldI pointer to old int array
* @param N number of floats/ints to copy
*************************************************************************************************/

__global__ void moveEm(float * newF, int *newI, float *oldF, int *oldI, int N) {
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<N){
    newF[i]=oldF[i];
    newI[i]=oldI[i];
  }
}

/************************************************************************************************
                   | sumEm and sumEmIndex : helper function for getSumAreas |

* @brief performs a sum of each successive pair of N numbers in source and stores the sums in sums. intended to be run multiple times to sum over a whole array. if N is odd, the last sum index will be N/2-1 and contain the sum of the last 3 numbers
*
* @param sums where to store the sums
* @param source where to get the numbers to sum together
* @param N the dimension of source
*
* @return */

__global__ void sumEm(float *sums, float *source, int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=(i<<1);
  if(j+3<N)sums[i]=source[j]+source[j+1];
  else if(j+3==N) sums[i]=source[j]+source[j+1]+source[j+2];
  else if(j+2==N) sums[i]=source[j]+source[j+1];
}

/*
* @brief performs a sum of pairs of N numbers in source, using locations indicated by pointers. pointers has indices multiplied by genomeSize. intended to be run multiple times to sum over a whole array. if N is odd, the last sum index will be N/2-1 and contain the sum of the last 3 numbers
*
* @param sums where to store the sums
* @param source where to get the numbers to sum together
* @param N the dimension of source
* @param ptrs the indices to use when gathering pairs for summation
* @param genomeSize the number by which the indices in ptrs are scaled
*
* @return 
*/
__global__ void sumEmIndex(float *sums, float *source, int N, const int *ptrs, const int genomeSize){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=(i<<1);
  if(j+3<N)sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize];
  else if(j+3==N) sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize]+source[ptrs[j+2]/genomeSize];
  else if(j+2==N) sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize];
#if DEBUG>1
  if(j+2<=N)printf(" %d:%f",i,sums[i]);
#endif
}
/*******************************| end of helper function |***************************************/
/*************************************************************************************************
*                                | function 6: getSumAreas |                                     * 
*                        ---------uses sumEmIndex and sumEM--------                              *
*                                                                                                *
* @brief get sum of all areas
*                                                                                                *
* @param areas_d pointer to areas on device                                                      *
* @param ptrs_d pointer to indices for each individual in population
* @param pSize population size
* @param temp_d pointer to temporary array on device
* @param genomeSize number of alleles in genome
************************************************************************************************/

float *getSumAreas(float *areas_d, int *ptrs_d, int pSize, float *temp_d, const int & genomeSize){
  int dim=pSize;
  int offset=0;
/* 
the triple chevron below describes an execution configuration
the first argument(((dim>>1)+BLOCK_SIZE-1)/BLOCK_SIZE) in the execution configuration specifies the 
number of thread blocks in the grid, and the second specifies (BLOCK_SIZE) the number of threads in a thread block 
*/
  sumEmIndex <<<((dim>>1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (temp_d, areas_d, dim, ptrs_d, genomeSize);
#if DEBUG>1
  std::cout << std::endl;
#endif
  pSize >>= 1;
  while((dim>>=1)>1){
    offset^=pSize;
    sumEm <<<((dim>>1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (temp_d+offset, temp_d+(offset^pSize), dim);
#if DEBUG>1
  std::cout << std::endl;
#endif
  }
  return temp_d+offset;
}







struct DeviceParameters {
  int curList;
  int *ptrs_d, *breaks_d;
  float *rands_d, *Vs_d, *tset_d, *tgts_d, *wts_d, *xx_d, *scores_d, *areas_d;
  float *scores_ds[2];
  int *ptrs_ds[2];
  curandGenerator_t gen;
};

struct Parameters {
  int pSize; // "Population Size (pSize): " << pSize << "\n\n";
  int nGen; // "Number of Generations (nGen): " << nGen << "\n\n";
  float pMut; // "Probability of Mutations (pMut): " << pMut << "\n\n";
  float maxMut; // "Maximal permissible mutation (maxMut): " << maxMut << "\n\n";
  float pCross; // "Probability of crossover (pCross): " << pCross << "\n\n";
  int rseed; // "Random seed (rseed): " << rseed << "\n\n";
  int peng; // "Print scores every  " << peng << "generations (peng)\n\n";
  int ncp; // "Print scores of only " << ncp << " chromosomes every peng \n\n";

  /*specify the string of the savefile, scorefile, loadfile name */
  std::string saveFile,loadFile,scoreFile;

/* Hardcoding these input but we will make user options 
  nisland is the number of subpopulation, iTime is the isolation time, nMig is the number of
  migrants added to migrant pool. nEx number of exchange btwn migrant pool and subpop */
  int nIsland, iTime, nMig, nEx;

  int genomeSize, nConf, N, save;
  float *Vs, *tset, *tgts, *wts, *scores; 
  int *ptrs, *breaks, nBreaks;
  size_t nRands; 
  
  int nBlocks;

  DeviceParameters deviceParameters[MAX_SUPPORTED_GPUS];
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
  parameters.maxMut = cfg.getValueOfKey<float>("max", 1);
  std::cout << "Maximal permissible mutation (max): " << parameters.maxMut << "\n\n";
  parameters.pCross = cfg.getValueOfKey<float>("pCross", 1);
  std::cout << "Probability of crossover (pCross): " << parameters.pCross << "\n\n";
  parameters.rseed = cfg.getValueOfKey<int>("rseed", 1);
  std::cout << "Random seed (rseed): " << parameters.rseed << "\n\n";
  parameters.peng  = cfg.getValueOfKey<int>("peng", 1);
  std::cout << "Print scores every  " << parameters.peng << "generations (peng)\n\n";
  parameters.ncp  = cfg.getValueOfKey<int>("ncp", 1);
  std::cout << "Print scores of only " << parameters.ncp << " chromosomes every peng \n\n";

  parameters.save=parameters.pSize/10; // 10% of the population

  /* dealing with loading the input file and save file string name */
  for (int i=2;i<argc;i++){
    if(i+1<argc){
      if(argv[i][0]=='-'&&argv[i][1]=='r')parameters.saveFile=argv[++i];
      else if(argv[i][0]=='-'&&argv[i][1]=='c')parameters.loadFile=argv[++i];
      else if(argv[i][0]=='-'&&argv[i][1]=='s')parameters.scoreFile=argv[++i];
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

  cudaMalloc((void **)&parameters.deviceParameters[gpuDevice].breaks_d, parameters.nBreaks*sizeof(int));
  cudaMalloc((void **)&parameters.deviceParameters[gpuDevice].tgts_d, (parameters.nBreaks-1+parameters.nConf*(1+parameters.genomeSize))*sizeof(float));
  parameters.deviceParameters[gpuDevice].wts_d=parameters.deviceParameters[gpuDevice].tgts_d+parameters.nConf;
  parameters.deviceParameters[gpuDevice].tset_d=parameters.deviceParameters[gpuDevice].wts_d+parameters.nBreaks-1;

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
  parameters.nBlocks=(parameters.pSize+BLOCK_SIZE-1)/BLOCK_SIZE;

/*******************************| initializing more host and device variables|************************
*         N (bitwise operation below) is the pSize (1st input) multiply by 2;                   *
*       initiating the chromosomes  which have the solns                                        *
************************************************************************************************/

  float *rands=(float *)malloc(parameters.nRands*sizeof(float));
  //cudaMalloc((void **)&rands_d, nRands*sizeof(float));
  parameters.N=(parameters.pSize<<1);
  error=cudaMalloc((void **)&parameters.deviceParameters[gpuDevice].Vs_d, (parameters.N*(parameters.genomeSize+4)+parameters.pSize*parameters.nConf+parameters.nRands)*sizeof(float));
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  parameters.deviceParameters[gpuDevice].rands_d=parameters.deviceParameters[gpuDevice].Vs_d+parameters.N*parameters.genomeSize;
  parameters.deviceParameters[gpuDevice].scores_d=parameters.deviceParameters[gpuDevice].rands_d+parameters.nRands;
  parameters.deviceParameters[gpuDevice].areas_d=parameters.deviceParameters[gpuDevice].scores_d+(parameters.N<<1);
  parameters.deviceParameters[gpuDevice].xx_d=parameters.deviceParameters[gpuDevice].areas_d+(parameters.N<<1);
  parameters.scores=(float *)malloc(sizeof(*parameters.scores)*parameters.N);
  parameters.deviceParameters[gpuDevice].scores_ds[0]=parameters.deviceParameters[gpuDevice].scores_d;
  parameters.deviceParameters[gpuDevice].scores_ds[1]=parameters.deviceParameters[gpuDevice].scores_d+parameters.N;

  parameters.Vs=(float *)malloc(parameters.N*parameters.genomeSize*sizeof(float));
  /*allocate the memory space to hold array of pointers (prts) of size N (2*pSize)
  these pointers point to the individuals (chromosome) in the population */
  parameters.ptrs=(int *)malloc(sizeof(int)*parameters.N);
  parameters.ptrs[0]=0;
  for (int g=1;g<parameters.N;g++)parameters.ptrs[g]=parameters.ptrs[g-1]+parameters.genomeSize;
  error = cudaMalloc((void **)&parameters.deviceParameters[gpuDevice].ptrs_d, parameters.N*2*sizeof(int));
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  parameters.deviceParameters[gpuDevice].ptrs_ds[0]=parameters.deviceParameters[gpuDevice].ptrs_d;
  parameters.deviceParameters[gpuDevice].ptrs_ds[1]=parameters.deviceParameters[gpuDevice].ptrs_d+parameters.N;
}


/* 
copying over the arrays from the CPU to GPU
nbreaks is the # of dataset + 1. e.g if you are doing alpha and beta backbone set then nbreaks=3
genomesize is the # of fitting dihedral * periodicity, e.g 3 set of dihedral * 4 periodicity = 12
nconf is the # of conformations you are fitting
tset is (E_QMi-E_MMi) + (E_MMref-E_QMref) for each conformation, which = nconf, see load.cpp
tgts is the cos(dih*periodicity) for 4 periodicity for a dihedral for each conformation
so 20 conf will give tgts of 20 (nconf) * 12 (# of dih * periodicity) = 120 
*/
void CopyArrays(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

  error = cudaMemcpy(parameters.deviceParameters[gpuDevice].breaks_d, parameters.breaks, parameters.nBreaks*sizeof(parameters.breaks[0]), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.deviceParameters[gpuDevice].tset_d, parameters.tset, parameters.nConf*parameters.genomeSize*sizeof(float), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.deviceParameters[gpuDevice].tgts_d, parameters.tgts, parameters.nConf*sizeof(float), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.deviceParameters[gpuDevice].wts_d, parameters.wts, (parameters.nBreaks-1)*sizeof(*parameters.wts), cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  error = cudaMemcpy(parameters.deviceParameters[gpuDevice].ptrs_d, parameters.ptrs, sizeof(int)*parameters.N, cudaMemcpyHostToDevice);
  if(error!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
}

/**************************| Create a random generator |********************************************
* curandCreateGenerator takes two parameters: pointer to generator (*gen), type of generator       *
Once created,random number generators can be defined using the general options seed, offset,& order*
When rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen is CURAND_RNG_PSEUDO_XORWOW            *
*__________________________________________________________________________________________________*
*curandSetPseudoRandomGeneratorSeed takes two parameters (1) the generator (gen) & (2) seed value  *
* seed value # is used to initialize the generator and control the set of random numbers;          *
* same seed will the give same set of random numbers of the psuedorandom generator                 *
* rseed is the random number specified from the 6th input)                                         *
*__________________________________________________________________________________________________*
*    curandGenerateNormal take 5 parameters:                                                       * 
*  (1) generator - Generator to use                                                                *
*  (2) outputPtr - Pointer to device memory to store CUDA-generated results,                       *
                or Pointer to host memory to store CPU-generated resluts                           *
*  (3) num - Number of floats to generate                                                          *
*  (4) mean - Mean of normal distribution                                                          *
*  (5) stddev - Standard deviation of normal distribution                                          *
* Results are 32-bit floating point values with mean and standard deviation.                       * 
***************************************************************************************************/
void GenerateRandom(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

  curandCreateGenerator(&parameters.deviceParameters[gpuDevice].gen, CURAND_RNG_PSEUDO_DEFAULT);

  // initiate the generator with the random seed (rseed)
  curandSetPseudoRandomGeneratorSeed(parameters.deviceParameters[gpuDevice].gen, parameters.rseed);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (normal)\n", cudaGetErrorString(error));}

  curandGenerateNormal(parameters.deviceParameters[gpuDevice].gen, parameters.deviceParameters[gpuDevice].Vs_d, parameters.N*parameters.genomeSize, 0, 1);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (normal)\n", cudaGetErrorString(error));}
}

void LoadAmplitudeParameters(int gpuDevice, Parameters &parameters) {
  // if we have a load file copy Vs (amplitude parameters) from the loaded file and populate Vs
  if(!parameters.loadFile.empty()) {
    std::ifstream loadS(parameters.loadFile.c_str(), std::ios::in | std::ios::binary);
    loadS.read((char*)parameters.Vs,parameters.pSize*parameters.genomeSize*sizeof(*parameters.Vs));
    cudaMemcpy(parameters.deviceParameters[gpuDevice].Vs_d, parameters.Vs, parameters.pSize*parameters.genomeSize*sizeof(*parameters.Vs), cudaMemcpyHostToDevice);
  }
}

/***************************| score of the first set of chromosomes |*******************************
* Here we score initial chromsomes                                                                 * 
***************************************************************************************************/
void
ScoreInitialChromsomes(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

  /* lauch first kernel to score the initial set of chromsomes (Vs_d) and output scores in scores_ds
    betweem the triple chervon is called the execution configuration that takes two parts
    1st part takes the number of thread blocks and the second part take the number of threads in a block */
  scoreIt <<<(parameters.N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (
    parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList],
    parameters.deviceParameters[gpuDevice].areas_d,
    parameters.deviceParameters[gpuDevice].Vs_d,
    parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList],
    parameters.deviceParameters[gpuDevice].tset_d,
    parameters.deviceParameters[gpuDevice].tgts_d,
    parameters.deviceParameters[gpuDevice].wts_d,
    parameters.deviceParameters[gpuDevice].breaks_d,
    parameters.nConf,
    parameters.pSize,
    parameters.genomeSize,
    parameters.deviceParameters[gpuDevice].xx_d
  );

  /* score of chromosomes out of psize since we initiated 2 times psize */
  scoreIt <<<(parameters.N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].areas_d,
      parameters.deviceParameters[gpuDevice].Vs_d,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].tset_d,
      parameters.deviceParameters[gpuDevice].tgts_d,
      parameters.deviceParameters[gpuDevice].wts_d,
      parameters.deviceParameters[gpuDevice].breaks_d,
      parameters.nConf,
      parameters.pSize,
      parameters.genomeSize,
      parameters.deviceParameters[gpuDevice].xx_d);

   if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stscore)\n", cudaGetErrorString(error));}

   /* sort the scores from each chromosome of the initial population */
  thrust::sort_by_key(
    thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]),
    thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.N),
    thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList])
  );
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stsort)\n", cudaGetErrorString(error));}
}

/****************************| Let us begin the iterations through generations |********************

 Genetic algorithm iterations through the number of generations or isolation time 

****************************************************************************************************/

void EvolveGenerations(int gpuDevice, Parameters &parameters) {
  cudaError_t error;

  /* for loop for the generation */
  for (int currentGeneration=0; currentGeneration<parameters.nGen; currentGeneration++) {

    /*************************| Step1: Generate random numbers |****************************************/
    
    // create an array of random numbers (rands_d) used for mutations and crossover where the number of random #s is nRands 
    curandGenerateUniform(parameters.deviceParameters[gpuDevice].gen, parameters.deviceParameters[gpuDevice].rands_d, parameters.nRands);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

    /***| Step2: calculate the probabilities (areas) each individual (chromosome) has of mating |******/
    calcAreas <<<parameters.nBlocks, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList],
      parameters.deviceParameters[gpuDevice].areas_d,
      parameters.deviceParameters[gpuDevice].ptrs_d,
      parameters.pSize, parameters.genomeSize
    );

    /***| Step3:  mate the individuals (chromosomes,Parent[0],[1]) selected for the next generation |***/
    mateIt <<<parameters.nBlocks, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].Vs_d,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList],
      parameters.deviceParameters[gpuDevice].areas_d, 
      getSumAreas(
        parameters.deviceParameters[gpuDevice].areas_d,
        parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList],
        parameters.pSize,
        parameters.deviceParameters[gpuDevice].areas_d+parameters.N,
        parameters.genomeSize),
      parameters.deviceParameters[gpuDevice].rands_d,
      parameters.pCross,
      parameters.pSize,
      parameters.genomeSize
    );
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mate)\n", cudaGetErrorString(error));}

    /*****************| Step4: mutate individuals generated after mating |*****************************/
    mutateIt <<<parameters.nBlocks, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].Vs_d,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].rands_d+parameters.pSize*3,
      parameters.pSize,
      parameters.pMut,
      parameters.maxMut,
      parameters.genomeSize
    );
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mutate)\n", cudaGetErrorString(error));}

    /**************| Step5: Score the individuals to select for the next generation |*******************/
    scoreIt <<<parameters.nBlocks, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].areas_d,
      parameters.deviceParameters[gpuDevice].Vs_d,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].tset_d,
      parameters.deviceParameters[gpuDevice].tgts_d,
      parameters.deviceParameters[gpuDevice].wts_d,
      parameters.deviceParameters[gpuDevice].breaks_d,
      parameters.nConf,
      parameters.pSize,
      parameters.genomeSize,
      parameters.deviceParameters[gpuDevice].xx_d);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (score)\n", cudaGetErrorString(error));}

    /*****| Step6: Sort the scored chromosomes (individuals) & select for mating for next generation |**/
    moveEm <<<(parameters.save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList^1],
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList^1],
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList],
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList],
      parameters.save
    );
    moveEm <<<(parameters.pSize+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList^1]+parameters.save,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList^1]+parameters.save,
      parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize,
      parameters.pSize
    ); //nOffspring);
    moveEm <<<(parameters.pSize-parameters.save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (
        parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList^1]+parameters.save+parameters.pSize,
        parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList^1]+parameters.save+parameters.pSize,
        parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.save,
        parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.save,
        parameters.pSize-parameters.save);

    // switch the index so we swap the buffers: input becomes output and output becomes input
    parameters.deviceParameters[gpuDevice].curList ^= 1;

    /* first sort only the ones that aren't going to be saved (elitist) */
    thrust::sort_by_key(
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.save),
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize+parameters.save),
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.save)
    );

    /* then sort all those that fit within pSize */
    thrust::sort_by_key(
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]),
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList]+parameters.pSize),
      thrust::device_pointer_cast(parameters.deviceParameters[gpuDevice].ptrs_ds[parameters.deviceParameters[gpuDevice].curList])
    );

    /****************************************************************************************************
    * Here you can print the score of chromosomes (total is 2 x population size) for each generation    *
    *   by uncommenting the if and end DEBUG statement, need to make this an input option               *
    *   such as -s which will mean print scores                                                         *
    ****************************************************************************************************/
    //peng --> print every n generation, make a user option
    //ncp --> number of chromosomes to print, make a user option as well
    //if generation is divisable by peng
    if (currentGeneration % parameters.peng == 0) {
      std::ofstream scorefile;
      scorefile.open (parameters.scoreFile.c_str(), std::ios::out | std::ios::app); //it append to the writeout so make sure u delete scores file
      scorefile << "#Generation" << std::setw(14) << "Chromosomes" << std::setw(12) << "Scores\n";
      cudaMemcpy(parameters.scores, parameters.deviceParameters[gpuDevice].scores_ds[parameters.deviceParameters[gpuDevice].curList], sizeof(*parameters.scores)*parameters.N, cudaMemcpyDeviceToHost);
      //cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
      //cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);

      if (parameters.ncp > parameters.pSize) {
      printf("Parmfile error: ncp should be smaller than psize! \n");
      std::abort();
      }
      for(int m = 0; m< parameters.ncp; m++){
        scorefile << std::setw(6) << currentGeneration << std::setw(14) << m << std::setw(18) << parameters.scores[m] << "\n";
        //scorefile << "Score: " << scores[m] << "\n";
        //for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it)
      }
      scorefile.close();
    }

  } // here the loop for generations ends
}

int
main(int argc, char *argv[]) {
  Parameters parameters;
  std::map<std::string,DihCorrection> correctionMap;
  int maxGpuDevices = 2;

  ParseArgs(argc, argv, parameters);

  LoadParameters(parameters, correctionMap);

  for (int device = 0; device < maxGpuDevices; device++)
    parameters.deviceParameters[device].curList=0;

  for (int device = 0; device < maxGpuDevices; device++)
    AllocateArrays(device, parameters);

  for (int device = 0; device < maxGpuDevices; device++)
    CopyArrays(device, parameters);

  for (int device = 0; device < maxGpuDevices; device++)
    GenerateRandom(device, parameters);

  for (int device = 0; device < maxGpuDevices; device++)
    LoadAmplitudeParameters(device, parameters);

  return 0;
}
