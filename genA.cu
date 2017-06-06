/*********************************************||********************************************
                               Genetic algorithm optimizer
                               genA.cu
   Runs iterations of a genetic algoirthm to optimize molecular mechanics dihedral parameters

   @author James Maier and edits Kellon Belfon 
   @lab Carlos Simmerling lab, Stony Brook University
   @version 2.0 2016 Aug 1
**********************************************||********************************************/


/*******************************************************************************************
 	                   ---------------LOAD LIBRARIES-------------
*******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
/*#undef __GLIBCXX_ATOMIC_BUILTINS
#undef __GLIBCXX_USE_INT128
#define _GLIBCXX_GTHREAD_USE_WEAK 0 */
#include <list>
#include <map>
#include "load.cpp"
#include "parse.cpp"
using namespace std;

/* specifying # of threads for a given block, 256 block threads (index 0 to 255) */
  const int BLOCK_SIZE=256;

//#define HANDLE_ERROR(x) x;error=cudaGetLastError();if(error!=cudaSuccess){printf("CUDA error: %s\n", cudaGetErrorString(error));exit(-1);}

#define HANDLE_ERROR(x) x;

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
    int j, from=0;
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

 * @brief introduces mutations to the genomes in Vs, according to probability pMut, with a max perturbation of max
 *
 * @param Vs a global array of all the parent and child genomes
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
   @param rands array of random numbers
 * @param pSize number of individuals in the population
 * @param pMut probability that a mutation occurs, evaluated for each gene
 * @param max maximum perturbation to an allele
 * @param genomeSize number of genes in a genome
*************************************************************************************************/

__global__ void mutateIt(float *Vs, int *ptrs, const float *rands, const int pSize, const float pMut, const float max, const int genomeSize)
{
/* figure out index */
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<pSize){
    // get index into random number array
    int r=i*genomeSize;
    i=ptrs[i];
    int j=i+genomeSize;
    // want random numbers from [-max, max). will subtract max later
    float scale=2.0f*max/pMut;
    // iterate through genome
    while(i<j){
      if(rands[r]<pMut){
        // mutate the amplitude by adding perturbation
        Vs[i]+=rands[r]*scale-max;
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
/******************************| function 5 ends |***********************************************/

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


/*
///////////////////////////////////////////////////////                       `
//////////////////////////////////                                             `
/////////////////////                                                        |   | 
/////////////                                                            ~ ~ ~ ~ ~ ~ ~
////////                                                                |              |
/////                                                               ____|              |____  
///                                                                |                        | 
//                                                              ___|          J.M           |___
/                                                              |              K.B               |
/                                     PROGRAM BEGINS HERE      |                                |
**************************************************************************************************/

/*************************************************************************************************
argc is a vairable with the number of arguments passed to GenA
argv is a vector of strings representing the the arguments the GenA takes
To run genA:
./genA -p parmfile < inputfile > outputfile 
parameters in the parmfile
psize: population size, 1000-2000 
nGen: number of generations, > 100000 
pMut: probability of mutation, 0.01 - 0.001
max: maximal permissible mutation, 0.5 - 0.001
pCross: probability of crossover 0.8-0.9 
randomseed: sixdigit random number, upon mutation, the lowest bit of the random number is 
used to determine whether the amplitude or the phase shift will change.
input file: parametersfitting data using the following format:
_____________________________________________________________________        
-<dihedral> <AMBER atom type for dihedral 1>                         |
-<dihedral> <AMBER atom type for dihedral 2>                         |
<name of data set> <dihedral 1> <dihedral 2>                         |
 <dihedral 1 value> <dihedral 2 value> <E_QM> <E_MM>                 |
 <dihedral 1 value> <dihedral 2 value> <E_QM> <E_MM>                 |
                    ...                                              |
/                                                                    | 
<name of data set> <dihedral 1> <dihedral 2>                         |
 <dihedral 1 value> <dihedral 2 value> <E_QM> <E_MM>                 |
 <dihedral 1 value> <dihedral 2 value> <E_QM> <E_MM>                 |  
                   ...                                               |
/                                                                    |  
_____________________________________________________________________|
<dihedral> is the name of dihedral e.g phi, psi, chi1, chi2, chi3, etc
<AMBER atom type for dihedral 1> e.g chi1 is N -CX-2C-2C for Met, get from frcmod file
<name of data set> is any name, e.g Metalpha, Metbeta, Metcharge
<dihedral 1 value> this is the dihedral value (deg) of the optimized QM structures e.g 105.62
<E_QM> the QM energy of conformation i with restraint dihedral
<E_MM> the MM energy of conformation i with restraint dihedral with zeroed dihedral parameters in the frcmod
... repeat for all conformations within a break 
/ (refer to as break (brk))
a break seperate conformations that are different e.g alpha backbone, beta backbone, charge amino acids
                                  GOODLUCK!!!
                                  [ O    O ]
                                  [    b ' ]
                                  [  ----- ]
contact: kellonbelfon@gmail.com with genA title for help
********************************************************************************************************/

int main(int argc, char *argv[]){

/* load genA parameters, see above */
  cudaError_t error;
  if (!(argv[1]=="-p")) std::cout << "please use -p for param file";
  ConfigFile cfg(argv[2]);
  // check if keys exixt
  if (!(cfg.keyExists("pSize"))) std::cout << "oops you forgot pSize"; //make it for others  
  //add the rest of parameters if exist as line above 
  // Retreive the value of keys pSize if key dont exist return value will be 1
  //add new parameters here
  int pSize = cfg.getValueOfKey<int>("pSize", 1);
  std::cout << "Population Size (pSize): " << pSize << "\n\n";
  int nGen = cfg.getValueOfKey<int>("nGen", 1);
  std::cout << "Number of Generations (nGen): " << nGen << "\n\n";
  float pMut = cfg.getValueOfKey<float>("pMut", 1);
  std::cout << "Probability of Mutations (pMut): " << pMut << "\n\n";
  float max = cfg.getValueOfKey<float>("max", 1);
  std::cout << "Maximal permissible mutation (max): " << max << "\n\n";
  float pCross = cfg.getValueOfKey<float>("pCross", 1);
  std::cout << "Probability of crossover (pCross): " << pCross << "\n\n";
  int rseed = cfg.getValueOfKey<int>("rseed", 1);
  std::cout << "Random seed (rseed): " << rseed << "\n\n";
  int peng  = cfg.getValueOfKey<int>("peng", 1);
  std::cout << "Print scores every  " << peng << "generations (peng)\n\n";
  int ncp  = cfg.getValueOfKey<int>("ncp", 1);
  std::cout << "Print scores of only " << ncp << " chromosomes every peng \n\n";

/* initializing CPU variables and arrays */
  int genomeSize, g, N, nConf=0, save=pSize/10;
  float *rands, *Vs, *tset, *tgts, *wts, *scores; 
  int *ptrs, *breaks, nBreaks;

/* initializing GPU variables and arrays */
  float *rands_d;
  size_t nRands;cuda 
  curandGenerator_t gen;
  int *ptrs_d, *breaks_d;
  float *Vs_d, *tset_d, *tgts_d, *wts_d, *xx_d, *scores_d, *areas_d;
  
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


/***************************| load data from load.cpp |******************************************
*                    Initializing host data(Normally the 2nd step)                              *
*  check load.cpp for this section                                                              *
*  map is a way to create a dictionary, correction map is an array with key                     * 
************************************************************************************************/
/* initiating container with key and values name correctionMap */
  std::map<std::string,DihCorrection> correctionMap;
  std::cout << "Input file loaded ('_')" << std::endl;
/* load in arrays generated from load.cpp, check it out for further comments 
& specifies the addrress, loading the variables that contain address of another variable 
correctionMap is .....   */
  load(std::cin, &tset, &tgts, &wts, &nConf, &breaks, &nBreaks, &genomeSize, correctionMap);

/*******************************| memory allocation |********************************************
*************************************************************************************************/
/* first cudaMalloc to initialize the CUDA subsystem
cudaMalloc allocates size bytes of linear memory on the device and returns in *devPtr
a pointer to the allocated memory. It takes two parameters:
 (1) devPtr - Pointer to allocated device memory e.g variable &breaks_d that have the address of the
the variable breaks_d (stored in memory)
 (2) size - Requested allocation size in bytes, which is nBreaks
*/

#if DEBUG && 0
  for(int i=0;i<nConf;i++){
    for(int j=0;j<genomeSize;j++)
      std::cerr << ' ' << tset[i*genomeSize+j];
    std::cerr << std::endl;
  }
  std::cerr << tgts[0] << ' ' << tgts[1] << ' ' << tgts[2] << ' ' << tgts[3] << std::endl;
  std::cerr << "first cudaMalloc, " << nBreaks << " breaks" << std::endl;
#endif

/* we are allocating space on the GPU to store four arrays (breaks_d, tgts_d, wts_d, tset_d)
with size specified below. The size (# of elements the array can hold, which is directly 
related to memory to store each element in the array) of the array on the GPU is a lot larger
than the host array at this point in the algorithm. Later we will add results to these arrays.
*/
  cudaMalloc((void **)&breaks_d, nBreaks*sizeof(int));
  cudaMalloc((void **)&tgts_d, (nBreaks-1+nConf*(1+genomeSize))*sizeof(float));
  wts_d=tgts_d+nConf;
  tset_d=wts_d+nBreaks-1;

/********************************| transfer data from host to device |****************************
*  copy data arrays from CPU host (breaks,tset,tgts,wts) to device GPU (breaks_d, etc)           *
*************************************************************************************************/

#if DEBUG
  std::cerr << "COPY" << std::endl;
#endif
/* 
copying over the arrays from the CPU to GPU
nbreaks is the # of dataset + 1. e.g if you are doing alpha and beta backbone set then nbreaks=3
genomesize is the # of fitting dihedral * periodicity, e.g 3 set of dihedral * 4 periodicity = 12
nconf is the # of conformations you are fitting
tset is (E_QMi-E_MMi) + (E_MMref-E_QMref) for each conformation, which = nconf, see load.cpp
tgts is the cos(dih*periodicity) for 4 periodicity for a dihedral for each conformation
so 20 conf will give tgts of 20 (nconf) * 12 (# of dih * periodicity) = 120 
*/
  cudaMemcpy(breaks_d, breaks, nBreaks*sizeof(breaks[0]), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(tset_d, tset, nConf*genomeSize*sizeof(float), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(tgts_d, tgts, nConf*sizeof(float), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(wts_d, wts, (nBreaks-1)*sizeof(*wts), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

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
  nRands=(3+genomeSize)*pSize;
  int nBlocks=(pSize+BLOCK_SIZE-1)/BLOCK_SIZE;

#ifdef DEBUG
  std::cerr << nRands << "nRands\n";
  std::cerr << nBlocks << " blocks\n";
#endif

/*******************************| initializing more host and device variables|************************
*         N (bitwise operation below) is the pSize (1st input) multiply by 2;                   *
*       initiating the chromosomes  which have the solns                                        *
************************************************************************************************/
#if DEBUG
  printf("Allocate memory\n");
#endif

  rands=(float *)malloc(nRands*sizeof(float));
  //cudaMalloc((void **)&rands_d, nRands*sizeof(float));
  N=(pSize<<1);
  HANDLE_ERROR(cudaMalloc((void **)&Vs_d, (N*(genomeSize+4)+pSize*nConf+nRands)*sizeof(float)));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  rands_d=Vs_d+N*genomeSize;
  scores_d=rands_d+nRands;
  areas_d=scores_d+(N<<1);
  xx_d=areas_d+(N<<1);
  scores=(float *)malloc(sizeof(*scores)*N);
  float *scores_ds[2];
  scores_ds[0]=scores_d;
  scores_ds[1]=scores_d+N;

  Vs=(float *)malloc(N*genomeSize*sizeof(float));
  /*allocate the memory space to hold array of pointers (prts) of size N (2*pSize)
  these pointers point to the individuals (chromosome) in the population */
  ptrs=(int *)malloc(sizeof(int)*N);
  ptrs[0]=0;
  for(g=1;g<N;g++)ptrs[g]=ptrs[g-1]+genomeSize;
  HANDLE_ERROR(cudaMalloc((void **)&ptrs_d, N*2*sizeof(int)));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  int *ptrs_ds[2];
  ptrs_ds[0]=ptrs_d;
  ptrs_ds[1]=ptrs_d+N;
  cudaMemcpy(ptrs_d, ptrs, sizeof(int)*N, cudaMemcpyHostToDevice);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  int curList=0;

#if 0
  HANDLE_ERROR(cudaMalloc((void **)&scores_d, N*sizeof(float)));

  HANDLE_ERROR(cudaMalloc((void **)&xx_d, nOffspring*nConf*sizeof(float)));
#endif

/*
for CUDA beginners on thrust
thrust is a c++ template library for CUDA similar to STL
it have two containers: thrust::host_vector<type> and thrust::device_vector<type>
the containers make common operations such as cudaMalloc, cudaFree, cudaMemcpy, more concise
 e.g thrust::host_vector<int> vec_h(2) will allocate host vector with 2 elements
     thrust::device_vectore<int> vec_d = vec_h will copy host vector to device
this will allow you to directly manipulate device values from the host
     so vec_d[0] = 5; can be done from host 
and once you output vector memory is automatically released 
    std::cout << "my vector" << vec_d[0] << std::endl;
it have a few algorithms, we use thrust::sort(), 
*/

  thrust::device_ptr<int> dPtrs(ptrs_d), dPtrs_save(ptrs_d+save);
  thrust::device_ptr<float> dScores(scores_d), dVs(Vs_d);
  thrust::device_ptr<float> dScores_save(scores_d+save),
                            dScores_pSize(scores_d+pSize),
                            dScores_N(scores_d+N);



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

#if DEBUG
  printf("Create random generator\n");
#endif

  // create the generator name gen
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

#if DEBUG
  printf("Seed random generator\n");
#endif
  // initiate the generator with the random seed (rseed)
  curandSetPseudoRandomGeneratorSeed(gen, rseed);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (seed)\n", cudaGetErrorString(error));}
#if DEBUG
   std::cerr << "GenerateNormal" << std::endl;
#endif
    curandGenerateNormal(gen, Vs_d, N*genomeSize, 0, 1);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (normal)\n", cudaGetErrorString(error));}
/***************************| END of random generator part |***************************************/

  // if we have a load file copy Vs (amplitude parameters) from the loaded file and populate Vs
  if(!loadFile.empty()) {
    std::ifstream loadS(loadFile.c_str(), std::ios::in | std::ios::binary);
    loadS.read((char*)Vs,pSize*genomeSize*sizeof(*Vs));
    cudaMemcpy(Vs_d, Vs, pSize*genomeSize*sizeof(*Vs), cudaMemcpyHostToDevice);
  }

/* timing event */
  cudaEvent_t events[3];
  int nevents = (sizeof events) / (sizeof events[0]);

  for (int i = 0; i < nevents ; ++i)
    cudaEventCreate(events+i, 0);


  cudaEventRecord(events[0], 0);

/***************************| score of the first set of chromosomes |*******************************
* Here we score initial chromsomes                                                                 * 
***************************************************************************************************/
#if DEBUG
    std::cerr << "1stscore" << std::endl;
#endif
    /* lauch first kernel to score the initial set of chromsomes (Vs_d) and output scores in scores_ds
      betweem the triple chervon is called the execution configuration that takes two parts
      1st part takes the number of thread blocks and the second part take the number of threads in a block */
    scoreIt <<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList], areas_d, Vs_d, ptrs_ds[curList], tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
    /* score of chromosomes out of psize since we initiated 2 times psize */
    scoreIt <<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList]+pSize, areas_d, Vs_d, ptrs_ds[curList]+pSize, tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
  
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stscore)\n", cudaGetErrorString(error));}
#if DEBUG
    std::cerr << "1stsort" << std::endl;
#endif
           /* sort the scores from each chromosome of the initial population */
thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]), thrust::device_pointer_cast(scores_ds[curList]+N), thrust::device_pointer_cast(ptrs_ds[curList]));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stsort)\n", cudaGetErrorString(error));}
         /* option to copy over scores from GPU device to CPU host */
#if DEBUG>2
    cudaMemcpy(scores, scores_ds[curList], sizeof(*scores)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
       /* i is each chromosome, scores[i] is scores, Vs[ptrs[i]] is the amplitude parameters;
         Vs[ptrs[i]]+n specifies the next n amplitude. e.g chromosome i have genomesize amplitude parms 
         e.g  Vs[ptrs[i]]+1 is the amplitude term when the periodicity is 3 for the 1st dihedral being
        fitted, and  Vs[ptrs[i]]+4, the amplitude term when the periodicity is 4 for the 2nd dihedral */
    for(int i=0;i<N;i++){
      std::cerr << i << ": [" << ptrs[i] << "] = " << scores[i] << " {"<<Vs[ptrs[i]]<<" "<<Vs[ptrs[i]+1]<<" "<<Vs[ptrs[i]+2]<<" "<<Vs[ptrs[i]+3]<<"}\n";
    }
#endif

  cudaEventRecord(events[1], 0);

/****************************| Let us begin the iterations through generations |********************

 Genetic algorithm iterations through the number of generations or isolation time 

****************************************************************************************************/
  //std::cout << "There is " << nGen << " generations" << " and " << N << " number of chromosomes (2 x population size)" << std::endl;

  /* for loop for the generation */
  for(g=0;g<nGen;g++){
  
/*************************| Step1: Generate random numbers |****************************************/
#if DEBUG>1
  printf("Generate random numbers\n");
    printf(" %d",g);fflush(stdout);
#endif
    
    // create an array of random numbers (rands_d) used for mutations and crossover where the number of random #s is nRands 
    curandGenerateUniform(gen, rands_d, nRands);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

/***| Step2: calculate the probabilities (areas) each individual (chromosome) has of mating |******/
#if DEBUG>2
    std::cerr << "Mate" << std::endl;
#endif
    calcAreas <<<nBlocks, BLOCK_SIZE>>> (scores_ds[curList], areas_d, ptrs_d, pSize, genomeSize);

/***| Step3:  mate the individuals (chromosomes,Parent[0],[1]) selected for the next generation |***/
    mateIt <<<nBlocks, BLOCK_SIZE>>> (Vs_d, ptrs_ds[curList], areas_d, 
  getSumAreas(areas_d, ptrs_ds[curList], pSize, areas_d+N, genomeSize),
 rands_d, pCross, pSize, genomeSize);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mate)\n", cudaGetErrorString(error));}

/*****************| Step4: mutate individuals generated after mating |*****************************/
#if DEBUG>2
    std::cerr << "Mutate" << std::endl;
#endif
    mutateIt <<<nBlocks, BLOCK_SIZE>>> (Vs_d, ptrs_ds[curList]+pSize, rands_d+pSize*3, pSize, pMut, max, genomeSize);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mutate)\n", cudaGetErrorString(error));}

/**************| Step5: Score the individuals to select for the next generation |*******************/
#if DEBUG>2
    std::cerr << "Score" << std::endl;
#endif
    scoreIt <<<nBlocks, BLOCK_SIZE>>> (scores_ds[curList]+pSize, areas_d, Vs_d, ptrs_ds[curList]+pSize, tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (score)\n", cudaGetErrorString(error));}

#if DEBUG>2
    //std::cerr << "Display em:\n\tCopy scores" << std::endl;
    cudaMemcpy(scores, scores_ds[curList], sizeof(*scores)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    //std::cerr << "\tCopy Vs" << std::endl;
    cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    //std::cerr << "\tCopy ptrs" << std::endl;
    cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    for(int i=0;i<N;i++){
      /* below you can print the scores for a chromosomes every generation */
      std::cout << "This is Generation: " << g << " and Chromosome (set of parameters): " << i << std::endl;
      std::cout << "Score: " << scores[i] << std::endl;
/* below you can print out the scores and the first four barrier parameters,since we are using 
4 periodicity, the first 4 barrier parameters are for the 1st dihedral in the input file */
      //std::cout << i << ": [" << ptrs[i] << "] = " << scores[i] << " {"<<Vs[ptrs[i]]<<" "<<Vs[ptrs[i]+1]<<" "<<Vs[ptrs[i]+2]<<" "<<Vs[ptrs[i]+3]<<"}\n";
     std::cerr << i << ": [" << ptrs[i] << "] = " << scores[i] << " {"<<Vs[ptrs[i]]<<" "<<Vs[ptrs[i]+1]<<" "<<Vs[ptrs[i]+2]<<" "<<Vs[ptrs[i]+3]<<"}\n";
    }
#endif

/*****| Step6: Sort the scored chromosomes (individuals) & select for mating for next generation |**/
#if DEBUG>2
    std::cerr << "Move 1" << std::endl;
#endif
    moveEm <<<(save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1], ptrs_ds[curList^1], scores_ds[curList], ptrs_ds[curList], save);
#if DEBUG>2
    std::cerr << "Move 2" << std::endl;
#endif
    moveEm <<<(pSize+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1]+save, ptrs_ds[curList^1]+save, scores_ds[curList]+pSize, ptrs_ds[curList]+pSize, pSize);//nOffspring);
#if DEBUG>2
    std::cerr << "Move 3" << std::endl;
#endif
    moveEm <<<(pSize-save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1]+save+pSize, ptrs_ds[curList^1]+save+pSize, scores_ds[curList]+save, ptrs_ds[curList]+save, pSize-save);
    curList^=1;

    /* first sort only the ones that aren't going to be saved (elitist) */
#if DEBUG>1
    std::cerr << "Selection sort (" << N << " items, less " << save << ")" << std::endl;
#endif
thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]+save), thrust::device_pointer_cast(scores_ds[curList]+pSize+save), thrust::device_pointer_cast(ptrs_ds[curList]+save));

    /* then sort all those that fit within pSize */
#if DEBUG>1
    std::cerr << "Rank sort" << std::endl;
#endif
    thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]), thrust::device_pointer_cast(scores_ds[curList]+pSize), thrust::device_pointer_cast(ptrs_ds[curList]));

/****************************************************************************************************
* Here you can print the score of chromosomes (total is 2 x population size) for each generation    *
*   by uncommenting the if and end DEBUG statement, need to make this an input option               *
*   such as -s which will mean print scores                                                         *
****************************************************************************************************/
    //peng --> print every n generation, make a user option
    //ncp --> number of chromosomes to print, make a user option as well
    //if generation is divisable by peng
    if(g%peng==0) {
      std::ofstream scorefile;
      scorefile.open (scoreFile.c_str(), ios::out | ios::app); //it append to the writeout so make sure u delete scores file
      scorefile << "#Generation" << std::setw(14) << "Chromosomes" << std::setw(12) << "Scores\n";
      cudaMemcpy(scores, scores_ds[curList], sizeof(*scores)*N, cudaMemcpyDeviceToHost);
      //cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
      //cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);

      for(int m=0;m<ncp;m++){
        scorefile << std::setw(6) << g << std::setw(14) << m << std::setw(18) << scores[m] << "\n";
        //scorefile << "Score: " << scores[m] << "\n";
        //for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it){
      }
      scorefile.close();
    }

  } // here the loop for generations ends

  cudaEventRecord(events[2], 0);

/****************************************************************************************************
*    TERMINATION, LAST RESULTS< SCORES AND PARAMETERS FOR EACH INDIVIDUAL
****************************************************************************************************/
 
/*  copy over the end result from GPU to the CPU to save the scores and parameters */
  cudaMemcpy(Vs, Vs_d, sizeof(float)*genomeSize*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(int)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(scores, scores_ds[curList], sizeof(float)*N, cudaMemcpyDeviceToHost);

  for(int i=0;i<pSize;i++){
/* these are the final scores for each individual in the population, print in the output file  */
  std::cout << std::fixed << scores[i] << std::endl;

  for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it){
/* second.setGenome(Vs+ptrs[i]) is the dihedral parameters for each individual in the population 
   print in the output file                                                                     */
    std::cout << it->second.setGenome(Vs+ptrs[i]);
  }
  }
  if(!saveFile.empty()){
    std::ofstream saveS(saveFile.c_str(), std::ios::out | std::ios::binary);
    for(int i=0;i<pSize;i++)
      saveS.write((char *)(Vs+ptrs[i]),genomeSize*sizeof(*Vs));
  }

  cudaEventSynchronize(events[nevents-1]);

  float elapsedTimeInit, elapsedTimeCompute;

  cudaEventElapsedTime(&elapsedTimeInit, events[0], events[1]);
  cudaEventElapsedTime(&elapsedTimeCompute, events[1], events[2]);

  std::cout << "Initialization time: " << elapsedTimeInit * 1e-3 << std::endl;
  std::cout << "Computation time: " << elapsedTimeCompute * 1e-3 << std::endl;

#if 0
  std::cout << scores[pSize] << std::endl;
  for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it){
    std::cout << it->second.setGenome(Vs+ptrs[pSize]);
    //std::cout << it->second;
  }
#endif

  free(ptrs);

#if 0
  printf("Copy random numbers\n");
  cudaMemcpy(rands, rands_d, nRands*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("Print random numbers\n");
  printf("%d",rands[0]);
  for(i=1;i<nRands;i++){
    printf(" %d",rands[i]);
  }
  putchar('\n');
#endif

/*****************| Free up GPU Memory |*******************************************************/
  curandDestroyGenerator(gen);
  cudaFree(Vs_d);
  cudaFree(ptrs_d);
  cudaFree(breaks_d);
  cudaFree(tgts_d);
  free(Vs);
  free(scores);
  free(rands);
  return 0;
}
