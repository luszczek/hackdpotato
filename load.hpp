#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

class DihCorrection {
private:
// maps periodicity to column
std::map<int,int> corrs;
std::string label;
const float *genome;
public:
DihCorrection() {}
DihCorrection(std::string label) : label(label) {
}
void addCorr(int n, int col) {
  corrs[n]=col;
}
std::map<int,int> getCorrs() { return corrs; }
const DihCorrection & setGenome(const float *genome) {
  this->genome=genome;
  return *this;
}
friend std::ostream & operator<<(std::ostream &os, const DihCorrection &dc);
};


//class DihCorrection {
//};


extern void load(std::istream & in, float **tset, float **tgts, float **wts, int *nConfs, int **brks, int *nBrks, int *genomeSize, std::map<std::string,DihCorrection> & correctionMap);
