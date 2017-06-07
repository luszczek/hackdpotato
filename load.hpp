
class DihCorrection {
};

extern void load(std::istream & in, float **tset, float **tgts, float **wts, int *nConfs, int **brks, int *nBrks, int *genomeSize, std::map<std::string,DihCorrection> & correctionMap);
