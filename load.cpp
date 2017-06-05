#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>


/* dihedral parameters in AMBER are: barrier, phase and periodicity
the format for these parameters in AMBER are:
atom type defn  divider barrier term    phase       periodicity
X -CA-CC-X          1       2.800       180.0         -2.000
The phase can be 0, meaning that a maximum energy is encountered at zero degrees.
Or it can be 180 degrees, meaning there is a minimum at 180 degrees
equation: Vbarrier/divider × [1+cos(periodicity×φ − phase)]
A "negative" periodicity (-2 in the case above), tells programs reading the parameter
file that additional terms are present for that particular connectivity
private only class DihCorrection is aware and public is where everything
has access so label and genome is accessible from all other function or class */

class DihCorrection {
private:
/*
maps periodicity to column
In this version we are only using 4 periodicity 
label, is the string from input file denoting the dihedral name(e.g chi1, chi2, etc)
what is DihCorrection function
*/

std::map<int,int> corrs;
std::string label;
const float *genome;
public:
DihCorrection() {}
DihCorrection(std::string label) : label(label) {

}
/* n is the key of the 4 periodicity (0 1 2 3),
col is the value of (periodicity x # of dihedral) we are fitting
so for three dihedral (chi1,2,3) then col is index 0 to 11 */
void addCorr(int n, int col) {
  corrs[n]=col;
}
std::map<int,int> getCorrs() { return corrs; }
const DihCorrection & setGenome(const float *genome) {
  /* this is address of genome, means genome=genome */
  this->genome=genome;
  /* return the address of genome, need address to access values of genome */
  return *this;
}
friend std::ostream & operator<<(std::ostream &os, const DihCorrection &dc);
};

std::ostream & operator<<(std::ostream &os, const DihCorrection &dc) {
/* iterating through dc.corrs, see std::map tutorial for further explanation, it is the iterator */
  for(std::map<int,int>::const_iterator it=dc.corrs.begin(), next=it;it!=dc.corrs.end();it=next){
    ++next;
    os << dc.label << std::fixed;
    if(dc.label.length()<=11)os << std::setw(8) << 1;
    float val=dc.genome[it->second];
/* it->first is the key in dc.corrs() */
    os << std::setw(12) << (val<0?-val:val) << std::setw(8) << (val<0?180:0) << std::setw(5) << (next==dc.corrs.end()?it->first:-it->first) << std::endl;
  }
  return os;
}

void load(std::istream & in, float **tset, float **tgts, float **wts, int *nConfs, int **brks, int *nBrks, int *genomeSize, std::map<std::string,DihCorrection> & correctionMap)
{
  int ch;
  int col=0;
  while(in.peek()=='+'||in.peek()=='-'){
    std::string label;
    std::string dih;
    ch=in.get();
    int strLen=(ch=='-'?11:35);
    in >> label;
    in >> std::ws;
    for(int i=0;i<strLen;i++){
      dih.push_back(in.get());
    }
/* dih is the atom type for the dihedral (e.g CX-2C-2C-S ) 
label is the dihedral name (e.g chi2) so Dihedral[chi2]=CX-2C-2C-S */
#if LOAD_DEBUG
    std::cout << "Dihedral[" << label  << "]=" << dih << std::endl;
#endif
    DihCorrection dc(dih);
    while((ch=in.get())==' ');
    if(ch=='\n'){
      for(int n=4;n>0;n--) {
/* print out n:col which is 4:0, 3:1, ....0;11 for three dihedral, see above */
#if LOAD_DEBUG
        std::cout << n << ": " << col << std::endl;
#endif
        dc.addCorr(n, col++);
      }
#if LOAD_DEBUG
      std::cout << dc.getCorrs().size() << std::endl;
#endif
    } else {
      while(ch!='\n'){
        if(ch<'0'||ch>'9'){
          std::string otherLabel;
          do {
            otherLabel.push_back(ch);
            ch=in.get();
          /* while ch is 0 or 9 */
          } while(ch<'0'||ch>'9');
          dc.addCorr(ch-'0',correctionMap[otherLabel].getCorrs()[ch-'0']);
        } else {
          dc.addCorr(ch-'0', col++);
        }
        do ch=in.get();while(ch==' ');
      }
    }
    correctionMap[label]=dc;
    //std::map<std::string,DihCorrection>::iterator it=correctionMap.find(label);
    //if(it!=)
  }
  *genomeSize=col;
  std::vector<std::vector<float> > data;
  std::vector<int> breaks;
  std::vector<float> weights;
  *nConfs=0;
  std::string line;
  double off;
  while(in.good()&&std::getline(in,line)){
    breaks.push_back(*nConfs);
    std::list<DihCorrection*> cols;
    std::string label;
    std::istringstream input(line);
    input >> label;
#if LOAD_DEBUG
    std::cout << "Residue=" << label << std::endl;
#endif
    weights.push_back(1.0f);
    if(input.good()){
     input >> label;
     /* atof converts string to double 
     label is from input file, e.g chi1 and label[0] is the first element in label which is c*/
     if(label[0]=='<') weights.back()=atof(label.data()+1);
     else { cols.push_back(&correctionMap[label]); }
     while(input.good()){
      input >> label;
      cols.push_back(&correctionMap[label]);
     }
    }
    std::vector<float> dataRow;
    while(std::getline(in,line)&&line[0]!='/'){
      input.clear();
      input.str(line);
      /* assignment of the dataRow to be 1 + cosine term */
      dataRow.assign(1+*genomeSize, 0);
      double dih;
      for(std::list<DihCorrection*>::iterator it=cols.begin();it!=cols.end();++it){
        input >> dih;
#if LOAD_DEBUG
        /* dih is the dihedral values located in input value */
        std::cout << dih << ":";
#endif
       /* convert degrees from input file (-180 to 180) to radians(-pi to pi) 
        dih*=pi/180 is same as dih=dih * pi/180 */
        dih*=3.141592653589793238/180.;
#if LOAD_DEBUG
        std::cout << (*it)->getCorrs().size() << std::endl;
#endif

#if 0
        for(std::map<int,int>::iterator jt=(*it)->getCorrs().begin();jt!=(*it)->getCorrs().end();++jt){
        //for(const auto& jt:(*it)->getCorrs()){
#if LOAD_DEBUG
          std::cout << " " << jt->first << "[" << jt->second << "]+=" << cos(dih*(float)jt->first);
#endif
          dataRow[jt->second]+=cos(dih*(float)jt->first);
        }
#endif

        /* four periodicity */
        for(int n=4;n>0;--n){
#if LOAD_DEBUG
       /* *it is the pointer to the correction map 
        n[col] = cos(dihedral* periodicity), where n is the periodicty and 
        col is an index for the periodicity, and dihedral in radians; 
        so for a given dihedral the cosine term is evaluated for the four
        periodicity (n), so four cos terms for a given dihedral*/
          std::cout << "dih=" << dih << std::endl; 
          std::cout << " " << n << "[" << (*it)->getCorrs()[n] << "]+=" << cos(dih*(float)n);
#endif
          dataRow[(*it)->getCorrs()[n]]+=cos(dih*(double)n);
          /* dataRow[(*it)->getCorrs()[n]] is now the cos term */ 
        }
#if LOAD_DEBUG
        /* (*it)->getCorrs().size() return the number of elements in *it which is 4 */
        std::cout << ' ' << (*it)->getCorrs().size() << std::endl;
#endif
      }
      /*  this section calculates deltaE btwn QM and MM for a given structure using the first 
          structure in the break as reference
          off = E (MM ref) - E0 (QM ref), where ref is the first QM & MM energy within the break
          delta E = 0 for the first structure 
          delta E = (E (QMi) - E (MMi) ) + off where i is every structure after the break
          E is the column with QM energies, E0 is column with MM energies*/
      double E,E0;
      /* input E and E0 */
      input >> E >> E0;
      /* the if statement evaluate the first conf after the break, else statement everything else */
      if(*nConfs==breaks.back()){
        off=E0-E;
        E=0;
      }else{
        E=E-E0+off;
      }
      /*  *genomeSize in dataRow = delta E */
      dataRow[*genomeSize]=(float)E;

#if LOAD_DEBUG
    std::cout << " deltaE="<<dataRow[*genomeSize]<<std::endl;
#endif

      /* for every conformation, push_back will add dataRow to the array data
         dataRow contains the deltaE and the cosine term of 4 periodicity per dihedral */
      ++*nConfs;
      data.push_back(dataRow);
    }
    /* weights here is 1 / nconf * (nconf-1)/2 for a given population in a break */
    weights.back()/=(float)((*nConfs-breaks.back())*(*nConfs-breaks.back()-1)/2);
  }
  breaks.push_back(*nConfs);
  *nBrks=breaks.size();

#if LOAD_DEBUG
  /* print the number of conformations (structures) and breaks */
  std::cout << *nConfs << " confs and " << *nBrks << " breaks" << std::endl;
#endif

  /* sizeof(x), number of bytes to represent type x 
     *tgts is the delta E for a given conformation (structure) 
     *tset is the cos(periodicity * dihedrals) terms for a given angle and periodicity
       e.g three dihedral with 4 periodicity will give 12 tset for a give structure that
           has the three dihedrals
      The idea if you have a structure that have a delta E of 0.4 kcal/mol. Then using 
      amber energy Vbarrier/divider × [1+cos(periodicity×dihedral − phase)]
      parameters Vbarrier and phase will be searched with the genA.cu to give a energy
      (sum of zeroed dihedral) as close to the target energy of 0.4 kcal/mol 

      *brks is the conformation position for the program to identify the breaks
        e.g 20 conf, 10 alpha, 10 beta, will have 3 breaks (break seperate alpha and beta)
            so break will be 0, 10, 20
      *wts is 1 / nconf * (nconf-1)/2 for a given population in a break 
        e.g 20 conf, 10 alpha, 10 beta will have two dataset alpha and beta
            so for alpha wts = 1/(10)*((10-1)/2) = 1/45 = 0.0222
            so for beta wts = 1/(10)*((10-1)/2) = 1/45 = 0.0222 */

  *brks=(int *)malloc(sizeof(**brks)*breaks.size());
  *wts=(float *)malloc(sizeof(**wts)*weights.size());
  *tset=(float *)malloc(sizeof(**tset)* *nConfs* *genomeSize);
  *tgts=(float *)malloc(sizeof(**tgts)* *nConfs);
  for(int i=0;i<*nConfs;i++){
    (*tgts)[i]=data[i][*genomeSize];
    //std::cout << data[i][*genomeSize] << " *tgts" << std::endl;
    for(int j=0;j<*genomeSize;j++){
      (*tset)[i* *genomeSize+j]=data[i][j];
    //std::cout << (*tset)[i* *genomeSize+j] << " *tset" << std::endl;
    }
  }
  for(int i=0;i<weights.size();i++){
    (*brks)[i]=breaks[i];
    (*wts)[i]=weights[i];
    //std::cout << (*brks)[i] << " *brks" << std::endl;
    //std::cout << (*wts)[i] << " *weight" << std::endl;
  }
  for(int i=weights.size();i<*nBrks;i++){
    (*brks)[i]=breaks[i];
    //std::cout << (*brks)[i] << " *brks" << std::endl;
  }
}

/* brks, wts, tset, tgts are the parameters loaded in the genetic algorithm */
