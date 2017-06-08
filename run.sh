#!/bin/bash
for psize in 2000;do
for ngen in 5000 100000;do
cat > parmfile.$psize.$ngen << EOF
pSize=$psize;
nGen=$ngen;
pMut=0.01
max=0.5
pCross=0.8
rseed=314568
peng=500
ncp=1
iTime=500
nEx=10
EOF
sbatch << EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -A hackathon
#SBATCH --reservation=hackathon
#SBATCH -p long
#SBATCH --gres=gpu:4
###SBATCH --exclusive
#SBATCH -t 5:00
#SBATCH -o %j-psize-${psize}-ngen-${ngen}-1-gpu.out
#SBATCH -e %j-psize-${psize}-ngen-${ngen}-1-gpu.err
hostname
#nvprof -o nvout 
./genAmultigpu -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < large.input 

#rm -f parmfile.$psize.$ngen
#rm -f scores.${psize}.${ngen}.dat
EOF
done
done
