#!/bin/bash
for psize in 2000;do
for ngen in 50000 100000;do
cat > parmfile.$psize.$ngen << EOF
pSize=$psize;
nGen=$ngen;
pMut=0.01
max=0.5
pCross=0.8
rseed=314568
peng=5000
ncp=5000
EOF
sbatch << EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A hackathon
#SBATCH --reservation=hackathon
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -o logfile.psize${psize}.ngen${ngen}.1gpu

nvprof -o output.$psize.$ngen.nvp ./genA -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < input > try.${psize}.${ngen}.frcmod
rm parmfile.$psize.$ngen
rm scores.${psize}.${ngen}.dat
EOF
done
done
