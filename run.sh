#!/bin/bash
for psize in 2000;do
for ngen in 5000;do
cat > parmfile.$psize.$ngen << EOF
pSize=$psize;
nGen=$ngen;
pMut=0.01
max=0.5
pCross=0.8
rseed=314568
peng=$ngen
ncp=$ngen
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
./genA -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < input > try.${psize}.${ngen}.frcmod
rm parmfile.$psize.$ngen
EOF
done
done
