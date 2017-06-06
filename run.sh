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
ncp=$psize
EOF
sbatch << EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A hackathon
#SBATCH --reservation=hackathon
#SBATCH -p long
#SBATCH --gres=gpu:1
###SBATCH --exclusive
#SBATCH -t 30:00
#SBATCH -o %j-psize-${psize}-ngen-${ngen}-1-gpu.out
#SBATCH -e %j-psize-${psize}-ngen-${ngen}-1-gpu.err

#nvprof --print-gpu-trace ./genA -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < input
#nvprof --cpu-profiling on --cpu-profiling-max-depth 10 --cpu-profiling-mode top-down --cpu-profiling-thread-mode separated -f -o small-profile.nvprof ./genA -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < input
./genA -p parmfile.${psize}.${ngen} -s scores.${psize}.${ngen}.dat < input
rm -f parmfile.$psize.$ngen
rm -f scores.${psize}.${ngen}.dat
EOF
done
done
