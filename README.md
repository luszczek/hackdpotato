# hackdpotato
Excerpt of AMBER code for genetic algorithm optimization on GPU

# Basic Usage

./genA 10 20 0.01 0.5 0.8 313897 -r try.log < input > try.frcmod
./genA 10 20 0.005 0.5 0.8 313897 -c try.log -r try1.log < input > try.frcmod
./genA 10 20 0.002 0.5 0.8 313897 -c try1.log -r try2.log < input > try.frcmod
./genA 10 20 0.002 0.001 0.8 313897 -c try2.log -r try3.log < input > try.frcmod

./genA -pSize 10 -nGen 20 -pMut 0.01 -max 0.5 -pCross 0.8 -rseed 313897 -r try.log < input > try.frcmod

http://www.dreamincode.net/forums/topic/183191-create-a-simple-configuration-file-parser/
for parse.cpp
