# hackdpotato
Excerpt of AMBER code for genetic algorithm optimization on GPU

# Basic Usage

```
./genA -p parmfile -r try.log -s scores.dat < input > try.frcmod

-p has the parmfile 
-r is to save restart file 
-c is the option to load a restart file
-s is the option to save score file 
```

# Examples


# Parser

`parse.cpp` usesi a simple parser of configuration files from
http://www.dreamincode.net/forums/topic/183191-create-a-simple-configuration-file-parser/
