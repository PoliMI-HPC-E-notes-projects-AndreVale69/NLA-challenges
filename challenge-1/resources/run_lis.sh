#!/bin/bash
source /u/sw/etc/profile
module load gcc-glibc
module load lis
cd ../challenge-1/resources || cd ../resources || exit
./test1 A2.mtx w.mtx sol.mtx hist.txt -tol 1.0e-9 -i cgs -p 8 -maxiter 100 > result.txt
