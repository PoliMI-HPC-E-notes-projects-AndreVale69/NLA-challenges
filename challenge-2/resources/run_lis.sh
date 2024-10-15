#!/bin/bash
source /u/sw/etc/profile
module load gcc-glibc
module load lis
cd ../challenge-2/resources || cd ../resources || exit
./eigen1 ATA.mtx eigvec.mtx hist.txt -e pi -etol 1.0e-8 > result.txt
